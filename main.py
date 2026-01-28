import math
import os
import sys
from enum import IntEnum

import pygame
import pymeshlab
from PIL import Image, ImageOps

from pyglm import glm

import imgui
from imgui.integrations.pygame import PygameRenderer

import log
import application_config as config
from rendering import shader, texture, debug_draw
from rendering.fbo import Fbo
from rendering.renderable import *

import arcball
import metashape_loader

OUTPUT_PATH = r"S:\ritm_output"

class RenderMode(IntEnum):
    LABEL_ONLY = 0
    MIXED = 1
    TEXTURE_ONLY = 2

class ViewMode(IntEnum):
    FREE = 0
    CAMERA = 1
    ORTHO = 2

def create_buffers_frame(frame_shader):
    """Create buffer for XYZ lines. Returns generated VAO's id."""

    # Create a new VAO (Vertex Array Object) and bind it
    vertex_array_object = glGenVertexArrays(1)
    glBindVertexArray( vertex_array_object )
    
    # Generate buffers to hold our vertices
    vertex_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer)
    
    # Get the position of the 'position' in parameter of our shader and bind it.
    position = glGetAttribLocation(frame_shader.program, 'aPosition')
    glEnableVertexAttribArray(position)
    
    # Describe the position data layout in the buffer
    glVertexAttribPointer(position, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))
    
    verts = [0,0,0, 1,0,0, 0,0,0, 0,1,0, 0,0,0, 0,0,1]
    verts = np.array(verts, dtype=np.float32)

    # Send the data over to the buffer
    glBufferData(GL_ARRAY_BUFFER,verts.nbytes, verts, GL_STATIC_DRAW)
    
    # Generate buffers to hold our vertices
    tcoord_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, tcoord_buffer)
    
    # Get the position of the 'position' in parameter of our shader and bind it.
    position = glGetAttribLocation(frame_shader.program, 'aColor')
    glEnableVertexAttribArray(position)
    
    # Describe the position data layout in the buffer
    glVertexAttribPointer(position, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))
    
    col = [1,0,0, 1,0,0, 0,1,0, 0,1,0, 0,0,1, 0,0,1]
    col = np.array(col, dtype=np.float32)

    # Send the data over to the buffer
    glBufferData(GL_ARRAY_BUFFER,col.nbytes, col, GL_STATIC_DRAW)

    # Unbind the VAO first (Important)
    glBindVertexArray( 0 )
    
    # Unbind other stuff
    glDisableVertexAttribArray(position)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    return vertex_array_object

def create_quad_buffer():
    vertices = [ 1.0, -1.0, 0.0,
                -1.0, -1.0, 0.0,
                -1.0,  1.0, 0.0,

                 1.0,  1.0, 0.0,
                 1.0, -1.0, 0.0,
                -1.0,  1.0, 0.0]

    buffer = glGenVertexArrays(1)
    glBindVertexArray(buffer)

    vertex_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer)

    #In una mia ipotetica shader la position e'sempre in posizione 0
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))

    vertices = np.array(vertices, dtype=np.float32)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    glBindVertexArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    return buffer

def create_mesh_buffers(verts, wed_tcoord, inds, mesh_shader):
    """
        Mesh buffer creation.
        Creates VAO and VBOs for the mesh
    
        - verts: vertex positions
        - wed_tcoord: texture coordinates
        - inds: indices
        - mesh_shader: shader to use
    
    Returns VAO
    """
    vert_pos = np.zeros((len(inds) * 3, 3), dtype=np.float32)
    tcoords = np.zeros((len(inds) * 3, 2), dtype=np.float32)
    
    for i in range(len(inds)):
        vert_pos[i*3] = verts[inds[i, 0]]
        vert_pos[i*3+1] = verts[inds[i, 1]]
        vert_pos[i*3+2] = verts[inds[i, 2]]
        
        tcoords[i*3] = wed_tcoord[i*3]
        tcoords[i*3+1] = wed_tcoord[i*3+1]
        tcoords[i*3+2] = wed_tcoord[i*3+2]
    
    vert_pos = vert_pos.flatten()
    tcoords = tcoords.flatten()
    
    # Create VAO
    vertex_array_object = glGenVertexArrays(1)
    glBindVertexArray(vertex_array_object)
    
    # Vertex positions
    vertex_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer)
    position = glGetAttribLocation(mesh_shader.program, 'aPosition')
    glEnableVertexAttribArray(position)
    glVertexAttribPointer(position, 3, GL_FLOAT, False, 0, None)
    glBufferData(GL_ARRAY_BUFFER, vert_pos.nbytes, vert_pos, GL_STATIC_DRAW)
    
    # Texture coordinates
    tcoord_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, tcoord_buffer)
    position = glGetAttribLocation(mesh_shader.program, 'aTexCoord')
    if not position == -1:
        glEnableVertexAttribArray(position)
        glVertexAttribPointer(position, 2, GL_FLOAT, False, 0, None)
        glBufferData(GL_ARRAY_BUFFER, tcoords.nbytes, tcoords, GL_STATIC_DRAW)
    
    # Unbind
    glBindVertexArray(0)
    #glDisableVertexAttribArray(position)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    
    return vertex_array_object

def load_mesh(filename):
    """ 
        - vertices
        - faces: triangle
        - wed_tcoord: texture coordinates
        - bbox_min: bounding box minimum
        - bbox_max: bounding box maximum
        - texture_id
        - w, h: texture dimensions
    """
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(filename)
    mesh = ms.current_mesh()
    
    # Extract vertices, faces, and texture coordinates
    vertices = mesh.vertex_matrix()
    faces = mesh.face_matrix()
    wed_tcoord = mesh.wedge_tex_coord_matrix()
    
    # Handle texture coordinates
    if mesh.has_wedge_tex_coord():
        ms.apply_filter("compute_texcoord_transfer_wedge_to_vertex")
    
    # Load texture if available
    texture_id = -1
    w, h = 0, 0
    if mesh.textures():
        texture_dict = mesh.textures()
        texture_name = next(iter(texture_dict.keys()))
        texture_name = os.path.join(os.path.dirname(filename), 
                                   os.path.basename(texture_name))
        texture_id, w, h = texture.load_texture(texture_name, GL_LINEAR)
    else:
        texture_name = os.path.join(os.path.dirname(filename), 
                                   os.path.basename('model1.tif'))
        texture_id, w, h = texture.load_texture(texture_name, GL_LINEAR)

        
    # Compute bounding box
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    
    log.print_info(f"Loaded mesh {filename}: \n\
    [ \n\
        vertices: {len(vertices)},\n\
        faces: {len(faces)},\n\
        Bounding box: {bbox_min} to {bbox_max}\n\
    ]\n")
    
    return vertices, faces, wed_tcoord, bbox_min, bbox_max, texture_id, w, h

def set_sensor(shader: shader.Shader, sensor, overscanFactor = 1.2):
    shader.set_int("resolution_width", sensor.resolution["width"])
    shader.set_int("resolution_height", sensor.resolution["height"])

    shader.set_float("f", float(sensor.calibration["f"]))
    shader.set_float("cx", float(sensor.calibration["cx"]))
    shader.set_float("cy", float(sensor.calibration["cy"]))
    shader.set_float("k1", float(sensor.calibration["k1"]))
    shader.set_float("k2", float(sensor.calibration["k2"]))
    shader.set_float("k3", float(sensor.calibration["k3"]))
    shader.set_float("p1", float(sensor.calibration["p1"]))
    shader.set_float("p2", float(sensor.calibration["p2"]))
    shader.set_float("b1", 0) #Se li riaggiungo non matcha per nulla
    shader.set_float("b2", 0)

    overscan_width = int(sensor.resolution["width"] * overscanFactor)
    overscan_height = int(sensor.resolution["height"] * overscanFactor)
    shader.set_vec2("overscanResolution", glm.vec2(overscan_width, overscan_height))

def build_proj_matrix(sensor, near = 1.0, far = 100.0, overscan_factor=1.2):
    w = sensor.resolution["width"] * overscan_factor
    h = sensor.resolution["height"] * overscan_factor
    f = sensor.calibration["f"]  # focal length in pixels

    cx = (sensor.resolution["width"] / 2.0 + sensor.calibration["cx"]) * overscan_factor
    cy = (sensor.resolution["height"] / 2.0 - sensor.calibration["cy"]) * overscan_factor

    fx = f + sensor.calibration["b1"] #Ho fatto un po' di test se lo includo qui ho un match migliore
    return glm.mat4(
        2 * fx / w, 0, 0, 0,
        0, 2 * f / h, 0, 0,
        (w - 2 * cx) / w, (h - 2 * cy) / h, -(far + near) / (far - near), -1,
        0, 0, -2 * far * near / (far - near), 0
    )

def main():
    config.init()
    glm.silence(4)
    Image.MAX_IMAGE_PIXELS = 181159576

    #Window context variables
    W, H = 1350, 900
    SCREEN = None
    CLOCK = None
    DELTA_TIME = 0

    #Setup PyGame
    pygame.init()
    pygame.display.set_caption("Polyp Detector")
    SCREEN = pygame.display.set_mode((W, H), pygame.OPENGL|pygame.DOUBLEBUF)
    CLOCK = pygame.time.Clock()

    # Initialize ImGui
    imgui.create_context()
    IMGUI_RENDERER = PygameRenderer()
    imgui.get_io().display_size = (W, H)

    log.print_info(f"OpenGL Version: {glGetString(GL_VERSION).decode()}")
    log.print_info(f"GLSL Version: {glGetString(GL_SHADING_LANGUAGE_VERSION).decode()}\n")

    #Load Sensors & Cameras
    sensors = metashape_loader.load_sensors_from_xml( config.METASHAPE_FILE )
    log.print_info(f"Loaded {len(sensors)} sensors.\n")
    for i in range(0, len(sensors)):
        log.print_info(f"[{i}] => {{ \n")
        log.print_info(f"{sensors[i]}\n")
        log.print_info(f"}}\n")
    
    cameras, chunk_rot, chunk_transl, chunk_scal = metashape_loader.load_cameras_from_xml( config.METASHAPE_FILE )
    chunk_rot = np.array(chunk_rot)
    chunk_transl = np.array(chunk_transl)

    log.print_info(f"    Loaded { len(cameras) } cameras,\n\
    Chunk Rotation:\n\
        {chunk_rot}\n\
    ,\n\
    Chunk Translation:\n\
        {chunk_transl}\n\
    ,\n\
    Chunk Scale:\n\
        {chunk_scal}\n\
    ,\n\n\
    ")

    #Load shaders
    SHADER_MAIN = shader.load_shader_from_files("main") #shader.Shader(VERTEX_SHADER, FRAGMENT_SHADER)
    SHADER_FRAME = shader.load_shader_from_files("frame")
    SHADER_QUAD = shader.load_shader_from_files("quad")

    #Create full-screen quad
    screen_quad = create_quad_buffer()

    #Load mesh
    vertices, faces, wed_tcoords, bbox_min, bbox_max, texture_id, tex_w, tex_h = load_mesh( config.MESH_FILE )
    rend = renderable(
        vao = create_mesh_buffers(vertices, wed_tcoords, faces, SHADER_MAIN),
        n_verts = len(vertices),
        n_faces = len(faces),
        texture_id = texture_id,
        mask_id = None
    )
    
    #Calculate chunk matrix
    mat4_np = np.eye(4)
    mat4_np[:3, :3] = chunk_rot.reshape(3, 3)
    chunk_rot_matrix =  glm.transpose(glm.mat4(*mat4_np.flatten()))
    chunk_tra_matrix =  glm.translate(glm.vec3(*chunk_transl))
    chunk_sca_matrix =  glm.scale(glm.vec3(chunk_scal))
    chunk_matrix : glm.mat4x4 = chunk_tra_matrix * chunk_sca_matrix * chunk_rot_matrix
    
    #Camera
    projection_matrix = glm.perspective(glm.radians(45), W / H, 0.0001, 100)

    arcBall = arcball.ArcballCamera(W, H)
    center = (bbox_min+bbox_max)/2.0
    center = glm.vec3(center[0], center[1], center[2])

    #Load camera frame
    camera_frame_vao = create_buffers_frame(SHADER_FRAME)
    center_frame_matrix = chunk_matrix * glm.translate(center)
    #center_frame_matrix = glm.mat4(1.0)

    arcBall.set_center(center)
    arcBall.set_distance(1)

    ortho_proj: glm.mat4 = glm.ortho(-2.7859610141394064, 2.8135035058605933, -2.365373768699055, 2.3856638113009452, 0.01, 10) #ortho.extents
    
    ortho_center =  glm.vec3(5.0721218747120318, 0.3702069405071875, -7.9174685381193006) #? #ortho.projection.translation
    ortho_view = glm.lookAt(ortho_center + glm.vec3(0, 0, 1), ortho_center, glm.vec3(0, 1, 0))

    #Calculate all camera matrices
    camera_matrices : list[glm.mat4x4] = [glm.mat4] * len(cameras)
    for i in range(0, len(cameras)):
        camera_matrices[i] = chunk_matrix * glm.transpose(glm.mat4(*cameras[i].transform)) * glm.rotate(
            glm.radians(180), glm.vec3(0, 1, 0)) * glm.rotate(glm.radians(180), glm.vec3(0, 0, 1))

    """* glm.rotate(glm.radians(180), glm.vec3(0, 1, 0)) * glm.rotate(glm.radians(180), glm.vec3(0, 0, 1))"""

    #Import Label map
    label_map, label_width, label_height = texture.load_texture(os.path.join(config.MAIN_PATH, "TAGLAB", "label.png"), GL_NEAREST, False)

    #Application settings
    view_mode = ViewMode.ORTHO
    render_mode = RenderMode.TEXTURE_ONLY
    selected_camera_id = 0
    selected_photo_id = 0

    show_origin_frame = True
    show_camera_frames = True
    show_debug = True

    OVERSCAN = 1.2
    MAX_CAMERA = math.inf #There are more than 300 cameras view to render, for debugging we can stop sooner
    
    #Set first sensor
    glUseProgram(SHADER_QUAD.program)
    set_sensor(SHADER_QUAD, sensors[cameras[selected_camera_id].sensor_id])
    glUseProgram(0)

    #Set ortho projection
    glUseProgram(SHADER_MAIN.program)
    SHADER_MAIN.set_mat4("uOrthoProj",  ortho_proj)
    SHADER_MAIN.set_mat4("uOrthoView",  ortho_view)
    glUseProgram(0)
    
    #OpenGl settings
    glClearColor(0, 0, 0, 1)
    running = True

    log.WARNING_LOG_ENABLED = False

    ORTHO_FBO = Fbo(label_width, label_height, depthAsTexture=True) #TODO: Il color component in teoria non mi serve
    RENDER_FBO = Fbo(W, H, depthAsTexture=True)

    #Renderizza la orto immagine una volta, verra' poi riutilizzata per "spalmare" la label map correttamente
    glViewport(0, 0, label_width, label_height)
    glBindFramebuffer(GL_FRAMEBUFFER, ORTHO_FBO.id_fbo)
    glClear(int(GL_COLOR_BUFFER_BIT) | int(GL_DEPTH_BUFFER_BIT))
    glEnable(GL_DEPTH_TEST)

    # Setta le shader necessarie ------------------------------
    glUseProgram(SHADER_MAIN.program)
    SHADER_MAIN.set_int("uViewMode", ViewMode.ORTHO)
    SHADER_MAIN.set_int("uRenderMode", RenderMode.TEXTURE_ONLY)
    SHADER_MAIN.set_mat4("uProj", ortho_proj)
    SHADER_MAIN.set_mat4("uView", ortho_view)
    SHADER_MAIN.set_mat4("uModel", glm.mat4(1.0))

    # Render the actual renderable obj off-screen -------------
    glBindVertexArray(rend.vao)
    glDrawArrays(GL_TRIANGLES, 0, rend.n_faces * 3)
    glBindVertexArray(0)

    # Reset ---------------------------------------------------
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glUseProgram(0)
    glViewport(0, 0, W, H)

    while running:
        glClear(int(GL_COLOR_BUFFER_BIT) | int(GL_DEPTH_BUFFER_BIT))
        glEnable(GL_DEPTH_TEST)

        #Handle PyGames&ImGui events ------------------
        for event in pygame.event.get():
            IMGUI_RENDERER.process_event(event)
            
            if event.type == pygame.QUIT:
                running = False
            
            if imgui.get_io().want_capture_mouse:
                continue

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_ESCAPE:
                    running = False

            # Mouse movement - trackball rotation
            if event.type == pygame.MOUSEMOTION:
                mouseX, mouseY = event.pos
                if view_mode == ViewMode.FREE:
                    arcBall.mouse_move(mouseX, mouseY)
                #tb.mouse_move(projection_matrix, view_matrix, mouseX, mouseY)
            
            # Mouse wheel - zoom
            if event.type == pygame.MOUSEWHEEL:
                xoffset, yoffset = event.x, event.y
                if view_mode == ViewMode.FREE:
                    arcBall.set_distance(arcBall.distance - yoffset  * 4 * DELTA_TIME) # pyright: ignore[reportPossiblyUnboundVariable]
                #tb.mouse_scroll(xoffset, yoffset)
            
            # Mouse button
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Not mouse wheel
                    mouseX, mouseY = event.pos
                    
                    if view_mode == ViewMode.FREE:
                        arcBall.mouse_pressed(mouseX, mouseY)
                    #tb.mouse_press(projection_matrix, view_matrix, mouseX, mouseY)
            
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button
                    if view_mode == ViewMode.FREE:
                        arcBall.mouse_release()
                    #tb.mouse_release()
        #----------------------------------------------

        #Imgui ----------------------------------------
        imgui.new_frame()
        if imgui.begin_main_menu_bar().opened:
            if imgui.begin_menu('Actions', True).opened:
                view_mode_changed, view_mode_value = imgui.combo("View Mode", view_mode, ["Free cam", "Sensors", "Ortho"])
                view_mode = view_mode_value
                _, render_mode = imgui.combo("Render Mode", render_mode, ["Label only", "Mixed", "Texture only"])
                selected_camera_id_changed, selected_camera_id = imgui.input_int("Camera ID", selected_camera_id, 1, 100)
                
                if view_mode_changed and view_mode == ViewMode.CAMERA:
                    show_camera_frames = False  #Se vai in modalita' metaashape-camera nascondi in automatico i camera frames

                if selected_camera_id_changed:
                    selected_camera_id = glm.clamp(selected_camera_id, 0, len(cameras) - 1)

                    glUseProgram(SHADER_QUAD.program)
                    set_sensor(SHADER_QUAD, sensors[cameras[selected_camera_id].sensor_id])
                    glUseProgram(0)

                if imgui.button("Render all pseudo-labels"):
                    glUseProgram(SHADER_MAIN.program)
                    SHADER_MAIN.set_int("uRenderMode", RenderMode.LABEL_ONLY)
                    SHADER_MAIN.set_mat4("uModel", glm.mat4(1))
                    glUseProgram(0)

                    length = max(0, min(MAX_CAMERA, len(cameras)))
                    print("Rendering Pseudo-Labels...")
                    print(f"{0}/{length} rendered.")
                    for i in range(0, length):
                        sensor = sensors[cameras[i].sensor_id]

                        sensor_width = sensor.resolution["width"]
                        sensor_height = sensor.resolution["height"]
                        overscan_width = int(sensor_width * OVERSCAN)
                        overscan_height = int(sensor_height * OVERSCAN)

                        OVERSCAN_FBO = Fbo(overscan_width, overscan_height, GL_NEAREST)
                        SAVE_FBO = Fbo(sensor_width, sensor_height, GL_NEAREST)

                        # Disegna il modello sul framebuffer con dell'overscan
                        glBindFramebuffer(GL_FRAMEBUFFER, OVERSCAN_FBO.id_fbo)
                        glViewport(0, 0, overscan_width, overscan_height)
                        # pygame.display.set_mode((overscan_width, overscan_height), pygame.OPENGL|pygame.DOUBLEBUF)

                        glClear(int(GL_COLOR_BUFFER_BIT) | int(GL_DEPTH_BUFFER_BIT))
                        glEnable(GL_DEPTH_TEST)

                        glUseProgram(SHADER_MAIN.program)
                        glActiveTexture(GL_TEXTURE0)
                        glBindTexture(GL_TEXTURE_2D, rend.texture_id)
                        SHADER_MAIN.set_int("uColorTex", 0)

                        glActiveTexture(GL_TEXTURE1)
                        glBindTexture(GL_TEXTURE_2D, label_map)
                        SHADER_MAIN.set_int("uLabelMap", 1)

                        glActiveTexture(GL_TEXTURE2)
                        glBindTexture(GL_TEXTURE_2D, ORTHO_FBO.id_depth)
                        SHADER_MAIN.set_int("uDepthTex", 2)

                        SHADER_MAIN.set_mat4("uProj", build_proj_matrix(sensor, OVERSCAN))
                        SHADER_MAIN.set_mat4("uView", glm.inverse(camera_matrices[i]))

                        glBindVertexArray(rend.vao)
                        glDrawArrays(GL_TRIANGLES, 0, rend.n_faces * 3)
                        glBindVertexArray(0)
                        glUseProgram(0)

                        # Applica il post processing sulla texture del framebuffer
                        glBindFramebuffer(GL_FRAMEBUFFER, SAVE_FBO.id_fbo)
                        glViewport(0, 0, sensor_width, sensor_height)
                        # pygame.display.set_mode((sensor_width, sensor_height), pygame.OPENGL|pygame.DOUBLEBUF)
                        glClear(int(GL_COLOR_BUFFER_BIT) | int(GL_DEPTH_BUFFER_BIT))
                        glDisable(GL_DEPTH_TEST)  # Don't need depth for post-processing

                        glUseProgram(SHADER_QUAD.program)
                        set_sensor(SHADER_QUAD, sensor, OVERSCAN)
                        glBindVertexArray(screen_quad)

                        glActiveTexture(GL_TEXTURE0)
                        glBindTexture(GL_TEXTURE_2D, OVERSCAN_FBO.id_color)
                        glDrawArrays(GL_TRIANGLES, 0, 6)

                        glBindVertexArray(0)
                        glUseProgram(0)

                        # Salva a texture il risultato
                        glBindFramebuffer(GL_FRAMEBUFFER, SAVE_FBO.id_fbo)
                        frameBytes = glReadPixels(0, 0, sensor_width, sensor_height, GL_RGB, GL_UNSIGNED_BYTE, None)
                        result = Image.frombuffer("RGB", (sensor_width, sensor_height), frameBytes, "raw", "RGB", 0, 1)
                        result = ImageOps.flip(result)
                        result.save(os.path.join(config.PSEUDOLABEL_OUTPUT_PATH, "PseudoLabel_" + cameras[i].label + ".png"))

                        print(f"{i+1}/{length} rendered.")

                    #Riporta allo stato precedente
                    glUseProgram(0)
                    glBindFramebuffer(GL_FRAMEBUFFER, 0)
                    glViewport(0, 0, W, H)
                    glEnable(GL_DEPTH_TEST)
                    #pygame.display.set_mode((W, H), pygame.OPENGL|pygame.DOUBLEBUF)

                if imgui.button("Render selected camera"):
                    sensor = sensors[cameras[selected_camera_id].sensor_id]

                    sensor_width = sensor.resolution["width"]
                    sensor_height = sensor.resolution["height"]
                    overscan_width = int(sensor_width * OVERSCAN)
                    overscan_height = int(sensor_height * OVERSCAN)

                    OVERSCAN_FBO = Fbo(overscan_width, overscan_height, GL_NEAREST)
                    SAVE_FBO = Fbo(sensor_width, sensor_height, GL_NEAREST)

                    # Disegna il modello sul framebuffer con dell'overscan
                    glBindFramebuffer(GL_FRAMEBUFFER, OVERSCAN_FBO.id_fbo)
                    glViewport(0, 0, overscan_width, overscan_height)
                    # pygame.display.set_mode((overscan_width, overscan_height), pygame.OPENGL|pygame.DOUBLEBUF)

                    glClear(int(GL_COLOR_BUFFER_BIT) | int(GL_DEPTH_BUFFER_BIT))
                    glEnable(GL_DEPTH_TEST)

                    glUseProgram(SHADER_MAIN.program)
                    glActiveTexture(GL_TEXTURE0)
                    glBindTexture(GL_TEXTURE_2D, rend.texture_id)
                    SHADER_MAIN.set_int("uColorTex", 0)

                    glActiveTexture(GL_TEXTURE1)
                    glBindTexture(GL_TEXTURE_2D, label_map)
                    SHADER_MAIN.set_int("uLabelMap", 1)

                    glActiveTexture(GL_TEXTURE2)
                    glBindTexture(GL_TEXTURE_2D, ORTHO_FBO.id_depth)
                    SHADER_MAIN.set_int("uDepthTex", 2)

                    SHADER_MAIN.set_mat4("uProj", build_proj_matrix(sensor, OVERSCAN))
                    SHADER_MAIN.set_mat4("uView", glm.inverse(camera_matrices[selected_camera_id]))

                    glBindVertexArray(rend.vao)
                    glDrawArrays(GL_TRIANGLES, 0, rend.n_faces * 3)
                    glBindVertexArray(0)
                    glUseProgram(0)

                    # Applica il post processing sulla texture del framebuffer
                    glBindFramebuffer(GL_FRAMEBUFFER, SAVE_FBO.id_fbo)
                    glViewport(0, 0, sensor_width, sensor_height)
                    # pygame.display.set_mode((sensor_width, sensor_height), pygame.OPENGL|pygame.DOUBLEBUF)
                    glClear(int(GL_COLOR_BUFFER_BIT) | int(GL_DEPTH_BUFFER_BIT))
                    glDisable(GL_DEPTH_TEST)  # Don't need depth for post-processing

                    glUseProgram(SHADER_QUAD.program)
                    set_sensor(SHADER_QUAD, sensor, OVERSCAN)
                    glBindVertexArray(screen_quad)

                    glActiveTexture(GL_TEXTURE0)
                    glBindTexture(GL_TEXTURE_2D, OVERSCAN_FBO.id_color)
                    glDrawArrays(GL_TRIANGLES, 0, 6)

                    glBindVertexArray(0)
                    glUseProgram(0)

                    # Salva a texture il risultato
                    glBindFramebuffer(GL_FRAMEBUFFER, SAVE_FBO.id_fbo)
                    frameBytes = glReadPixels(0, 0, sensor_width, sensor_height, GL_RGB, GL_UNSIGNED_BYTE, None)
                    result = Image.frombuffer("RGB", (sensor_width, sensor_height), frameBytes, "raw", "RGB", 0, 1)
                    result = ImageOps.flip(result)
                    result.save(os.path.join(config.PSEUDOLABEL_OUTPUT_PATH, "PseudoLabel_" + cameras[selected_camera_id].label + ".png"))

                #Riporta allo stato precedente
                glUseProgram(0)
                glBindFramebuffer(GL_FRAMEBUFFER, 0)
                glViewport(0, 0, W, H)
                glEnable(GL_DEPTH_TEST)

                imgui.separator()
                _, show_origin_frame = imgui.checkbox("Show origin frame", show_origin_frame)
                _, show_camera_frames = imgui.checkbox("Show camera frames", show_camera_frames)
                _, show_debug = imgui.checkbox("Show debug draw", show_debug)

                imgui.separator()
                imgui.text("- Info --------------------")
                imgui.text(f"Current camera-id: {cameras[selected_camera_id].id}")
                imgui.text(f"Current img name: {cameras[selected_camera_id].label}")
                imgui.end_menu()

            if imgui.begin_menu('Reproject', True).opened:
                selected_photo_id_changed, selected_photo_id = imgui.input_int("Photo ID", selected_photo_id, 1, 100)
                imgui.text(f"Current img name: {cameras[selected_photo_id].label}")

                if imgui.button('Reproject image on model'):
                    label_photo,_,_ = texture.load_texture(os.path.join(config.RITM_OUTPUT_PATH, f"{cameras[selected_photo_id].label}.png"), GL_LINEAR, mipmap = False)

                    if label_photo is None:
                        print(f"{cameras[selected_photo_id].label} is not available")

                    sensor = sensors[cameras[selected_photo_id].sensor_id]

                    sensor_width = sensor.resolution["width"]
                    sensor_height = sensor.resolution["height"]
                    overscan_width = int(sensor_width * OVERSCAN)
                    overscan_height = int(sensor_height * OVERSCAN)

                    OVERSCAN_FBO = Fbo(overscan_width, overscan_height, GL_NEAREST)
                    SAVE_FBO = Fbo(sensor_width, sensor_height, GL_NEAREST)

                    # Disegna il modello sul framebuffer con dell'overscan
                    glBindFramebuffer(GL_FRAMEBUFFER, OVERSCAN_FBO.id_fbo)
                    glViewport(0, 0, overscan_width, overscan_height)
                    # pygame.display.set_mode((overscan_width, overscan_height), pygame.OPENGL|pygame.DOUBLEBUF)

                    glClear(int(GL_COLOR_BUFFER_BIT) | int(GL_DEPTH_BUFFER_BIT))
                    glEnable(GL_DEPTH_TEST)

                    glUseProgram(SHADER_MAIN.program)
                    glActiveTexture(GL_TEXTURE0)
                    glBindTexture(GL_TEXTURE_2D, label_photo)
                    SHADER_MAIN.set_int("uViewMode", ViewMode.CAMERA)
                    SHADER_MAIN.set_int("uReprojectionTex", 0)

                    SHADER_MAIN.set_int("uRenderMode", 3)
                    SHADER_MAIN.set_mat4("uProj", build_proj_matrix(sensor, OVERSCAN))
                    SHADER_MAIN.set_mat4("uView", glm.inverse(camera_matrices[selected_photo_id]))

                    glBindVertexArray(rend.vao)
                    glDrawArrays(GL_TRIANGLES, 0, rend.n_faces * 3)
                    SHADER_MAIN.set_int("uViewMode", view_mode)
                    SHADER_MAIN.set_int("uRenderMode", render_mode)
                    glBindVertexArray(0)
                    glUseProgram(0)

                    # Applica il post processing sulla texture del framebuffer
                    glBindFramebuffer(GL_FRAMEBUFFER, SAVE_FBO.id_fbo)
                    glViewport(0, 0, sensor_width, sensor_height)
                    # pygame.display.set_mode((sensor_width, sensor_height), pygame.OPENGL|pygame.DOUBLEBUF)
                    glClear(int(GL_COLOR_BUFFER_BIT) | int(GL_DEPTH_BUFFER_BIT))
                    glDisable(GL_DEPTH_TEST)  # Don't need depth for post-processing

                    glUseProgram(SHADER_QUAD.program)
                    set_sensor(SHADER_QUAD, sensor, OVERSCAN)
                    glBindVertexArray(screen_quad)

                    glActiveTexture(GL_TEXTURE0)
                    glBindTexture(GL_TEXTURE_2D, OVERSCAN_FBO.id_color)
                    glDrawArrays(GL_TRIANGLES, 0, 6)

                    glBindVertexArray(0)
                    glUseProgram(0)

                    # Salva a texture il risultato
                    glBindFramebuffer(GL_FRAMEBUFFER, SAVE_FBO.id_fbo)
                    frameBytes = glReadPixels(0, 0, sensor_width, sensor_height, GL_RGB, GL_UNSIGNED_BYTE, None)
                    result = Image.frombuffer("RGB", (sensor_width, sensor_height), frameBytes, "raw", "RGB", 0, 1)
                    result = ImageOps.flip(result)
                    result.save(
                        os.path.join(config.MAIN_PATH, "Reprojected_" + cameras[selected_photo_id].label + ".png"))

                imgui.end_menu()

            imgui.end_main_menu_bar()
        #----------------------------------------------

        #Rendering ------------------------------------
        glUseProgram(SHADER_MAIN.program)
        SHADER_MAIN.set_int("uViewMode", view_mode)
        SHADER_MAIN.set_int("uRenderMode", render_mode)

        final_view : glm.mat4 = glm.mat4(1.0)
        match view_mode:
            case ViewMode.FREE:
                SHADER_MAIN.set_mat4("uProj", projection_matrix)
                final_view = arcBall.get_view_matrix()
            
            case ViewMode.CAMERA:
                SHADER_MAIN.set_mat4("uProj", build_proj_matrix(sensors[cameras[selected_camera_id].sensor_id], OVERSCAN))
                final_view = glm.inverse(camera_matrices[selected_camera_id])

            case ViewMode.ORTHO:
                SHADER_MAIN.set_mat4("uProj", ortho_proj)
                final_view = ortho_view

        
        SHADER_MAIN.set_mat4("uView", final_view)
        SHADER_MAIN.set_mat4("uModel", glm.mat4(1.0))

        #Activate renderable obj's texture -------------
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, rend.texture_id)
        SHADER_MAIN.set_int("uColorTex", 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, label_map)
        SHADER_MAIN.set_int("uLabelMap", 1)

        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, ORTHO_FBO.id_depth)
        SHADER_MAIN.set_int("uDepthTex", 2)

        if view_mode == ViewMode.CAMERA:
            glBindFramebuffer(GL_FRAMEBUFFER, RENDER_FBO.id_fbo)
            glClear(int(GL_COLOR_BUFFER_BIT) | int(GL_DEPTH_BUFFER_BIT))
            glEnable(GL_DEPTH_TEST)

            #Render the actual renderable obj off-screen -------------
            glBindVertexArray(rend.vao)
            glDrawArrays(GL_TRIANGLES, 0, rend.n_faces * 3)
            glBindVertexArray(0)

            glBindTexture(GL_TEXTURE_2D, 0)
            glUseProgram(0)
            glBindFramebuffer(GL_FRAMEBUFFER, 0)

            #Render quad with fbo with distortion effect applied -------------
            glUseProgram(SHADER_QUAD.program)
            set_sensor(SHADER_QUAD, sensors[cameras[selected_camera_id].sensor_id], OVERSCAN)
            glBindVertexArray(screen_quad) #In teoria qui dovresti levare il depthTest pero' mi torna utile cosi non disegno tutta la roba di debug

            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, RENDER_FBO.id_color)
            glDrawArrays(GL_TRIANGLES, 0, 6)

            glBindVertexArray(0)
            glUseProgram(0)
        else:
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            #Render the actual renderable obj -------------
            glBindVertexArray(rend.vao)
            glDrawArrays(GL_TRIANGLES, 0, rend.n_faces * 3)
            glBindVertexArray(0)

            glBindTexture(GL_TEXTURE_2D, 0)
            glUseProgram(0)

        #Render debug/handles objects -----------------
        glUseProgram(SHADER_FRAME.program)
        #Set view/proj matrices
        SHADER_FRAME.set_mat4("uProj", projection_matrix)
        SHADER_FRAME.set_mat4("uView", final_view)

        if show_debug:
            SHADER_FRAME.set_mat4("uModel", glm.inverse(ortho_view) * glm.inverse(ortho_proj)) # type: ignore
            debug_draw.draw_box(glm.vec3(0), glm.vec3(2))

        SHADER_FRAME.set_mat4("uModel", center_frame_matrix)
        glBindVertexArray(camera_frame_vao)

        #Draw origin frame
        if show_origin_frame:
            glDrawArrays(GL_LINES, 0, 6)

        #Draw all the camera's frame
        if show_camera_frames:
            SHADER_FRAME.set_mat4("uModel", camera_matrices[selected_camera_id])
            glDrawArrays(GL_LINES, 0, 6)

            SHADER_FRAME.set_mat4("uModel", (camera_matrices[selected_camera_id]) * glm.inverse(build_proj_matrix(sensors[cameras[selected_camera_id].sensor_id]))) # type: ignore
            debug_draw.draw_box(glm.vec3(0), glm.vec3(2))
            """
            for i in range(0,len(cameras)):
                SHADER_FRAME.set_mat4("uModel", camera_matrices[i])
                glDrawArrays(GL_LINES, 0, 6)
            """

        glBindVertexArray(0)
        glUseProgram(0)
        #----------------------------------------------

        #Check for OpenGL errors ----------------------
        check_gl_errors()
        #----------------------------------------------


        #End of frame----------------------------------
        glActiveTexture(GL_TEXTURE0)
        imgui.render()
        IMGUI_RENDERER.render(imgui.get_draw_data())

        pygame.display.flip()
        DELTA_TIME = CLOCK.tick(60) / 1000
        #----------------------------------------------

    return 0

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pygame.quit()
