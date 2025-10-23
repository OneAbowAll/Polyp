import os
import sys
import pygame
import pymeshlab
import numpy as np
from OpenGL.GL import *

import glm
import imgui
from imgui.integrations.pygame import PygameRenderer

import arcball
import log
import shader
import texture
import trackball
from shaders import VERTEX_SHADER, FRAGMENT_SHADER
from renderable import *

DBG_FILEPATH = "C:/Users/dadod/Downloads/test2/model.ply"

def create_buffers_frame(shader):
    """Create buffer for XYZ lines. Returns generated VAO's id."""

    # Create a new VAO (Vertex Array Object) and bind it
    vertex_array_object = glGenVertexArrays(1)
    glBindVertexArray( vertex_array_object )
    
    # Generate buffers to hold our vertices
    vertex_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer)
    
    # Get the position of the 'position' in parameter of our shader and bind it.
    position = glGetAttribLocation(shader.program, 'aPosition')
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
    position = glGetAttribLocation(shader.program, 'aColor')
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

def create_mesh_buffers(verts, wed_tcoord, inds, shader):
    """
        Mesh buffer creation.
        Creates VAO and VBOs for the mesh
    
        - verts: vertex positions
        - wed_tcoord: texture coordinates
        - inds: indices
        - shader: shader to use
    
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
    position = glGetAttribLocation(shader.program, 'aPosition')
    glEnableVertexAttribArray(position)
    glVertexAttribPointer(position, 3, GL_FLOAT, False, 0, None)
    glBufferData(GL_ARRAY_BUFFER, vert_pos.nbytes, vert_pos, GL_STATIC_DRAW)
    
    # Texture coordinates
    tcoord_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, tcoord_buffer)
    position = glGetAttribLocation(shader.program, 'aTexCoord')
    glEnableVertexAttribArray(position)
    glVertexAttribPointer(position, 2, GL_FLOAT, False, 0, None)
    glBufferData(GL_ARRAY_BUFFER, tcoords.nbytes, tcoords, GL_STATIC_DRAW)
    
    # Unbind
    glBindVertexArray(0)
    glDisableVertexAttribArray(position)
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
        texture_id, w, h = texture.load_texture(texture_name)
    
    # Compute bounding box
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    
    log.print_info(f"Loaded mesh {filename}: \n\
        [ \n\
            vertices: {len(vertices)},\n\
            faces: {len(faces)},\n\
            Bounding box: {bbox_min} to {bbox_max}\n\
        ]")
    
    return vertices, faces, wed_tcoord, bbox_min, bbox_max, texture_id, w, h

def load_filepaths():
    """ 
        Try to read the filepaths from sys.argv or last.txt.\n
        Output:
        - main_path
        - imgs_path
        - mesh_name
        - metashape_file 
    """
    main_path = imgs_path = mesh_name = metashape_file = ""
    
    if(len(sys.argv) == 4):
        main_path = sys.argv[1]
        imgs_path = sys.argv[2]
        mesh_name = sys.argv[3]
        metashape_file = sys.argv[4]
    else:
        with open("last.txt", "r") as f:
            lines = f.read().splitlines()
            if len(lines) >= 5:
                main_path = lines[0]
                imgs_path = lines[1]
                mesh_name = lines[3]
                metashape_file = lines[4]
            else:
                print("[ERROR] last.txt does not contain enough lines.")

    log.print_info(f"Main path: {main_path}")
    log.print_info(f"Images path: {imgs_path}")
    log.print_info(f"Mesh: {mesh_name}")
    log.print_info(f"Metashape file: {metashape_file}")
    return main_path, imgs_path, mesh_name, metashape_file

def main():
    glm.silence(4)

    #Load filepaths
    MAIN_PATH, IMGS_PATH, MESH_NAME, METASHAPE_FILE = load_filepaths()

    #Window context variables
    W, H = 1200, 800
    SCREEN = None
    CLOCK = None
    DELTA_TIME = 0

    #Setup PyGame
    pygame.init()
    pygame.display.set_caption("Polyp Detector")
    SCREEN = pygame.display.set_mode((W, H), pygame.OPENGL|pygame.DOUBLEBUF)
    CLOCK = pygame.time.Clock()

    log.print_info(f"OpenGL Version: {glGetString(GL_VERSION).decode()}")
    log.print_info(f"GLSL Version: {glGetString(GL_SHADING_LANGUAGE_VERSION).decode()}")

    #OpenGl settings
    glClearColor(0, 0, 0, 1)
    glEnable(GL_DEPTH_TEST)

    # Initialize ImGui
    imgui.create_context()
    IMGUI_RENDERER = PygameRenderer()
    imgui.get_io().display_size = (W, H)

    #Debug tests
    main_shader = shader.load_shader_from_files("main") #shader.Shader(VERTEX_SHADER, FRAGMENT_SHADER)
    
    vertices, faces, wed_tcoords, bbox_min, bbox_max, texture_id, tex_w, tex_h = load_mesh(DBG_FILEPATH)
    rend = renderable(
        vao=None,
        n_verts=len(vertices),
        n_faces=len(faces),
        texture_id=texture_id,
        mask_id=None
    )

    mesh_vao = create_mesh_buffers(vertices, wed_tcoords, faces, main_shader)
    rend.vao = mesh_vao
    
    #Camera 
    projection_matrix = glm.perspective(glm.radians(45), 1.5,0.1,10)

    arcBall = arcball.ArcballCamera(W, H)
    center = (bbox_min+bbox_max)/2.0
    center = glm.vec3(center[0], center[1], center[2])
    arcBall.set_center(center, 1)
    check_gl_errors()

    running = True
    while running:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        #Handle PyGames&ImGui events ------------------
        for event in pygame.event.get():
            IMGUI_RENDERER.process_event(event)

            if event.type == pygame.QUIT:
                running = False;

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_ESCAPE:
                    running = False;

            # Mouse movement - trackball rotation
            if event.type == pygame.MOUSEMOTION:
                mouseX, mouseY = event.pos
                arcBall.mouse_move(mouseX, mouseY)
                #tb.mouse_move(projection_matrix, view_matrix, mouseX, mouseY)
            
            # Mouse wheel - zoom
            if event.type == pygame.MOUSEWHEEL:
                xoffset, yoffset = event.x, event.y
                arcBall.set_distance(arcBall.distance + yoffset * 5 * DELTA_TIME)
                #tb.mouse_scroll(xoffset, yoffset)
            
            # Mouse button
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Not mouse wheel
                    mouseX, mouseY = event.pos
                    arcBall.mouse_pressed(mouseX, mouseY)
                    #tb.mouse_press(projection_matrix, view_matrix, mouseX, mouseY)
            
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button
                    arcBall.mouse_release()
                    #tb.mouse_release()
        #----------------------------------------------

        #Imgui ----------------------------------------
        imgui.new_frame()
        if imgui.begin_main_menu_bar().opened:
            if imgui.begin_menu('Actions', True).opened:
                imgui.text("Test")
                imgui.end_menu()

            imgui.end_main_menu_bar()
        #----------------------------------------------

        #Rendering ------------------------------------
        glUseProgram(main_shader.program)
        
        #Set view/proj matrices
        main_shader.set_mat4("uProj", projection_matrix)
        final_view = arcBall.get_view_matrix()
        main_shader.set_mat4("uView", final_view)

        #Activate renderable obj's texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, rend.texture_id)

        #Render the actual renderable obj
        glBindVertexArray(rend.vao)
        glDrawArrays(GL_TRIANGLES, 0, rend.n_faces * 3)
        glBindVertexArray(0)
    
        glBindVertexArray(0)

        glUseProgram(0)
        #----------------------------------------------

        #End of frame----------------------------------
        imgui.render()
        IMGUI_RENDERER.render(imgui.get_draw_data())

        pygame.display.flip()
        DELTA_TIME = CLOCK.tick(60) / 1000
        #----------------------------------------------

    #Stuff to do before close
    log.print_debug(f"Test {W}:{H}")
    log.print_warning(f"Test {W}:{H}")
    log.print_info(f"Test {W}:{H}")
    return 0;

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pygame.quit()
