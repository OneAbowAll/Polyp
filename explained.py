# ============================================================================
# IMPORTS SECTION
# ============================================================================
# Standard libraries for 3D rendering and UI
import os
import sys
import time
import metrics  # Can ignore - used for polyp analysis

import ctypes
from ctypes import c_uint32, cast, POINTER

import numpy as np

import pygame  # Window management and input
from pygame.locals import *

import pymeshlab  # Mesh loading

# OpenGL imports - CORE RENDERING
from OpenGL.GL import glDrawElements
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL.shaders import compileProgram, compileShader

from PIL import Image  # Image loading

import glm  # Math library for 3D transformations
import imgui  # UI library
from imgui.integrations.pygame import PygameRenderer

# Custom modules
import fbo  # Framebuffer objects
import texture  # Texture loading
import trackball  # Camera control
import maskout  # CAN IGNORE - polyp detection stuff
import metashape_loader  # FILE LOADING - important!

from renderable import *  # Renderable object wrapper
from shaders import vertex_shader, fragment_shader, vertex_shader_fsq, fragment_shader_fsq, bbox_shader_str
from plane import fit_plane, project_point_on_plane  # CAN IGNORE
from detector import apply_yolo  # CAN IGNORE - YOLO detection

import pandas as pd  # CAN IGNORE - data export
from collections import Counter


# ============================================================================
# OPENGL BUFFER CREATION FUNCTIONS - RENDERING SETUP
# ============================================================================

def create_buffers_frame():
    """Creates OpenGL buffers for drawing coordinate frame/axes (RGB XYZ lines)"""
    # Creates VAO with position and color buffers for 3 lines representing axes
    vertex_array_object = glGenVertexArrays(1)
    glBindVertexArray(vertex_array_object)
    
    # ... buffer setup code ...
    return vertex_array_object

     
def create_buffers_fsq():
    """Creates buffers for full-screen quad (for 2D image display)"""
    # Creates a simple quad that covers the screen for image overlay
    vertex_array_object = glGenVertexArrays(1)
    # ... buffer setup ...
    return vertex_array_object


def create_buffers(verts, wed_tcoord, inds, shader0):
    """
    MAIN MESH BUFFER CREATION - IMPORTANT FOR RENDERING
    Creates OpenGL VAO and VBOs for the 3D mesh
    - verts: vertex positions
    - wed_tcoord: texture coordinates
    - inds: triangle indices
    - shader0: shader program
    """
    # Expands indexed geometry to flat triangle list
    # Creates vertex, texture coordinate, color, and triangle ID buffers
    # ... detailed buffer setup ...
    return vertex_array_object


# ============================================================================
# 2D IMAGE DISPLAY FUNCTION - Can mostly ignore unless you need 2D overlays
# ============================================================================

def display_image():
    """Displays 2D images with zoom/pan controls"""
    # Renders full-screen quad with texture
    # Handles zoom and translation for image viewing
    # ... rendering code ...


# ============================================================================
# CAMERA/VIEW MATRIX FUNCTIONS - IMPORTANT FOR RENDERING
# ============================================================================

def camera_matrix(id_camera):
    """
    Computes view matrix for a specific camera
    Applies chunk transformations (rotation, translation, scale) from Metashape
    Returns camera_matrix (view matrix) and camera_frame (world transform)
    """
    # Builds transformation from Metashape chunk data
    mat4_np = np.eye(4)
    mat4_np[:3, :3] = chunk_rot.reshape(3, 3)
    chunk_rot_matrix = glm.transpose(glm.mat4(*mat4_np.flatten()))
    chunk_tra_matrix = glm.translate(glm.mat4(1.0), glm.vec3(*chunk_transl))
    chunk_sca_matrix = glm.scale(glm.mat4(1.0), glm.vec3(chunk_scal))
    chunk_matrix = chunk_tra_matrix * chunk_sca_matrix * chunk_rot_matrix
    camera_frame = chunk_matrix * (glm.transpose(glm.mat4(*cameras[id_camera].transform)))
    camera_matrix = glm.inverse(camera_frame)
    return camera_matrix, camera_frame


# Similar FLUO camera functions - Can ignore if not using fluorescence
def camera_matrix_FLUO(id_camera):
    # ... FLUO-specific camera matrix ...
    pass

def chunk_matrix_FLUO(id_camera):
    # ... FLUO-specific chunk matrix ...
    pass


# ============================================================================
# MAIN RENDERING FUNCTION - VERY IMPORTANT
# ============================================================================

def display(shader0, r, tb, detect, get_uvmap):
    """
    CORE 3D RENDERING FUNCTION
    Renders the 3D mesh with camera view
    
    Parameters:
    - shader0: main shader program
    - r: renderable object (contains mesh VAO, texture, etc.)
    - tb: trackball for camera control
    - detect: bool for detection mode (can ignore)
    - get_uvmap: bool for UV mapping (can ignore)
    """
    
    # Set up framebuffer (offscreen or onscreen)
    if detect:
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_uv.id_fbo)
        # ... offscreen rendering setup ...
    else:
        glBindFramebuffer(GL_FRAMEBUFFER, 0)  # Render to screen
        glViewport(0, 0, W, H)

    # Clear and setup
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glUseProgram(shader0.program)

    # Build transformation matrices from Metashape chunk data
    mat4_np = np.eye(4)
    mat4_np[:3, :3] = chunk_rot.reshape(3, 3)
    chunk_rot_matrix = glm.transpose(glm.mat4(*mat4_np.flatten()))
    chunk_tra_matrix = glm.translate(glm.mat4(1.0), glm.vec3(*chunk_transl))
    chunk_sca_matrix = glm.scale(glm.mat4(1.0), glm.vec3(chunk_scal))
    chunk_matrix = chunk_tra_matrix * chunk_sca_matrix * chunk_rot_matrix
    
    # Set view matrix (either user-controlled or from camera)
    if user_camera and not detect:
        # Free camera view with trackball
        view_matrix = user_matrix
        projection_matrix = glm.perspective(glm.radians(45), 1.5, 0.1, 10)
        glUniformMatrix4fv(shader0.uni("uProj"), 1, GL_FALSE, glm.value_ptr(projection_matrix))
        glUniformMatrix4fv(shader0.uni("uTrack"), 1, GL_FALSE, glm.value_ptr(tb.matrix()))
    else:
        # View from specific camera
        camera_frame = chunk_matrix * (glm.transpose(glm.mat4(*cameras[id_camera].transform)))
        view_matrix = glm.inverse(camera_frame)

    # Send matrices to shader
    glUniformMatrix4fv(shader0.uni("uView"), 1, GL_FALSE, glm.value_ptr(view_matrix))
    glUniform1i(shader0.uni("uMode"), user_camera)
    glUniform1i(shader0.uni("uModeProj"), project_image)

    # Set camera sensor parameters
    set_sensor(shader0, sensors[cameras[id_camera].sensor_id])

    # Bind texture (either projected from camera or mesh texture)
    glActiveTexture(GL_TEXTURE0)
    if project_image:
        glBindTexture(GL_TEXTURE_2D, texture_IMG_id)
        camera_matrix_val, _ = camera_matrix(id_camera)
        glUniformMatrix4fv(shader0.uni("uViewCam"), 1, GL_FALSE, glm.value_ptr(camera_matrix_val))
    else:
        glBindTexture(GL_TEXTURE_2D, r.texture_id)

    # DRAW THE MESH
    glBindVertexArray(r.vao)
    glDrawArrays(GL_TRIANGLES, 0, r.n_faces * 3)
    glBindVertexArray(0)

    # Additional rendering (polyps visualization - can ignore this section)
    if not detect:
        # ... polyp rendering code ...
        pass

    # Draw camera frames if in user view mode
    if not detect and user_camera:
        for i in range(0, len(cameras)):
            # Draw coordinate axes for each camera position
            # ... camera frame drawing ...
            pass

    glUseProgram(0)

    # Optional: Display image overlay
    if user_camera == 0 and show_image:
        # ... full-screen image overlay ...
        pass

    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    
    # Rest is polyp-specific output - can ignore
    # ...


# ============================================================================
# 3D POINT PICKING FUNCTION - For interaction
# ============================================================================

def clicked(x, y):
    """
    Converts 2D screen click to 3D world position
    Useful for camera center manipulation
    """
    y = viewport[3] - y
    depth = glReadPixels(x, y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT).item()
    # Unproject screen coords to world space
    mm = np.array(view_matrix * tb.matrix(), dtype=np.float64).flatten()
    pm = np.array(projection_matrix, dtype=np.float64).flatten()
    p = gluUnProject(x, y, depth, mm, pm, np.array(viewport, dtype=np.int32))
    # ... coordinate conversion ...
    return p, depth


# ============================================================================
# FILE LOADING FUNCTIONS - IMPORTANT
# ============================================================================

def load_camera_image(id):
    """Loads a camera image from disk and creates OpenGL texture"""
    global texture_IMG_id
    filename = imgs_path + "/" + cameras[id].label + ".JPG"
    print(f"loading {cameras[id].label}.JPG")
    glDeleteTextures(1, [texture_IMG_id])
    texture_IMG_id, _, __ = texture.load_texture(filename)
    maskout.texture_IMG_id = texture_IMG_id


def load_mesh(filename):
    """
    LOADS 3D MESH - IMPORTANT
    Uses PyMeshLab to load mesh file
    Returns vertices, faces, texture coordinates, bounding box, texture ID, dimensions
    """
    global ms
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(filename)
    mesh = ms.current_mesh()

    # Extract mesh data
    vertices = mesh.vertex_matrix()
    faces = mesh.face_matrix()
    wed_tcoord = mesh.wedge_tex_coord_matrix()
    
    # Handle texture coordinates
    if mesh.has_wedge_tex_coord():
        ms.apply_filter("compute_texcoord_transfer_wedge_to_vertex")

    # Load texture if available
    texture_id = -1
    if mesh.textures():
        texture_dict = mesh.textures()
        texture_name = next(iter(texture_dict.keys()))
        texture_name = os.path.join(os.path.dirname(filename), os.path.basename(texture_name))
        texture_id, w, h = texture.load_texture(texture_name)

    # Compute bounding box
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    
    return vertices, faces, wed_tcoord, bbox_min, bbox_max, texture_id, w, h


# ============================================================================
# POLYP DETECTION FUNCTIONS - CAN IGNORE ALL OF THESE
# ============================================================================

def estimate_range():
    """Polyp detection - can ignore"""
    pass

def process_masks(n):
    """Polyp detection - can ignore"""
    pass

def load_masks(masks_path):
    """Loads mask files for polyp detection - can ignore"""
    pass

def refresh_domain():
    """Updates polyp visualization - can ignore"""
    pass

class polyp:
    """Polyp data structure - can ignore"""
    pass

def estimate_plane(mask):
    """Polyp plane estimation - can ignore"""
    pass

def compute_plane_slantedness(mask):
    """Polyp analysis - can ignore"""
    pass

def compute_avg_fluo(mask):
    """Fluorescence analysis - can ignore"""
    pass

def project_to_3D(pol):
    """Projects polyp to 3D - can ignore"""
    pass

def camera_distance(id_cam_rgb, id_cam_fluo):
    """Camera distance calculation - can ignore"""
    pass

def neighbor_FLUO_cameras(id_cam_rgb):
    """Finds nearest fluorescence camera - can ignore"""
    pass

def fill_polyps():
    """Fills polyp data - can ignore"""
    pass

def compute_bounding_boxes_per_camera():
    """Bounding box computation for detection - can ignore"""
    pass

def segment_imgs():
    """YOLO segmentation - can ignore"""
    pass

def show_node(id_node):
    """Visualization helper - can ignore"""
    pass

def show_comp(id_comp):
    """Visualization helper - can ignore"""
    pass

def export_masks_as_3D():
    """Export function - can ignore"""
    pass

def make_circle(pol, offset):
    """Creates circle geometry for polyp - can ignore"""
    pass

def export_stats():
    """Exports polyp statistics - can ignore"""
    pass

def reset_display_image():
    """Resets image display state - can ignore"""
    pass

def clear_redundants():
    """Removes redundant polyps - can ignore"""
    pass

def quantify_coverage(m):
    """Quantifies mask coverage - can ignore"""
    pass

def count_polyps():
    """Counts detected polyps - can ignore"""
    pass

def set_sensor(shader, sensor):
    """
    Sets camera sensor parameters in shader
    Important for proper camera projection
    """
    glUniform1i(shader.uni("uMasks"), 3)
    glUniform1i(shader.uni("resolution_width"), sensor.resolution["width"])
    glUniform1i(shader.uni("resolution_height"), sensor.resolution["height"])
    glUniform1f(shader.uni("f"), sensor.calibration["f"])
    glUniform1f(shader.uni("cx"), sensor.calibration["cx"])
    glUniform1f(shader.uni("cy"), -sensor.calibration["cy"])
    glUniform1f(shader.uni("k1"), sensor.calibration["k1"])
    glUniform1f(shader.uni("k2"), sensor.calibration["k2"])
    glUniform1f(shader.uni("k3"), sensor.calibration["k3"])
    glUniform1f(shader.uni("p1"), sensor.calibration["p1"])
    glUniform1f(shader.uni("p2"), sensor.calibration["p2"])


# ============================================================================
# MAIN FUNCTION - INITIALIZATION AND MAIN LOOP
# ============================================================================

def main():
    """
    Main entry point
    Handles:
    1. Window/OpenGL initialization
    2. File loading (mesh, cameras, textures)
    3. Main render loop
    4. UI (ImGui)
    5. Input handling
    """
    
    glm.silence(4)

    # Window dimensions
    global W, H
    W = 1200
    H = 800

    # Many global declarations for state management
    # (In a refactor, you'd want to encapsulate these)
    
    # ========== FILE PATHS - IMPORTANT ==========
    # Reads configuration from last.txt or command line
    with open("last.txt", "r") as f:
        content = f.read()
        lines = content.splitlines()
        if len(lines) >= 5:
            main_path = lines[0]          # Project path
            imgs_path = lines[1]           # Camera images path
            masks_path = lines[2]          # Masks path (can ignore)
            mesh_name = lines[3]           # 3D mesh file - IMPORTANT
            metashape_file = lines[4]      # Metashape XML - IMPORTANT
            # Optional fluorescence paths (can ignore)
            # ...

    # ========== PYGAME/OPENGL INITIALIZATION - IMPORTANT ==========
    pygame.init()
    screen = pygame.display.set_mode((W, H), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("Polyp Detector")  # You can rename this
    
    # OpenGL info
    max_ssbo_size = glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE)
    print(f"Max SSBO size: {max_ssbo_size / (1024*1024):.2f} MB")

    # ========== UI INITIALIZATION - IMPORTANT ==========
    imgui.create_context()
    imgui_renderer = PygameRenderer()
    imgui.get_io().display_size = (W, H)

    # ========== TRACKBALL CAMERA CONTROL - IMPORTANT ==========
    tb = trackball.Trackball()
    tb.reset()
    
    # OpenGL setup
    glClearColor(1, 1, 1, 0.0)
    glEnable(GL_DEPTH_TEST)

    # ========== LOAD FILES - IMPORTANT ==========
    app_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(main_path)
    
    # Load 3D mesh
    vertices, faces, wed_tcoords, bmin, bmax, texture_id, texture_w, texture_h = load_mesh(mesh_name)
    
    # Load cameras from Metashape XML - IMPORTANT
    sensors = metashape_loader.load_sensors_from_xml(metashape_file)
    cameras, chunk_rot, chunk_transl, chunk_scal = metashape_loader.load_cameras_from_xml(metashape_file)
    
    # Default chunk transform if not in file
    if chunk_rot is None:
        chunk_rot = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    if chunk_transl is None:
        chunk_transl = [0, 0, 0]
    if chunk_scal is None:
        chunk_scal = 1
    
    chunk_rot = np.array(chunk_rot)
    chunk_transl = np.array(chunk_transl)

    # Create renderable object (wraps VAO, texture, etc.)
    rend = renderable(
        vao=None,
        n_verts=len(vertices),
        n_faces=len(faces),
        texture_id=texture_id,
        mask_id=texture.create_texture(texture_w, texture_h)
    )

    # ========== SHADER COMPILATION - IMPORTANT ==========
    shader0 = shader(vertex_shader, fragment_shader)
    shader_fsq = shader(vertex_shader_fsq, fragment_shader_fsq)
    
    # ========== CREATE OPENGL BUFFERS - IMPORTANT ==========
    vertex_array_object = create_buffers(vertices, wed_tcoords, faces, shader0)
    rend.vao = vertex_array_object
    vao_frame = create_buffers_frame()  # Coordinate axes
    vao_fsq = create_buffers_fsq()      # Full-screen quad

    # ========== CAMERA SETUP - IMPORTANT ==========
    viewport = [0, 0, W, H]
    clock = pygame.time.Clock()
    
    # Initial camera position
    center = (bmin + bmax) / 2.0
    eye = center + glm.vec3(2, 0, 0)
    user_matrix = glm.lookAt(glm.vec3(eye), glm.vec3(center), glm.vec3(0, 0, 1))
    projection_matrix = glm.perspective(glm.radians(45), 1.5, 0.1, 10)
    tb.set_center_radius(center, np.linalg.norm(bmax - bmin) / 2.0)

    # Camera state
    id_camera = 0
    user_camera = 1  # 1 = free camera, 0 = fixed camera view
    detect = False
    
    # ========== MAIN RENDER LOOP - IMPORTANT ==========
    while True:
        time_delta = clock.tick(60) / 1000.0
        
        # ========== EVENT HANDLING - IMPORTANT ==========
        for event in pygame.event.get():
            imgui_renderer.process_event(event)
            
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYUP and event.key == pygame.K_ESCAPE:
                return
                
            # Mouse movement - trackball control
            if event.type == pygame.MOUSEMOTION:
                mouseX, mouseY = event.pos
                if show_image and is_translating:
                    # Image pan mode (can ignore if not using 2D view)
                    mask_xpos = mouseX
                    mask_ypos = mouseY
                else:
                    # 3D trackball rotation
                    tb.mouse_move(projection_matrix, user_matrix, mouseX, mouseY)

            # Mouse wheel - zoom
            if event.type == pygame.MOUSEWHEEL:
                xoffset, yoffset = event.x, event.y
                if show_image:
                    # 2D image zoom (can ignore)
                    mask_zoom = 1.1 if yoffset > 0 else 0.97
                    # ...
                else:
                    # 3D camera zoom
                    tb.mouse_scroll(xoffset, yoffset)
                    
            # Mouse button - rotation/picking
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button not in (4, 5):
                    mouseX, mouseY = event.pos
                    keys = pygame.key.get_pressed()
                    if show_image:
                        # 2D pan start (can ignore)
                        is_translating = True
                        # ...
                    else:
                        # 3D interaction
                        if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
                            # Ctrl+click to set camera center
                            cp, depth = clicked(mouseX, mouseY)
                            if depth < 0.99:
                                tb.reset_center(cp)
                        else:
                            # Regular click-drag rotation
                            tb.mouse_press(projection_matrix, user_matrix, mouseX, mouseY)
                            
            if event.type == pygame.MOUSEBUTTONUP:
                mouseX, mouseY = event.pos
                if event.button == 1:
                    if show_image:
                        is_translating = False
                    else:
                        tb.mouse_release()
                        
            # Keyboard
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    # Toggle between free camera and fixed camera
                    user_camera = 1 - user_camera

        # ========== UI RENDERING (ImGui) - IMPORTANT ==========
        imgui.new_frame()

        if imgui.begin_main_menu_bar().opened:
            if imgui.begin_menu('Actions', True).opened:
                
                # Most of this menu is polyp-specific - can ignore
                # But here are the relevant UI elements you might want:
                
                # Camera selection
                changed_id, id_camera = imgui.input_int("n camera", id_camera)
                if changed_id:
                    id_camera = max(0, min(id_camera, len(cameras)-1))
                    if show_image or project_image:
                        load_camera_image(id_camera)
                
                imgui.text_ansi(f"Curr camera {cameras[id_camera].label}")
                
                # Toggle free camera view
                changed, user_camera = imgui.checkbox("free point of view", user_camera)
                
                # ... lots of polyp-specific UI code you can ignore ...
                
                imgui.end_menu()

        imgui.end_main_menu_bar()

        # ========== RENDERING - IMPORTANT ==========
        if show_image:
            display_image()  # 2D image view
        else:
            display(shader0, rend, tb, False, False)  # 3D mesh view

        # ========== IMGUI FINALIZATION - IMPORTANT ==========
        imgui.render()
        imgui_renderer.render(imgui.get_draw_data())

        # Swap buffers
        pygame.display.flip()
        clock = pygame.time.Clock()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    try:
        main()
    finally:
        pygame.quit()