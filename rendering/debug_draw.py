import sys
import glm
import numpy as np
from OpenGL.GL import *
import inspect

import log

buffer_cache = {};
rectangle_buffer = None;

def generate_buffer(vertices: list[float], colors: list[float]):
    buffer = glGenVertexArrays(1)
    glBindVertexArray(buffer)

    
    vertex_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer)

    #In una mia ipotetica shader la position e'sempre in posizione 0
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))

    vertices = np.array(vertices, dtype=np.float32)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    
    
    color_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, color_buffer)

    #Come sopra ma per il colore
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))
    colors = np.array(colors, dtype=np.float32)
    glBufferData(GL_ARRAY_BUFFER, colors.nbytes, colors, GL_STATIC_DRAW)

    
    glBindVertexArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    return buffer

def get_unique_id():
    result = []
    frame = sys._getframe(1)
    
    for _ in range(5):
        if frame is None:
            break
        result.append(str(frame.f_lineno))
        frame = frame.f_back
    
    return ".".join(result)

def draw_line(start: glm.vec3, end: glm.vec3, color: glm.vec3 = glm.vec3(1)):
    id = get_unique_id()
    if id not in buffer_cache:
        buffer_cache[id] = generate_buffer(
            [start.x, start.y, start.z, end.x, end.y, end.z], 
            [color.x, color.y, color.z, color.x, color.y, color.z]
        )

    glBindVertexArray(buffer_cache[id])
    glDrawArrays(GL_LINES, 0, 2)
    glBindVertexArray(0)
    return 0;

def draw_rect_xz(center: glm.vec3, size: glm.vec3):
    halfSize = size/2

    #Left Edge
    draw_line(
        center + glm.vec3(-halfSize.x, 0, halfSize.z), 
        center + glm.vec3(-halfSize.x, 0, -halfSize.z) 
    )

    #Right Edge
    draw_line(
        center + glm.vec3(halfSize.x, 0, halfSize.z), 
        center + glm.vec3(halfSize.x, 0, -halfSize.z) 
    )
    
    #Far Edge
    draw_line(
        center + glm.vec3(-halfSize.x, 0, -halfSize.z), 
        center + glm.vec3(halfSize.x, 0, -halfSize.z) 
    )

    
    #Near Edge
    draw_line(
        center + glm.vec3(-halfSize.x, 0, halfSize.z), 
        center + glm.vec3(halfSize.x, 0, halfSize.z) 
    )

def draw_rect_yx(center: glm.vec3, size: glm.vec3):
    halfSize = size/2

    #Left Edge
    draw_line(
        center + glm.vec3(-halfSize.x, -halfSize.y, 0), 
        center + glm.vec3(-halfSize.x, halfSize.y, 0) 
    )

    #Right Edge
    draw_line(
        center + glm.vec3(halfSize.x, -halfSize.y, 0), 
        center + glm.vec3(halfSize.x, halfSize.y, 0) 
    )

    #Top Edge
    draw_line(
        center + glm.vec3(-halfSize.x, halfSize.y, 0), 
        center + glm.vec3(halfSize.x, halfSize.y, 0) 
    )

    #Bot Edge
    draw_line(
        center + glm.vec3(-halfSize.x, -halfSize.y, 0), 
        center + glm.vec3(halfSize.x, -halfSize.y, 0) 
    )

def draw_rect_zy(center: glm.vec3, size: glm.vec3):
    halfSize = size/2

    #Near Edge
    draw_line(
        center + glm.vec3(0, -halfSize.y, halfSize.z), 
        center + glm.vec3(0, halfSize.y, halfSize.z) 
    )

    #Far Edge
    draw_line(
        center + glm.vec3(0, -halfSize.y, -halfSize.z), 
        center + glm.vec3(0, halfSize.y, -halfSize.z) 
    )

    #Top Edge
    draw_line(
        center + glm.vec3(0, halfSize.y, halfSize.z), 
        center + glm.vec3(0, halfSize.y, -halfSize.z) 
    )

    #Bot Edge
    draw_line(
        center + glm.vec3(0, -halfSize.y, halfSize.z), 
        center + glm.vec3(0, -halfSize.y, -halfSize.z) 
    )

def draw_box(center: glm.vec3, size: glm.vec3):
    halfSize = size/2

    draw_rect_xz(glm.vec3(0, center.y + halfSize.y, 0), size)
    draw_rect_xz(glm.vec3(0, center.y - halfSize.y, 0), size)

    draw_rect_yx(glm.vec3(0, 0, center.z - halfSize.z), size)
    draw_rect_yx(glm.vec3(0, 0, center.z + halfSize.z), size)

    draw_rect_zy(glm.vec3(center.x - halfSize.x, 0, 0), size)
    draw_rect_zy(glm.vec3(center.x + halfSize.x, 0, 0), size)