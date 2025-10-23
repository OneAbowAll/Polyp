from OpenGL.GL import glDrawElements
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

def check_gl_errors():
    """Check for OpenGL errors and print them if any exist."""
    error = glGetError()
    if error != GL_NO_ERROR:
        print(f"OpenGL Error: {gluErrorString(error).decode('utf-8')}")


class renderable:
     def __init__(self, vao, n_verts, n_faces,texture_id,mask_id):
         self.vao = vao
         self.n_verts = n_verts
         self.n_faces = n_faces
         self.texture_id = texture_id
         self.mask_id = mask_id
         
     vao = None #vertex array object
     n_verts = None
     n_faces = None

    

