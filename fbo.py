from OpenGL.GL import *
from OpenGL.raw.GL.NV.packed_depth_stencil import GL_DEPTH_STENCIL_NV


class Fbo:
    def __init__(self, w, h, sample_type = GL_LINEAR):
        self.w = w
        self.h = h
        self.sample_type = sample_type

        self.id_fbo = -1
        self.id_color = -1
        self.id_depth = -1
        self.create(w, h)
         
    def create(self,w, h):
        """
        Creates a frame buffer object (FBO) with a float32 texture target.

        Parameters:
            w (int): Width of the frame buffer.
            h (int): Height of the frame buffer.

        Returns:
            tuple: (framebuffer ID, texture ID, renderbuffer ID)
        """
        # Generate FBO
        self.id_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.id_fbo)

        # Generate texture for FBO
        self.id_color = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.id_color)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, self.sample_type)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, self.sample_type)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        # Attach texture to FBO
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.id_color, 0)

        self.id_depth = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.id_depth)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, w, h)

        # Attach renderbuffer to FBO
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, self.id_depth)

        # Check FBO completeness
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            print("Error: Framebuffer is not complete!")
            glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # Unbind the framebuffer to avoid unintended rendering
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindRenderbuffer(GL_RENDERBUFFER, 0)

        self.check()

    def check(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.id_fbo)
        fbo_status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        
        if fbo_status == GL_FRAMEBUFFER_COMPLETE:
            pass
        elif fbo_status == GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
            print("FBO Incomplete: Attachment")
        elif fbo_status == GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
            print("FBO Incomplete: Missing Attachment")
        elif fbo_status == GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
            print("FBO Incomplete: Draw Buffer")
        elif fbo_status == GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
            print("FBO Incomplete: Read Buffer")
        elif fbo_status == GL_FRAMEBUFFER_UNSUPPORTED:
            print("FBO Unsupported")
        else:
            print("Undefined FBO error")

    def __del__(self):
        glDeleteTextures(1, [self.id_color])
        glDeleteRenderbuffers(1, [self.id_depth])
        glDeleteFramebuffers(1, [self.id_fbo])