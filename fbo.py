from OpenGL.GL import *

class fbo:
    def __init__(self, w,h):
        self.w = w
        self.h = h
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
        self.id_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.id_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        # Attach texture to FBO
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.id_tex, 0)

        self.id_tex1 = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.id_tex1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        # Attach texture to FBO
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, self.id_tex1, 0)

        self.id_tex2 = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.id_tex2)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        # Attach texture to FBO
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, self.id_tex2, 0)

        self.id_tex3 = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.id_tex3)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, w, h, 0, GL_RGB, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        # Attach texture to FBO
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D, self.id_tex3, 0)

        # Create a renderbuffer for depth and stencil
        if False:
            renderbuffer = glGenRenderbuffers(1)
            glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer)
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, w, h)
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, renderbuffer)
        else:
            # Create a texture for the depth (and stencil, if needed) component
            self.id_depth = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.id_depth)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, w, h, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.id_depth, 0)

        # Check FBO completeness
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            print("Error: Framebuffer is not complete!")
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            return None

        # Unbind the framebuffer to avoid unintended rendering
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindRenderbuffer(GL_RENDERBUFFER, 0)

        self.check()
        return fbo
    
    

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
