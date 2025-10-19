import OpenGL.GL as gl
from PIL import Image
import numpy as np

def load_texture(image_path):
    """
    Load a JPEG image and set it as a texture in PyOpenGL.
    
    :param image_path: Path to the JPEG image file.
    :return: OpenGL texture ID.
    """
    # Load the image using Pillow (PIL)
    image = Image.open(image_path)
    
    image = image.transpose(Image.FLIP_TOP_BOTTOM)  # Flip the image vertically for OpenGL
     
    if image.mode == "I;16":
        # Normalize 16-bit grayscale to 8-bit
        arr = np.array(image, dtype=np.uint16)
        arr = (arr / 256).astype(np.uint8)  # scale 0-65535 to 0-255
        image_rgb = Image.fromarray(arr, mode="L").convert("RGBA")
    else:
        image_rgb = image.convert("RGBA")
    
     
    image_data = image_rgb.tobytes()  # Convert to bytes for OpenGL
    
    texture_id = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)

    # Set texture parameters
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

    # Specify the 2D texture
    gl.glTexImage2D(
        gl.GL_TEXTURE_2D,   # Target
        0,                  # Mipmap level
        gl.GL_RGBA,         # Internal format
        image.width,        # Image width
        image.height,       # Image height
        0,                  # Border (must be 0)
        gl.GL_RGBA,         # Format of the pixel data
        gl.GL_UNSIGNED_BYTE,  # Data type of the pixel data
        image_data          # Actual image data
    )

    # Generate mipmaps (optional, for smoother scaling)
    gl.glGenerateMipmap(gl.GL_TEXTURE_2D)

    # Unbind the texture
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    del image_data

    return texture_id, image.width, image.height

def create_texture(w, h, type = "uint8"):
    # Generate texture ID
    texture_id = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)

    if(type == "rgba32f"):
        # Create a black pixel buffer (RGBA format)
        black_data = np.zeros((h, w, 4), dtype=np.float32)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, w, h, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, black_data)
    elif (type == "int32"):
        # Create a black pixel buffer (RGBA format)
        result = gl.glGetInternalformativ(gl.GL_TEXTURE_2D,  gl.GL_R32UI, gl.GL_INTERNALFORMAT_SUPPORTED, 1)
        print(result == True)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 4)
        black_data = np.zeros((h, w), dtype=np.uint32)   
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_R32UI, w, h, 0, gl.GL_RED_INTEGER , gl.GL_UNSIGNED_INT, black_data)
    else:
        # Create a black pixel buffer (RGBA format)
        black_data = np.zeros((h, w, 4), dtype=np.uint8)  # Shape: (height, width, 4)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, w, h, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, black_data)

    # Allocate and upload the texture
    

    # Set texture parameters
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

    # Unbind the texture
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    return texture_id  # Return texture handle
