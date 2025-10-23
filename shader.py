import os
import glm
from OpenGL.GL import glGetUniformLocation, glUniform1i, glUniform1f, glUniform3f, glUniformMatrix4fv, GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, GL_FALSE
from OpenGL.GL.shaders import compileProgram, compileShader

import log

def load_shader_from_files(shader_files_name):
    """
    Loads from the shader folder the fragment shader {shader_file_name}.vert and {shader_file_name}.frag
    Returns shader object.
    """

    vertex_shader_path = f"./shaders/{shader_files_name}.vert"
    fragment_shader_path = f"./shaders/{shader_files_name}.frag"

    if not os.path.exists(vertex_shader_path):
        raise FileExistsError(f"[ERROR] Could not find {shader_files_name}.vert in shaders folder.")
    
    if not os.path.exists(fragment_shader_path):
        raise FileExistsError(f"[ERROR] Could not find {shader_files_name}.frag in shaders folder.")
    
    vertex_file_content = fragment_file_content = ""
    with open(vertex_shader_path, "r") as vertex_file:
        vertex_file_content = vertex_file.read()
        
    with open(fragment_shader_path, "r") as fragment_file:
        fragment_file_content = fragment_file.read()

    return Shader(vertex_file_content, fragment_file_content)



class Shader:
    def __init__(self, vertex_shader_str , fragment_shader_str):
        self.uniforms = {}
        self.program = compileProgram(
            compileShader(vertex_shader_str, GL_VERTEX_SHADER),
            compileShader(fragment_shader_str, GL_FRAGMENT_SHADER)
        )
        

    def uni(self, name):
            if name not in self.uniforms:
                location = glGetUniformLocation(self.program, name)

                if location == -1:
                    log.print_warning(f"Cannot find {name} in ShaderProgram")
                    return -1
                
                self.uniforms[name] = location
    

            return self.uniforms[name]
    
    def set_int(self, name, value):
        glUniform1i(self.uni(name), value)

    def set_float(self, name, value):
        glUniform1f(self.uni(name), value)

    def set_float3(self, name, value: list[float]):
        glUniform3f(self.uni(name), value[0], value[1], value[2])

    def set_vec3(self, name, value: glm.vec3):
        glUniform3f(self.uni(name), value.x, value.y, value.z)

    def set_mat4(self, name, value: glm.mat4x4):
        glUniformMatrix4fv(self.uni(name), 1, GL_FALSE, glm.value_ptr(value))