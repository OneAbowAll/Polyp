#version 430 core
in vec3 aPosition;
in vec3 aColor;

out vec3 vColor;

uniform mat4 uProj; 
uniform mat4 uView;
uniform mat4 uModel;


void main(void)
{
    vColor = aColor;

    vec4 pos = uProj*uView*uModel*vec4(aPosition, 1.0);
    gl_Position = pos;
}