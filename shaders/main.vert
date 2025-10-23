#version 430 core
in vec3 aPosition;
in vec2 aTexCoord;

out vec4 vPos;
out vec2 vTexCoord;

uniform mat4 uProj; 
uniform mat4 uView;


void main(void)
{
    vTexCoord = aTexCoord;

    vec4 pos = uProj*uView*vec4(aPosition, 1.0);
    vPos = pos;
    gl_Position = pos;
}