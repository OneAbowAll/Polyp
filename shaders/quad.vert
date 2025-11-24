#version 430 core
layout(location = 0)  in vec3 aPosition;

out vec2 vTexCoord;



void main(void)
{
    vTexCoord = (aPosition.xy + vec2(1, 1))/2;

    gl_Position = vec4(aPosition.x, aPosition.y, 0.0, 1.0);
}