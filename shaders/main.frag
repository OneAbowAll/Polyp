#version 460 core
layout(location = 0) out vec4 color;

in vec3 vPos;
in vec2 vTexCoord;

uniform sampler2D uColorTex;

void main()
{   
    color = vec4(texture(uColorTex, vTexCoord.xy).rgb, 1);
}