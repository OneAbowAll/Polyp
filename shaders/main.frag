#version 460 core
layout(location = 0) out vec4 color;

in vec3 vPos;
in vec2 vTexCoord;

uniform sampler2D uColorTex;

void main()
{   vTexCoord;
    color = vec4(texture(uColorTex, vTexCoord.xy).rgb, 1);
    /*
    if(length(vPos.xy - vec2(0, 0)) < 1)
        color = vec4(0, 0, 0, 1);
    else
        color = vec4(vPos.xyz, 1);
    */
}