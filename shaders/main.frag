#version 460 core
layout(location = 0) out vec4 color;

in vec3 vPos;
in vec2 vTexCoord;

uniform sampler2D uColorTex;
uniform sampler2D uLabelMap;

void main()
{
    vTexCoord;
    uColorTex;

    vec2 orthoPos = (vPos.xy + vec2(1, 1))/2;

    vec4 label = vec4(texture(uLabelMap, orthoPos.xy).rgb, 1);
    if(length(label.xyz) > 0.0)
        color = label;
    else
        color = vec4(texture(uColorTex, vTexCoord.xy).rgb, 1);
    /*
    if(length(vPos.xy - vec2(0, 0)) < 1)
        color = vec4(0, 0, 0, 1);
    else
        color = vec4(vPos.xyz, 1);
    */
}