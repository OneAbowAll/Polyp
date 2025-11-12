#version 460 core
layout(location = 0) out vec4 color;

in vec3 vPos;
in vec3 vOrthoPos;
in vec2 vTexCoord;

uniform int uRenderMode;

uniform sampler2D uColorTex;
uniform sampler2D uLabelMap;

void main()
{
    vTexCoord;
    uColorTex;

    //Da [-1, 1] a [0, 1]
    vec2 orthoPos = (vOrthoPos.xy + vec2(1, 1))/2;

    vec4 label = vec4(texture(uLabelMap, orthoPos.xy).rgb, 1);

    //Use only the labelMap
    if(uRenderMode == 0)
    {
        color = label;
    }
    //Rendereizza la labelmap sopra alla texture del modello (uso l'addizione per visualizzare meglio la texture sotto)
    else
    {

    if(length(label.xyz) > 0.0)
        color = vec4(texture(uColorTex, vTexCoord.xy).rgb + label.rgb, 1);
    else
        color = vec4(texture(uColorTex, vTexCoord.xy).rgb, 1);
    }


    /*
    if(length(vPos.xy - vec2(0, 0)) < 1)
        color = vec4(0, 0, 0, 1);
    else
        color = vec4(vPos.xyz, 1);
    */
}