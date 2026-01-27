#version 460 core
layout(location = 0) out vec4 color;

in vec4 vPos;
in vec3 vOrthoPos;
in vec2 vTexCoord;

uniform int uRenderMode;

uniform sampler2D uColorTex;
uniform sampler2D uLabelMap;
uniform sampler2D uDepthTex;
uniform sampler2D uReprojectionTex;

void main()
{
    vTexCoord;
    uColorTex;
    uDepthTex;

    //Da [-1, 1] a [0, 1]
    float orthoZ = vOrthoPos.z * 0.5+0.5;
    vec2 orthoPos = (vOrthoPos.xy + vec2(1, 1))/2;
    float depth =  texture(uDepthTex, orthoPos.xy).r;

    vec4 label = vec4(texture(uLabelMap, orthoPos.xy).rgb, 1);

    //Use only the labelMap
    if(uRenderMode == 0)
    {
        float occlusion = orthoZ - 0.005 < depth ? 1.0 : 0.0;
        color = label * occlusion;
    }
    //Rendereizza la labelmap sopra alla texture del modello (uso l'addizione per visualizzare meglio la texture sotto)
    else if(uRenderMode == 1)
    {
        float occlusion = orthoZ - 0.005 < depth ? 1.0 : 0.0;

        if(length(label.xyz) > 0.0)
        {
            float alpha = 0.8;
            if(occlusion > 0)
                color = vec4(texture(uColorTex, vTexCoord.xy).rgb * (1-alpha) + label.rgb * occlusion * alpha, 1);
            else
                color = vec4(texture(uColorTex, vTexCoord.xy).rgb, 1);
        }
        else
            color = vec4(texture(uColorTex, vTexCoord.xy).rgb, 1);
    }
    else if(uRenderMode == 3)
    {
        vec2 reprojTexPos = ((vPos.xy/vPos.w)*vec2(1.2, 1.2) + vec2(1, 1))/2; //Il *ve2(1.2, 1.2) e' un po' un hack, in teoria sarebbe da portarci dietro l'overscan factor
        color = vec4(texture(uReprojectionTex, reprojTexPos.xy).rgb, 1);
    }
    else
    {
        color = vec4(texture(uColorTex, vTexCoord.xy).rgb, 1);
    }


    /*
    if(length(vPos.xy - vec2(0, 0)) < 1)
        color = vec4(0, 0, 0, 1);
    else
        color = vec4(vPos.xyz, 1);
    */
}