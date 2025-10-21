
VERTEX_SHADER = """
#version 430 core
in vec3 aPosition;
in vec2 aTexCoord;

uniform mat4 uProj; 
uniform mat4 uView;


void main(void)
{
    vec4 pos = uProj*uView*vec4(aPosition, 1.0);
    gl_Position = pos;

    return;
}
"""

FRAGMENT_SHADER = """
#version 460 core
layout(location = 0) out vec4 color;

void main()
{
    color = vec4(1, 0, 0, 1);
}
"""

vertex_shader_fsq = """
#version 430 core
layout(location = 0) in vec3 aPosition;
out vec2 vTexCoord;
uniform float uSca;
uniform vec2 uTra;

void main(void)
{
    if(uSca == 0.0)  
        gl_Position = vec4(aPosition, 1.0);
     else
         gl_Position = vec4(aPosition*uSca+vec3(uTra,0.0), 1.0);
    vTexCoord = aPosition.xy *0.5+0.5;
}
"""
fragment_shader_fsq = """  
#version 430 core
in vec2 vTexCoord;
out vec4 FragColor;
uniform sampler2D uColorTex;
uniform sampler2D uMask;
uniform ivec2 uOff;
uniform ivec2 uSize;
uniform float uSca;

uniform  int resolution_width;
uniform  int resolution_height;

void main()
{
   FragColor = vec4(texture(uColorTex, vTexCoord).rgb,1.0);
   // FragColor = vec4(vTexCoord.xy,0.0, 1.0);
   ivec2 texel_coord = ivec2(vTexCoord.xy*ivec2(resolution_width,resolution_height)) ;
   texel_coord.y = resolution_height - texel_coord.y; // flip y coordinate
   texel_coord = texel_coord - uOff;
   texel_coord.y = uSize.y - texel_coord.y; // flip y coordinate

   float v = texelFetch(uMask, texel_coord, 0).x; 

   if (uSca != 0.0) 
       FragColor+= vec4(1,1,1, 0.0)*v/uSca; // red if mask is set
   
}
"""

bbox_shader_str = """
#version 460

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in; 


layout(binding = 12)  uniform sampler2D uColorTexture;    // for each pixel of the image, the coordinates in parametric space

layout(std430, binding = 6) buffer bBbox {
    uint bbox[];  
};


uniform  int resolution_width;
uniform  int resolution_height;

void main() {

    if (gl_GlobalInvocationID.x >= resolution_width || gl_GlobalInvocationID.y >= resolution_height)
        return;

    vec4 texel = texelFetch(uColorTexture, ivec2(gl_GlobalInvocationID.xy), 0);

    if (texel.r < 1.0) {
        // bbox[0]: min_x, bbox[1]: min_y, bbox[2]: max_x, bbox[3]: max_y
        atomicMin(bbox[0], gl_GlobalInvocationID.x);
        atomicMin(bbox[1], gl_GlobalInvocationID.y);
        atomicMax(bbox[2], gl_GlobalInvocationID.x);
        atomicMax(bbox[3], gl_GlobalInvocationID.y);
    }
}
"""