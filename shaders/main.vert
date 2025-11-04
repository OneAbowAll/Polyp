#version 430 core
in vec3 aPosition;
in vec2 aTexCoord;

out vec3 vPos;
out vec2 vTexCoord;

uniform int uViewMode;  //0 for arcball camera controll - 1 for look through sensor mode

uniform mat4 uProj; 
uniform mat4 uView;
uniform mat4 uModel;

//Sensor settings
uniform int resolution_width;
uniform int resolution_height;

//Sensor properties
uniform float pixel_width;
uniform float pixel_height;
uniform float focal_length;

//Sensor calibration
uniform float f;
uniform float cx; // this is the offset w.r.t. the center
uniform float cy; // this is the offset w.r.t. the center
uniform float k1;
uniform float k2;
uniform float k3;
uniform float p1;
uniform float p2;
uniform float near;
uniform float far;

vec2 xyz_to_uv(vec3 p){
    float x = p.x/p.z;
    float y = -p.y/p.z;
    float r = sqrt(x*x+y*y);
    float r2 = r*r;
    float r4 = r2*r2;
    float r6 = r4*r2;
    float r8 = r6*r2;

    float A = (1.0+k1*r2+k2*r4+k3*r6  /*+k4*r8*/ ); 
    float B = (1.0 /* +p3*r2+p4*r4 */ );

    float xp = x * A+ (p1*(r2+2*x*x)+2*p2*x*y) * B;
    float yp = y * A+ (p2*(r2+2*y*y)+2*p1*x*y) * B;

    float u = resolution_width*0.5+cx+xp*f; //+xp*b1+yp*b2
    float v = resolution_height*0.5+cy+yp*f;

    u /= resolution_width;
    v /= resolution_height;

    return vec2(u,v);
}

void main(void)
{
    vPos = aPosition;
    vTexCoord = aTexCoord;

    if(uViewMode == 1)
    {
        float focmm = f / resolution_width;
        vec3 pos_vs = (uView * uModel * vec4(aPosition, 1.0)).xyz;
        gl_Position = vec4(xyz_to_uv(pos_vs)*2.0-1.0, pos_vs.z/(100.0*focmm), 1.0);
    }
    else
    {
        if(uViewMode == 2)
            vPos = (uProj*uView*uModel*vec4(aPosition, 1.0)).xyz;

        vec4 pos = uProj*uView*uModel*vec4(aPosition, 1.0);
        gl_Position = pos;
    }
}