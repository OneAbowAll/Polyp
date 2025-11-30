#version 430 core

layout(location = 0) out vec4 color;

in vec2 vTexCoord;

uniform sampler2D screenTex;

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
uniform float b1;
uniform float b2;
uniform float near;
uniform float far;

uniform vec2 overscanResolution;

// Helper function for the weights
vec4 cubic(float v) {
    vec4 n = vec4(1.0, 2.0, 3.0, 4.0) - v;
    vec4 s = n * n * n;
    float x = s.x;
    float y = s.y - 4.0 * s.x;
    float z = s.z - 4.0 * s.y + 6.0 * s.x;
    float w = 6.0 - x - y - z;
    return vec4(x, y, z, w) * (1.0/6.0);
}

//In teoria usare un sampling bicubico dovrerbbe portare a un risultato piu' accurato (nei miei test e' una differenza minima)
//Ma nel mio caso d'uso non ha senso, le label sono delle maschere in un certo senso.
vec4 textureBicubic(sampler2D sampler, vec2 texCoords) {
    // 1. Get the size of the texture (Overscan resolution in your case)
    vec2 texSize = vec2(textureSize(sampler, 0));
    vec2 invTexSize = 1.0 / texSize;

    // 2. Calculate pixel coordinates
    texCoords = texCoords * texSize - 0.5;

    // 3. Separate integer and fractional parts
    vec2 fxy = fract(texCoords);
    texCoords -= fxy;

    // 4. Calculate the weights for the X and Y axis
    vec4 xcubic = cubic(fxy.x);
    vec4 ycubic = cubic(fxy.y);

    // 5. Calculate the offsets for the 4 combined samples
    // This is a "GPU Trick": instead of sampling 16 pixels individually,
    // we sample 4 locations using linear interpolation to mathematically
    // approximate the 16-sample sum.
    vec4 c = texCoords.xxyy + vec2(-0.5, +1.5).xyxy;

    vec4 s = vec4(xcubic.xz + xcubic.yw, ycubic.xz + ycubic.yw);
    vec4 offset = c + vec4(xcubic.yw, ycubic.yw) / s;

    offset *= invTexSize.xxyy;

    // 6. Sample the texture 4 times
    vec4 sample0 = texture(sampler, offset.xz);
    vec4 sample1 = texture(sampler, offset.yz);
    vec4 sample2 = texture(sampler, offset.xw);
    vec4 sample3 = texture(sampler, offset.yw);

    // 7. Combine the samples
    float sx = s.x / (s.x + s.y);
    float sy = s.z / (s.z + s.w);

    return mix(
       mix(sample3, sample2, sx),
       mix(sample1, sample0, sx),
       sy
    );
}

vec2 applyInverseDistortion(vec2 distorted)
{
    vec2 resolution = vec2(resolution_width, resolution_height);
    float normalizer = max(resolution.x, resolution.y);

    //Metashape usa le cordinate per la y inverse, questo e' il punto fisico del sensore da quel che ho capito
    vec2 principalPoint = vec2(resolution_width/2.0f + cx, resolution_height/2.0f - cy);
    vec2 centered = (distorted - principalPoint) / normalizer;
    vec2 normalized = centered / (f / normalizer);

    vec2 undistorted = normalized;

    //Itera per trovare la distorsione inversa
    for(int i = 0; i < 10; i++) {
        float x = undistorted.x;
        float y = undistorted.y;

        //In teoria questa e' per avere meno perdita di precisione
        float xx = x * x;
        float yy = y * y;
        float r2 = xx + yy;

        //Anche questo come sopra
        float radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3));

        // Tangential distortion
        float xy = x * y;
        vec2 tangential;
        tangential.x = 2.0 * p1 * xy + p2 * (r2 + 2.0 * xx);
        tangential.y = p1 * (r2 + 2.0 * yy) + 2.0 * p2 * xy;

        // Affinity
        float affinity_x = (b1 * x) + (b2 * y); //Se non uso b1 e b2 (nella shader) e' molto piu' accurato (rispetto alla foto)

        //Stima
        vec2 distorted_estimate;
        distorted_estimate.x = (x * radial) + tangential.x + affinity_x;
        distorted_estimate.y = (y * radial) + tangential.y;

        //Correggi la stima, in teoria piu' passi vengono fatti e meno ci muoviamo meglio e' (devo ripassare calcolo numerico per confermarlo)
        vec2 error = normalized - distorted_estimate;
        undistorted += error * 0.3;
    }

    //Passa a coordinate pixel
    return undistorted * (f / normalizer) * normalizer + principalPoint;

    /* LESS NUMERICALY STABLE VERSION ----------------------------------------------------------------
    // Convert from pixel to normalized coordinates
    vec2 principalPoint = vec2(resolution_width/2.0f + cx, resolution_width/2.0f - cy);
    vec2 centered = distorted - principalPoint;
    vec2 normalized = centered / f;

    //Iteration to find undistorted point
    vec2 undistorted = normalized;

    for(int i = 0; i < 8; i++) {
        float x = undistorted.x;
        float y = undistorted.y;

        float r2 = x * x + y * y;
        float r4 = r2 * r2;
        float r6 = r4 * r2;

        // Radial distortion
        float radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;

        // Tangential distortion
        vec2 tangential;
        tangential.x = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
        tangential.y = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;

        // Affinity and non-orthogonality (b1, b2)
        vec2 affinity;
        affinity.x = b1 * x + b2 * y;
        affinity.y = 0.0;  // Only affects x-coordinate

        // Complete distortion model
        vec2 distorted_estimate;
        distorted_estimate.x = x * radial + tangential.x + affinity.x;
        distorted_estimate.y = y * radial + tangential.y + affinity.y;

        vec2 error = normalized - distorted_estimate;
        undistorted += error * 0.5;
    }

    // Convert back to pixel coordinates
    return undistorted * f + principalPoint;
    */
}

void main()
{
    //Centro reale del sensore
    vec2 principalPoint = vec2(resolution_width/2.0f + cx, resolution_height/2.0f - cy);

    // Current pixel in output image (sensor resolution)
    vec2 pixelPos = vTexCoord * vec2(resolution_width, resolution_height);

    // Apply inverse distortion to find source pixel
    vec2 undistortedPixel = applyInverseDistortion(pixelPos);

    //Scala il PP per trovare il PP nell' overscan
    vec2 overscanScale = overscanResolution / vec2(resolution_width, resolution_height);
    vec2 principalPointOverscan = principalPoint * overscanScale;

    //E' l'offset dal centro fisico reale
    vec2 vectorFromCenter = undistortedPixel - principalPoint;

    //Trova il pixel nellp "spazio" dell'overscan
    vec2 overscanPixel = principalPointOverscan + vectorFromCenter;

    //Convert to texture coordinates [0, 1]
    vec2 sourceTexCoord = overscanPixel / overscanResolution;

    //Sample with boundary check
    if (sourceTexCoord.x >= 0.0 && sourceTexCoord.x <= 1.0 &&
        sourceTexCoord.y >= 0.0 && sourceTexCoord.y <= 1.0) {
        color = texture(screenTex, sourceTexCoord);
    } else {
        color = vec4(0.0, 0.0, 0.0, 1.0);
    }
}