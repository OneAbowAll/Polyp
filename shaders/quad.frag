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

vec2 applyInverseDistortion(vec2 distorted) {

    vec2 resolution = vec2(resolution_width, resolution_width);
    vec2 imageCenter = resolution * 0.5;
    float normalizer = max(resolution.x, resolution.y);

    vec2 principalPoint = vec2(resolution_width/2.0f + cx, resolution_width/2.0f - cy);
    vec2 centered = (distorted - principalPoint) / normalizer;
    vec2 normalized = centered / (f / normalizer);

    vec2 undistorted = normalized;

    for(int i = 0; i < 8; i++) {  // More iterations
        float x = undistorted.x;
        float y = undistorted.y;

        // Compute r2 carefully to avoid precision loss
        float xx = x * x;
        float yy = y * y;
        float r2 = xx + yy;

        // Use Horner's method for polynomial evaluation (more stable)
        float radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3));

        // Tangential distortion
        float xy = x * y;
        vec2 tangential;
        tangential.x = 2.0 * p1 * xy + p2 * (r2 + 2.0 * xx);
        tangential.y = p1 * (r2 + 2.0 * yy) + 2.0 * p2 * xy;

        // Affinity
        float affinity_x = b1 * x + b2 * y; //Se non uso b1 e b2 e' molto piu' accurato (rispetto alla foto)

        // Complete model
        vec2 distorted_estimate;
        distorted_estimate.x = x * radial + tangential.x + affinity_x;
        distorted_estimate.y = y * radial + tangential.y;

        vec2 error = normalized - distorted_estimate;
        undistorted += error * 0.3;  // Reduced from 0.5
    }

    // Convert back to pixel coordinates
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
    vec2 principalPoint = vec2(resolution_width/2.0f + cx, resolution_width/2.0f - cy);

    // Current pixel in output image (sensor resolution)
    vec2 pixelPos = vTexCoord * vec2(resolution_width, resolution_height);

    // Apply inverse distortion to find source pixel
    vec2 undistortedPixel = applyInverseDistortion(pixelPos);

    // Map to overscan texture coordinates
    // Account for the fact that overscan has a wider FOV
    vec2 overscanCenter = overscanResolution * 0.5;
    vec2 sensorCenter = vec2(resolution_width, resolution_height) * 0.5;

    // Map from sensor space to overscan space
    // The pixel offset from center remains the same, but we need to account for the larger texture
    vec2 offsetFromCenter = undistortedPixel - sensorCenter;
    vec2 overscanPixel = overscanCenter + offsetFromCenter;

    // Convert to texture coordinates [0, 1]
    vec2 sourceTexCoord = overscanPixel / overscanResolution;

    // Sample with boundary check
    if (sourceTexCoord.x >= 0.0 && sourceTexCoord.x <= 1.0 &&
        sourceTexCoord.y >= 0.0 && sourceTexCoord.y <= 1.0) {
        color = texture(screenTex, sourceTexCoord);
    } else {
        color = vec4(0.0, 0.0, 0.0, 1.0);  // Black for out-of-bounds
    }
    /*
    vec2 undistortedUV = applyInverseDistortion(vTexCoord);

    if (undistortedUV.x < 0.0 || undistortedUV.x > 1.0 ||
        undistortedUV.y < 0.0 || undistortedUV.y > 1.0) {
        color = vec4(0.0, 0.0, 0.0, 1.0);
    } else {
        color = texture(screenTex, undistortedUV);
    }
    */
}