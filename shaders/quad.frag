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

vec2 applyInverseDistortionOld(vec2 uvDistorted) {
    // This function iteratively finds the undistorted coordinate
    // that would produce the given distorted coordinate

    vec2 imageSize = vec2(resolution_width, resolution_height);
    vec2 principalPoint = vec2(resolution_width/2.0f + cx, resolution_width/2.0f - cy);

    vec2 pixelDistorted = uvDistorted * imageSize;
    vec2 normalizedDistorted = (pixelDistorted - principalPoint) / f;

    // Initial guess: undistorted = distorted
    vec2 normalizedUndistorted = normalizedDistorted;
    normalizedUndistorted.y = -normalizedDistorted.y;

    // Iterate to find the inverse (usually converges in 3-5 iterations)
    for (int i = 0; i < 16; i++) {
        float x = normalizedUndistorted.x;
        float y = normalizedUndistorted.y;

        float r2 = x*x + y*y;
        float r4 = r2 * r2;
        float r6 = r4 * r2;

        float A = 1.0 + k1*r2 + k2*r4 + k3*r6;

        float xp = x * A + (p1*(r2 + 2.0*x*x) + 2.0*p2*x*y);
        float yp = y * A + (p2*(r2 + 2.0*y*y) + 2.0*p1*x*y);

        // Compute error
        vec2 error = vec2(xp, yp) - normalizedDistorted;

        // Update estimate
        normalizedUndistorted -= error;
    }

    // Convert back to UV
    vec2 pixelUndistorted = normalizedUndistorted * f + principalPoint;
    return pixelUndistorted / imageSize;
}

vec2 applyAffinityDistortion(vec2 point, float b1, float b2) {
    // This models sensor skew and scale differences
    vec2 result;
    result.x = b1 * point.x + b2 * point.y;
    result.y = point.y;
    return result;
}

vec2 applyInverseDistortion(vec2 distorted) {
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

        /*
        float r2 = dot(undistorted, undistorted);
        float r4 = r2 * r2;
        float r6 = r4 * r2;

        float radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;

        vec2 tangential;
        tangential.x = 2.0 * p1 * undistorted.x * undistorted.y +
                       p2 * (r2 + 2.0 * undistorted.x * undistorted.x);
        tangential.y = p1 * (r2 + 2.0 * undistorted.y * undistorted.y) +
                       2.0 * p2 * undistorted.x * undistorted.y;


        vec2 distorted_estimate = undistorted * radial + tangential;
        vec2 error = normalized - distorted_estimate;

        undistorted += error * 0.5;
        */
    }

    // Convert back to pixel coordinates
    return undistorted * f + principalPoint;
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