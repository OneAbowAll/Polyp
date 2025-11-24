#version 430 core
layout(location = 0) out vec4 color;

in vec2 vTexCoord;

uniform sampler2D screenTex;

void main() {
    color = texture(screenTex, vTexCoord);
}