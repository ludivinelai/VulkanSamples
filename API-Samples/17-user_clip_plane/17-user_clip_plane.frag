#version 400
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable
layout (location = 0) in vec4 color;
layout (location = 1) in float v_clipDist;
layout (location = 0) out vec4 outColor;
void main() {
    if (v_clipDist < 0.0)
        discard;
    outColor = color;
}

