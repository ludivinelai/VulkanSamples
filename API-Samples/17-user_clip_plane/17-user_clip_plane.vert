#version 400

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (std140, set = 0, binding = 0) uniform bufferVals {
    mat4 mvp;
} myBufferVals;
layout (std140, set = 0, binding = 1) uniform u_clipPlane {
    vec4 plane;
} uClipPlane;

layout (location = 0) in vec4 pos;
layout (location = 1) in vec4 inColor;

layout (location = 0) out vec4 outColor;
layout (location = 1) out float v_clipDist;
layout (location = 2) out vec4 outPlane;
layout (location = 3) out mat4 outMVP;
void main() {
    outColor = inColor;
    gl_Position = myBufferVals.mvp * pos;
    v_clipDist = dot(pos.xyz, uClipPlane.plane.xyz) + uClipPlane.plane.w;
    outPlane = uClipPlane.plane;
    outMVP = myBufferVals.mvp;
}
