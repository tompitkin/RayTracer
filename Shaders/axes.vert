#version 330
layout(location=0) in vec4 vertex;
// vertex shader for axes drawing
uniform mat4 viewMat;
uniform mat4 projMat;
void main(void) {
   gl_Position = projMat * viewMat * vertex;
   //gl_Position = vertex;
}
