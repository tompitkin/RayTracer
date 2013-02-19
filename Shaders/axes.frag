#version 330
// line drawing fragment shader with no uniforms
out vec4 fragColor;
uniform vec4 axisColor;

void main(void) {
               fragColor = axisColor;
}
