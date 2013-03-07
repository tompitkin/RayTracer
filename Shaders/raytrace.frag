#version 330

uniform sampler2D textures[1];

varying vec2 texcoord;

void main()
{
    gl_FragColor = texture2D(textures[0], texcoord);
}
