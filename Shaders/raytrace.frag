#version 410

uniform sampler2D tex;

varying vec2 texcoord;

void main()
{
    gl_FragColor = texture2D(tex, texcoord);
    //gl_FragColor = vec4(texcoord[0], texcoord[1], 0.0, 1.0);
}
