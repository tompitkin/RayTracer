#include "gl.h"

void GL::checkGLErrors(string otherInfo)
{
    GLenum errCode = glGetError();
    if (errCode != GL_NO_ERROR)
    {
        if (otherInfo.empty())
            fprintf(stderr, "OpenGL Error: %s\n", gluErrorString(errCode));
        else
            fprintf(stderr, "OpenGL Error: %s %s\n", otherInfo.c_str(), gluErrorString(errCode));
    }
}
