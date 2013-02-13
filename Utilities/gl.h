#ifndef GL_H
#define GL_H

#include <GL/glew.h>
#include <stdio.h>
#include <string>

using namespace std;

class GL
{
public:
    static void checkGLErrors(string otherInfo);
};

#endif // GL_H
