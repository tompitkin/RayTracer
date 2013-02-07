#ifndef MATRIX4UNIFORM_H
#define MATRIX4UNIFORM_H

#include "uniform.h"
#include <GL/gl.h>

class Matrix4Uniform : public Uniform
{
public:
    Matrix4Uniform(GLfloat *aBuffer, string varName);
    virtual ~Matrix4Uniform();

    virtual void update();

    GLfloat *theBuffer;
};

#endif // MATRIX4UNIFORM_H
