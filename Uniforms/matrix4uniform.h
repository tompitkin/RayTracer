#ifndef MATRIX4UNIFORM_H
#define MATRIX4UNIFORM_H

#include <GL/glew.h>
#include "uniform.h"
#include "Utilities/gl.h"

class Matrix4Uniform : public Uniform
{
public:
    Matrix4Uniform(GLfloat *aBuffer, string varName);
    virtual ~Matrix4Uniform();

    virtual void update(int shaderProgID);

    GLfloat *theBuffer = nullptr;
};

#endif // MATRIX4UNIFORM_H
