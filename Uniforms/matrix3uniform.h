#ifndef MATRIX3UNIFORM_H
#define MATRIX3UNIFORM_H

#include <GL/glew.h>
#include "uniform.h"
#include "Utilities/gl.h"

class Matrix3Uniform : public Uniform
{
public:
    Matrix3Uniform(GLfloat *aBuffer, string varName);
    virtual ~Matrix3Uniform();

    virtual void update(int shaderProgID);

    GLfloat *theBuffer = nullptr;
};

#endif // MATRIX3UNIFORM_H
