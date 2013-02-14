#ifndef VEC4UNIFORM_H
#define VEC4UNIFORM_H

#include <GL/glew.h>
#include "uniform.h"
#include "Utilities/gl.h"

class Vec4Uniform : public Uniform
{
public:
    Vec4Uniform(GLfloat *aBuffer, string varName);
    virtual ~Vec4Uniform();

    virtual void update(int shaderProgID);

    GLfloat *theBuffer = nullptr;
};

#endif // VEC4UNIFORM_H
