#ifndef VEC3UNIFORM_H
#define VEC3UNIFORM_H

#include <GL/glew.h>
#include "uniform.h"
#include "Utilities/gl.h"

class Vec3Uniform : public Uniform
{
public:
    Vec3Uniform(GLfloat *aBuffer, string varName);
    virtual ~Vec3Uniform();

    virtual void update(int shaderProgID);

    GLfloat *theBuffer = nullptr;
};

#endif // VEC3UNIFORM_H
