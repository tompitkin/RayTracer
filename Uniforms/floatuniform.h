#ifndef FLOATUNIFORM_H
#define FLOATUNIFORM_H

#include <GL/glew.h>
#include "uniform.h"
#include "Utilities/gl.h"

class FloatUniform : public Uniform
{
public:
    FloatUniform(float aFloat, string varName);

    virtual void update(int shaderProgID);

    float theFloat;
};

#endif // FLOATUNIFORM_H
