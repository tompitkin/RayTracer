#ifndef INTUNIFORM_H
#define INTUNIFORM_H

#include <GL/glew.h>
#include "uniform.h"
#include "Utilities/gl.h"

class IntUniform : public Uniform
{
public:
    IntUniform(int anInt, string varName);

    virtual void update(int shaderProgID);

    int theInt;
};

#endif // INTUNIFORM_H
