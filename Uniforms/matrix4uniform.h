#ifndef MATRIX4UNIFORM_H
#define MATRIX4UNIFORM_H

#include "uniform.h"

class Matrix4Uniform : public Uniform
{
public:
    Matrix4Uniform();

    virtual void update();
};

#endif // MATRIX4UNIFORM_H
