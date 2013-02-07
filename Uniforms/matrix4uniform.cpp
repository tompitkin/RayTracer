#include "matrix4uniform.h"

Matrix4Uniform::Matrix4Uniform(GLfloat *aBuffer, string varName)
{
    theBuffer = aBuffer;
    shaderVarName = varName;
    needsUpdate = true;
}

Matrix4Uniform::~Matrix4Uniform()
{
    delete []theBuffer;
}

void Matrix4Uniform::update()
{
}
