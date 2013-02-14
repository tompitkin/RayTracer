#include "vec4uniform.h"

Vec4Uniform::Vec4Uniform(GLfloat *aBuffer, string varName)
{
    theBuffer = aBuffer;
    shaderVarName = varName;
    needsUpdate = true;
}

Vec4Uniform::~Vec4Uniform()
{
    delete []theBuffer;
}

void Vec4Uniform::update(int shaderProgID)
{
    if (theBuffer != nullptr)
    {
        int progLocation = glGetUniformLocation(shaderProgID, shaderVarName.c_str());
        if (progLocation != -1)
            glUniform4fv(progLocation, 1, theBuffer);
        else
            fprintf(stderr, "Can not locate %s location: %d\n", shaderVarName.c_str(), progLocation);
        GL::checkGLErrors("updateVec4Uniform: " + shaderVarName);
    }
    else
        fprintf(stderr, "Vec4Uniform.update: theBuffer for %s is null-no update\n", shaderVarName.c_str());
}
