#include "vec3uniform.h"

Vec3Uniform::Vec3Uniform(GLfloat *aBuffer, string varName)
{
    theBuffer = aBuffer;
    shaderVarName = varName;
    needsUpdate = true;
}

Vec3Uniform::~Vec3Uniform()
{
    delete []theBuffer;
}

void Vec3Uniform::update(int shaderProgID)
{
    if (theBuffer != nullptr)
    {
        int progLocation = glGetUniformLocation(shaderProgID, shaderVarName.c_str());
        if (progLocation != -1)
            glUniform3fv(progLocation, 1, theBuffer);
        else
            fprintf(stderr, "Can not locate %s location: %d\n", shaderVarName.c_str(), progLocation);
        GL::checkGLErrors("updateVec3Uniform: " + shaderVarName);
    }
    else
        fprintf(stderr, "Vec3Uniform.update: theBuffer for %s is null-no update\n", shaderVarName.c_str());
}
