#include "matrix3uniform.h"

Matrix3Uniform::Matrix3Uniform(GLfloat *aBuffer, string varName)
{
    theBuffer = aBuffer;
    shaderVarName = varName;
    needsUpdate = true;
}

Matrix3Uniform::~Matrix3Uniform()
{
    delete []theBuffer;
}

void Matrix3Uniform::update(int shaderProgID)
{
    if (theBuffer != nullptr)
    {
        int matLocation = glGetUniformLocation(shaderProgID, shaderVarName.c_str());
        if (matLocation != -1)
        {
            if (theBuffer != nullptr)
                glUniformMatrix3fv(matLocation, 1, GL_FALSE, theBuffer);
            else
                fprintf(stderr, "Matrix3Uniform.update: theBuffer for %s is null\n", shaderVarName.c_str());
        }
        else
            fprintf(stderr, "Can not locate %s location: %d\n", shaderVarName.c_str(), matLocation);
        GL::checkGLErrors(" updateMatrix3Uniform: " + shaderVarName);
    }
    else
        fprintf(stderr, "Matrix3Uniform.update: theBuffer for %s is null-no update\n", shaderVarName.c_str());
}
