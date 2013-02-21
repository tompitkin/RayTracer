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

void Matrix4Uniform::update(int shaderProgID)
{
    if (theBuffer != nullptr)
    {
        int matLocation = glGetUniformLocation(shaderProgID, shaderVarName.c_str());
        if (matLocation != -1)
        {
            if (theBuffer != nullptr)
                glUniformMatrix4fv(matLocation, 1, GL_FALSE, theBuffer);
            else
                fprintf(stderr, "Matrix4Uniform.update: theBuffer for %s is null\n", shaderVarName.c_str());
        }
        else
            fprintf(stderr, "Can not locate %s location: %d\n", shaderVarName.c_str(), matLocation);
        GL::checkGLErrors(" updateMatrix4Uniform: " + shaderVarName);
    }
    else
        fprintf(stderr, "Matrix4Uniform.update: theBuffer for %s is null-no update\n", shaderVarName.c_str());
}
