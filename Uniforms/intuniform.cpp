#include "intuniform.h"

IntUniform::IntUniform(int anInt, string varName)
{
    theInt = anInt;
    shaderVarName = varName;
    needsUpdate = true;
}

void IntUniform::update(int shaderProgID)
{
    int progLocation = glGetUniformLocation(shaderProgID, shaderVarName.c_str());
    if (progLocation != -1)
        glUniform1i(progLocation, theInt);
    else
        fprintf(stderr, "Can not locate %s location: %d\n", shaderVarName.c_str(), progLocation);
    GL::checkGLErrors("updateIntUniform: " + shaderVarName);
}


