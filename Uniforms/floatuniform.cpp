#include "floatuniform.h"

FloatUniform::FloatUniform(float aFloat, string varName)
{
    theFloat = aFloat;
    shaderVarName = varName;
    needsUpdate = true;
}

FloatUniform::~FloatUniform()
{
}

void FloatUniform::update(int shaderProgID)
{
    int progLocation = glGetUniformLocation(shaderProgID, shaderVarName.c_str());
    if (progLocation != -1)
        glUniform1f(progLocation, theFloat);
    else
        fprintf(stderr, "Can not locate %s location: %d\n", shaderVarName.c_str(), progLocation);
    GL::checkGLErrors("updateFloatUniform: " + shaderVarName);
}
