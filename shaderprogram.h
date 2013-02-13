#ifndef SHADERPROGRAM_H
#define SHADERPROGRAM_H

#include <vector>
#include "Uniforms/uniform.h"

using namespace std;

class ShaderProgram
{
public:
    ShaderProgram();

    int progID;
    vector<Uniform> uniformList;
};

#endif // SHADERPROGRAM_H
