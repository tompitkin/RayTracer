#ifndef SHADERUTILS_H
#define SHADERUTILS_H

#include <QFile>
#include <fstream>
#include "shaderprogram.h"
#include "Utilities/gl.h"

class Scene;

class ShaderUtils
{
public:
    ShaderUtils(Scene *theScene);
    virtual ~ShaderUtils();

    bool setupShaders(ShaderProgram *prog);
    static bool loadAndCompileShader(ShaderProgram::Shader *shaderInfo);
    static vector<string> *loadShaderSourceFile(string fileName);

    Scene *theScene;
};

#endif // SHADERUTILS_H
