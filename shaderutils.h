#ifndef SHADERUTILS_H
#define SHADERUTILS_H

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
    static GLchar *loadShaderSourceFile(string fileName, GLint *length);

    Scene *theScene;
};

#endif // SHADERUTILS_H
