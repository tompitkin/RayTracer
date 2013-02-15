#include "shaderprogram.h"

ShaderProgram::ShaderProgram()
{
    progID = -1;
}

void ShaderProgram::addShader(Shader *shader)
{
    if (shader->type == GL_VERTEX_SHADER)
    {
        vertexShader = shader;
        baseName = shader->shaderName.substr(0, shader->shaderName.find("."));
    }
    else
        fragmentShader = shader;
}

ShaderProgram::Shader::Shader(string name, bool compiled, bool attached, int id, int type)
{
    shaderName = name;
    isCompiled = compiled;
    isAttached = attached;
    shaderID = id;
    this->type = type;
}
