#include "shaderprogram.h"

ShaderProgram::ShaderProgram()
{
    progID = -1;
}

ShaderProgram::~ShaderProgram()
{
    delete vertexShader;
    vertexShader = nullptr;
    delete fragmentShader;
    fragmentShader = nullptr;
}

void ShaderProgram::addUniform(Uniform *toAdd)
{
    uniformList.push_back(toAdd);
}

ShaderProgram::Shader::Shader(string name, bool compiled, bool attached, int id, int type, shared_ptr<ShaderProgram> prog)
{
    shaderName = name;

    if (type == GL_VERTEX_SHADER)
        prog->baseName = name.substr(0, name.find("."));

    isCompiled = compiled;
    isAttached = attached;
    shaderID = id;
    this->type = type;
}

ShaderProgram::Shader::~Shader()
{
}
