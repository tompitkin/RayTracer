#include "axes.h"
#include "scene.h"

Axes::Axes(Scene *top)
{
    theScene = top;
    axesShaderProg = new ShaderProgram();
    axesShaderProg->vertexShader = new ShaderProgram::Shader("axes.vert", false, false, -1, GL_VERTEX_SHADER, axesShaderProg);
    axesShaderProg->fragmentShader = new ShaderProgram::Shader("axes.frag", false, false, -1, GL_FRAGMENT_SHADER, axesShaderProg);
    theScene->shUtil.setupShaders(axesShaderProg);
}

Axes::~Axes()
{
    theScene = nullptr;
    delete axesShaderProg;
}
