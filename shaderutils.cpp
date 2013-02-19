#include "shaderutils.h"
#include "scene.h"

ShaderUtils::ShaderUtils(Scene *theScene)
{
    this->theScene = theScene;
}

ShaderUtils::~ShaderUtils()
{
    theScene = nullptr;
}

bool ShaderUtils::setupShaders(ShaderProgram *prog)
{
    bool compileVertSuccess = false;
    bool compileFragSuccess = false;
    bool programChanged = false;
    GLint length;
    if (prog->vertexShader->isAttached)
    {
        fprintf(stdout, "Deleting/deattaching existing vertex shader, preparing for %s\n", prog->vertexShader->shaderName.c_str());
        if (glIsShader(prog->vertexShader->shaderID))
        {
            if (glIsProgram(prog->progID))
            {
                glDeleteShader(prog->vertexShader->shaderID);
                glGetShaderiv(prog->vertexShader->shaderID, GL_DELETE_STATUS, &length);
                if (length == 1)
                    fprintf(stdout, "Shader: %s ID: %d marked for deletion\n", prog->vertexShader->shaderName.c_str(), prog->vertexShader->shaderID);
                glDetachShader(prog->progID, prog->vertexShader->shaderID);
                prog->vertexShader->isAttached = false;
            }
        }
    }
    prog->vertexShader->isCompiled = false;
    GL::checkGLErrors("Existing shaders detached and deleted: creating new shader");
    compileVertSuccess = loadAndCompileShader(prog->vertexShader);
}

bool ShaderUtils::loadAndCompileShader(ShaderProgram::Shader *shaderInfo)
{
    GLint length;
    vector<string> *shaderSource = loadShaderSourceFile("Shaders/"+shaderInfo->shaderName);
}

vector<string> *ShaderUtils::loadShaderSourceFile(string fileName)
{
    vector<string> *shaderSource = new vector<string>;
    ifstream data(fileName);
    string line;
    if (data.is_open())
    {
        while (getline(data, line))
            shaderSource->push_back(line);
        data.close();
        return shaderSource;
    }
    else
        fprintf(stderr, "Fatal error loading shader source file\n");
    return nullptr;
}
