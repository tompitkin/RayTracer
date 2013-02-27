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

bool ShaderUtils::setupShaders(shared_ptr<ShaderProgram> prog)
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
    if (compileVertSuccess)
        prog->vertexShader->isCompiled = true;
    GL::checkGLErrors("Vertex Shader compiled");
    if (!glIsProgram(prog->progID))
        prog->progID = glCreateProgram();
    glAttachShader(prog->progID, prog->vertexShader->shaderID);
    GL::checkGLErrors("Vertex Shader attached");
    prog->vertexShader->isAttached = true;
    programChanged = true;

    if (prog->fragmentShader->isAttached)
    {
        if (glIsShader(prog->fragmentShader->shaderID))
        {
            if (glIsProgram(prog->progID))
            {
                glDetachShader(prog->progID, prog->fragmentShader->shaderID);
                prog->fragmentShader->isAttached = false;
            }
            glDeleteShader(prog->fragmentShader->shaderID);
            prog->fragmentShader->isCompiled = false;
            programChanged = true;
        }
    }

    compileFragSuccess = loadAndCompileShader(prog->fragmentShader);
    if (compileFragSuccess)
        prog->fragmentShader->isCompiled = true;
    GL::checkGLErrors("Fragment Shader compiled");
    if (!glIsProgram(prog->progID))
        prog->progID = glCreateProgram();
    glAttachShader(prog->progID, prog->fragmentShader->shaderID);
    GL::checkGLErrors("Fragment Shader attached");
    prog->fragmentShader->isAttached = true;
    programChanged = true;
    if (programChanged)
    {
        glLinkProgram(prog->progID);
        GL::checkGLErrors("Right After Linking");
        glUseProgram(prog->progID);
        programChanged = false;
    }

    GLint num;
    glGetProgramiv(prog->progID, GL_ATTACHED_SHADERS, &num);
    fprintf(stdout, "%d attached shaders\n", num);
    glGetProgramiv(prog->progID, GL_ACTIVE_UNIFORMS, &num);
    fprintf(stdout, "%d active uniforms\n", num);
    glValidateProgram(prog->progID);
    GLint valid;
    glGetProgramiv(prog->progID, GL_VALIDATE_STATUS, &valid);
    if (valid == 1)
    {
        fprintf(stdout, "Program is valid\n");
        prog->fragmentShader->isAttached = true;
        prog->vertexShader->isAttached = true;
    }
    else
        fprintf(stderr, "Program is not valid\n");
    glGetProgramiv(prog->progID, GL_INFO_LOG_LENGTH, &length);
    fprintf(stdout, "Program log length: %d\n", length);
    if (length > 0)
    {
        char *log = (char*)malloc(length);
        glGetProgramInfoLog(prog->progID, length, NULL, log);
        fprintf(stdout, "Program log: %s\n", log);
        free(log);
    }

    return true;
}

bool ShaderUtils::loadAndCompileShader(ShaderProgram::Shader *shaderInfo)
{
    GLint length;
    const GLchar *shaderSource = loadShaderSourceFile("Shaders/"+shaderInfo->shaderName, &length);
    shaderInfo->shaderID = glCreateShader(shaderInfo->type);
    if (shaderInfo->shaderID == 0)
    {
        fprintf(stderr, "loadAndCompileShader: Could not create new shader %s\n", shaderInfo->shaderName.c_str());
        return false;
    }
    else
    {
        glShaderSource(shaderInfo->shaderID, 1, &shaderSource, &length);
        free((GLchar *)shaderSource);
        glCompileShader(shaderInfo->shaderID);
        glGetShaderiv(shaderInfo->shaderID, GL_COMPILE_STATUS, &length);
        if (length == 1)
        {
            fprintf(stdout, "Compile of %s successful\n", shaderInfo->shaderName.c_str());
            shaderInfo->isCompiled = true;
        }
        else
        {
            fprintf(stderr, "Compile of %s failed\n", shaderInfo->shaderName.c_str());
            shaderInfo->isCompiled = false;
        }
        glGetShaderiv(shaderInfo->shaderID, GL_INFO_LOG_LENGTH, &length);
        fprintf(stdout, "Shader log length: %d\n", length);
        if (length > 0)
        {
            char *log = (char*)malloc(length);
            glGetShaderInfoLog(shaderInfo->shaderID, length, NULL, log);
            fprintf(stdout, "Shader %s log: %s\n",shaderInfo->shaderName.c_str(), log);
            free(log);
        }
    }
    return true;
}

GLchar *ShaderUtils::loadShaderSourceFile(string fileName, GLint *length)
{
    FILE *f = fopen(fileName.c_str(), "r");
    GLchar *buffer;

    if (!f)
    {
        fprintf(stderr, "Fatal error loading shader source file\n");
        return nullptr;
    }

    fseek(f, 0, SEEK_END);
    *length = ftell(f);
    fseek(f, 0, SEEK_SET);

    buffer = (GLchar *)malloc(*length+1);
    *length = fread(buffer, 1, *length, f);
    fclose(f);
    buffer[*length] = '\0';

    return buffer;
}
