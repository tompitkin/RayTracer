#ifndef SHADERPROGRAM_H
#define SHADERPROGRAM_H

#include <GL/glew.h>
#include <vector>
#include "Uniforms/uniform.h"

using namespace std;

class ShaderProgram
{
public:
    class Shader
    {
    public:
        Shader(string name, bool compiled, bool attached, int id, int type);

        int type;
        string shaderName;
        bool isCompiled;
        bool isAttached;
        int shaderID;
    };

    ShaderProgram();
    virtual ~ShaderProgram();

    void addShader(Shader *shader);
    void addUniform(Uniform *toAdd);

    int progID;
    vector<Uniform*> uniformList;

private:
        string baseName;
        Shader *vertexShader;
        Shader *fragmentShader;
};

#endif // SHADERPROGRAM_H
