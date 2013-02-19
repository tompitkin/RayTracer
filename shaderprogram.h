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
        Shader(string name, bool compiled, bool attached, int id, int type, ShaderProgram *prog);

        int type;
        string shaderName;
        bool isCompiled;
        bool isAttached;
        int shaderID;
    };

    ShaderProgram();
    virtual ~ShaderProgram();

    void addUniform(Uniform *toAdd);

    int progID;
    vector<Uniform*> uniformList;
    string baseName;
    Shader *vertexShader;
    Shader *fragmentShader;
};

#endif // SHADERPROGRAM_H
