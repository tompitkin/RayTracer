#ifndef SHADERPROGRAM_H
#define SHADERPROGRAM_H

#include <GL/glew.h>
#include <vector>
#include <memory>
#include "Uniforms/uniform.h"

using namespace std;

class ShaderProgram
{
public:
    class Shader
    {
    public:
        Shader(string name, bool compiled, bool attached, int id, int type, shared_ptr<ShaderProgram> prog);
        virtual ~Shader();

        int type;
        string shaderName;
        bool isCompiled;
        bool isAttached;
        int shaderID;
    };

    ShaderProgram();
    virtual ~ShaderProgram();

    void addUniform(shared_ptr<Uniform> toAdd);

    GLuint progID;
    vector<shared_ptr<Uniform>> uniformList;
    string baseName;
    Shader *vertexShader;
    Shader *fragmentShader;
};

#endif // SHADERPROGRAM_H
