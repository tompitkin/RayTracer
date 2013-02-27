#ifndef AXES_H
#define AXES_H

#include "shaderprogram.h"
#include "Uniforms/vec4uniform.h"

class Scene;

class Axes
{
public:
    class Axis
    {
    public:
        Axis(char which, float height);
        virtual ~Axis();

        void draw(Axes *axes);

        float *vertices;
        GLuint idVAO;
        GLuint idBuffers;
        float *color;
        char whichone;
        float howbig;
        Vec4Uniform *colorUniform;
    };

    Axes(Scene *top);
    virtual ~Axes();

    void draw();

    Scene *theScene;
    shared_ptr<ShaderProgram> axesShaderProg;
    Axis *xAxis = nullptr;
    Axis *yAxis = nullptr;
    Axis *zAxis = nullptr;
};

#endif // AXES_H
