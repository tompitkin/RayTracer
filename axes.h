#ifndef AXES_H
#define AXES_H

#include "shaderprogram.h"

class Scene;

class Axes
{
public:
    Axes(Scene *top);
    virtual ~Axes();

    Scene *theScene;
    ShaderProgram *axesShaderProg;
};

#endif // AXES_H
