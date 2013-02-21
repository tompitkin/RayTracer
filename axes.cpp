#include "axes.h"
#include "scene.h"

Axes::Axes(Scene *top)
{
    theScene = top;
    axesShaderProg = new ShaderProgram();
    axesShaderProg->vertexShader = new ShaderProgram::Shader("axes.vert", false, false, -1, GL_VERTEX_SHADER, axesShaderProg);
    axesShaderProg->fragmentShader = new ShaderProgram::Shader("axes.frag", false, false, -1, GL_FRAGMENT_SHADER, axesShaderProg);
    theScene->shUtil.setupShaders(axesShaderProg);
    xAxis = new Axis('X', 1.0f);
    yAxis = new Axis('Y', 1.0f);
    zAxis = new Axis('Z', 1.0f);
}

Axes::~Axes()
{
    theScene = nullptr;
    delete axesShaderProg;
    delete xAxis;
    delete yAxis;
    delete zAxis;
}

void Axes::draw()
{
    if (xAxis != nullptr)
        xAxis->draw(this);
    if (yAxis != nullptr)
        yAxis->draw(this);
    if (zAxis != nullptr)
        zAxis->draw(this);
}


Axes::Axis::Axis(char which, float height)
{
    whichone = which;
    howbig = height;
    color = new float[4]{0.0f, 0.0f, 0.0f, 1.0f};
    colorUniform = new Vec4Uniform(nullptr, "axisColor");
    vertices = new float[6];
    vertices[0]=0.0f; vertices[1]=0.0f; vertices[2]=0.0f;
    if (whichone == 'X')
    {
        vertices[3] = howbig*1.0f;
        vertices[4] = 0.0f;
        vertices[5] = 0.0f;
        color[0]=1.0f; color[1]=0.0f; color[2]=0.0f; color[3]=1.0f;
    }
    else if (whichone == 'Y')
    {
        vertices[3] = 0.0f;
        vertices[4] = howbig*1.0f;
        vertices[5] = 0.0f;
        color[0]=0.0f; color[1]=1.0f; color[2]=0.0f; color[3]=1.0f;
    }
    else if (whichone == 'Z')
    {
        vertices[3] = 0.0f;
        vertices[4] = 0.0f;
        vertices[5] = howbig*1.0f;
        color[0]=0.0f; color[1]=0.0f; color[2]=1.0f; color[3]=1.0f;
    }
    else fprintf(stderr, "Axis: illegal value of whichone: %c", whichone);

    glGenVertexArrays(1, &idVAO);
    glGenBuffers(1, &idBuffers);
}

Axes::Axis::~Axis()
{
    delete []color;
    delete []vertices;
    delete colorUniform;
}

void Axes::Axis::draw(Axes *axes)
{
    glUseProgram(axes->axesShaderProg->progID);
    glBindVertexArray(idVAO);
    glBindBuffer(GL_ARRAY_BUFFER, idBuffers);
    glBufferData(GL_ARRAY_BUFFER, 6*sizeof(float), vertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, 0);

    GL::checkGLErrors("Enabled Vertices VBO");

    if (colorUniform->theBuffer == nullptr)
        colorUniform->theBuffer = new GLfloat[4];
    copy(color, color+4, colorUniform->theBuffer);
    colorUniform->update(axes->axesShaderProg->progID);

    axes->theScene->camera->viewMatUniform->update(axes->axesShaderProg->progID);
    axes->theScene->camera->projMatUniform->update(axes->axesShaderProg->progID);
    glDrawArrays(GL_LINES, 0, 2);
}
