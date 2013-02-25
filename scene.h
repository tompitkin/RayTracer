#ifndef SCENE_H
#define SCENE_H

#include <stdio.h>
#include <GL/glew.h>
#include <QGLWidget>
#include <vector>
#include "camera.h"
#include "pmesh.h"
#include "lights.h"
#include "shaderprogram.h"
#include "shaderutils.h"
#include "axes.h"
#include "Loaders/objecttypes.h"
#include "Loaders/objloaderbuffer.h"

class Scene : public QGLWidget
{
    Q_OBJECT
public:
    explicit Scene(QWidget *parent = 0);
    virtual ~Scene();

    void setupUniforms(ShaderProgram *theShader);
    void adjustWindowAspect();
    void addObject(QString fileName, int fileType);

    PMesh *curObject;
    Camera *camera;
    Lights *lights;
    Axes *axes = nullptr;
    ShaderProgram *shaderProg;
    ShaderUtils shUtil = ShaderUtils(this);
    vector<PMesh*> objects;
    vector<ShaderProgram*> shaders;
    vector<bool> updateLights;
    string vertShaderName = "";
    string fragShaderName = "";
    string defaultVertShader = "multi.vs";
    string defaultFragShader = "multi.vs";
    bool drawAxis = true;
    bool updateLight;
    bool updateShaders = true;
    float clearColorR = 0.0f;
    float clearColorG = 0.0f;
    float clearColorB = 0.0f;
    float clearColorA = 1.0f;
    int numLoaded = 0;

protected:
    void initializeGL();
    void resizeGL(int x, int h);
    void paintGL();
    
};

#endif // SCENE_H
