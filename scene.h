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

class Scene : public QGLWidget
{
    Q_OBJECT
public:
    explicit Scene(QWidget *parent = 0);
    virtual ~Scene();

    void setupUniforms(ShaderProgram *theShader);

    vector<PMesh> *objects;
    PMesh *curObject;
    Camera *camera;
    Lights *lights;
    bool updateLight;
    vector<bool> updateLights;
    float clearColorR = 0.0f;
    float clearColorG = 0.0f;
    float clearColorB = 0.0f;
    float clearColorA = 1.0f;
    vector<ShaderProgram*> shaders;
    ShaderProgram *shaderProg;
    bool drawAxis = true;
    Axes *axes = nullptr;
    string vertShaderName = "";
    string fragShaderName = "";
    string defaultVertShader = "multi.vs";
    string defaultFragShader = "multi.vs";
    bool updateShaders = true;
    ShaderUtils shUtil = ShaderUtils(this);

protected:
    void initializeGL();
    void resizeGL(int x, int h);
    void paintGL();
    
signals:
    
public slots:
    
};

#endif // SCENE_H
