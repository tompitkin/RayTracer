#ifndef SCENE_H
#define SCENE_H

#include <stdio.h>
#include <GL/glew.h>
#include <QGLWidget>
#include <vector>
#include <memory>
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

    void setupUniforms(shared_ptr<ShaderProgram> theShader);
    void adjustWindowAspect();
    void addObject(QString fileName, int fileType);
    void deleteObject(int index);

    Camera *camera = nullptr;
    Lights *lights = nullptr;
    Axes *axes = nullptr;
    shared_ptr<PMesh> curObject;
    shared_ptr<ShaderProgram> shaderProg;
    ShaderUtils shUtil = ShaderUtils(this);
    vector<shared_ptr<PMesh>> objects;
    vector<shared_ptr<ShaderProgram>> shaders;
    vector<bool> updateLights;
    string vertShaderName = "";
    string fragShaderName = "";
    string defaultVertShader = "multi.vert";
    string defaultFragShader = "multi.frag";
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
