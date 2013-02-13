#ifndef SCENE_H
#define SCENE_H

#include <stdio.h>
#include <GL/glew.h>
#include <QGLWidget>
#include <vector>
#include "camera.h"
#include "pmesh.h"
#include "lights.h"

class Scene : public QGLWidget
{
    Q_OBJECT
public:
    explicit Scene(QWidget *parent = 0);
    virtual ~Scene();

    vector<PMesh> *objects;
    PMesh *curObject;
    Camera *camera;
    Lights *lights;
    bool updateLight;
    vector<bool> updateLights;
protected:
    void initializeGL();
    void resizeGL(int x, int h);
    void paintGL();
    
signals:
    
public slots:
    
};

#endif // SCENE_H
