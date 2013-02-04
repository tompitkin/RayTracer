#ifndef SCENE_H
#define SCENE_H

#include <stdio.h>
#include <GL/glew.h>
#include <QGLWidget>
#include "camera.h"

class Scene : public QGLWidget
{
    Q_OBJECT
public:
    explicit Scene(QWidget *parent = 0);

protected:
    void initializeGL();
    void resizeGL(int x, int h);
    void paintGL();

private:
    Camera camera;
    
signals:
    
public slots:
    
};

#endif // SCENE_H
