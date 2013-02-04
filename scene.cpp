#include "scene.h"

Scene::Scene(QWidget *parent) :
    QGLWidget(parent)
{
    //This is a work around for a bug in QT 5.0 that won't allow you
    //to use any OpenGL version >= 3.0
    QGLFormat format;
    format.setVersion(4, 3);
    QGLFormat::setDefaultFormat(format);
    this->setFormat(format);

    makeCurrent();
    GLenum err = glewInit();
    if (err != GLEW_OK)
    {
        fprintf(stderr, "GLEW Error: %s\n", glewGetErrorString(err));
    }
    fprintf(stdout, "Driver Version String: %s\n", glGetString(GL_VERSION));
    if (glewIsSupported("GL_VERSION_3_0"))
        fprintf(stdout, "GL3 is available\n");
    else
    {
        fprintf(stdout, "GL3 is NOT available\n");
    }
}

void Scene::initializeGL()
{
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void Scene::resizeGL(int x, int h)
{
}

void Scene::paintGL()
{
}
