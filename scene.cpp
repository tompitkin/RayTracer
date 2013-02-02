#include "scene.h"

Scene::Scene(QWidget *parent) :
    QGLWidget(parent)
{
}

void Scene::initializeGL()
{
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

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
}

void Scene::resizeGL(int x, int h)
{
}

void Scene::paintGL()
{
}
