#include "camera.h"

Camera::Camera()
{
    windowWidth = windowRight - windowLeft;
    windowHeight = windowTop - windowBottom;
    viewportWidth = viewportRight - viewportLeft;
    viewportHeight = viewportTop - viewportBottom;
    viewMat = makeViewingTransform();
}

Camera::~Camera()
{
    delete []viewMat;
}

double *Camera::makeViewingTransform()
{
    double *rotations = MatrixOps::newIdentity();
    return rotations;
}
