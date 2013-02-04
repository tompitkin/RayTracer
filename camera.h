#ifndef CAMERA_H
#define CAMERA_H

#include "matrixops.h"

class Camera
{
public:
    Camera();
    virtual ~Camera();

private:
    double *makeViewingTransform();

    double windowWidth;
    double windowHeight;
    double windowRight = 5.0;
    double windowLeft = -5.0;
    double windowTop = 5.0;
    double windowBottom = -5.0;

    double viewportWidth;
    double viewportHeight;
    double viewportRight = 600.0;
    double viewportLeft = 0.0;
    double viewportTop = 600.0;
    double viewportBottom = 0.0;

    double *viewMat;
};

#endif // CAMERA_H
