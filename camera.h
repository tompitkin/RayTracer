#ifndef CAMERA_H
#define CAMERA_H

#include "matrixops.h"
#include "double3d.h"

class Camera
{
public:
    Camera();
    virtual ~Camera();

private:
    double *makeViewingTransform();

    Double3D eye = Double3D(0.0, 0.0, 10.0);
    Double3D center = Double3D(0.0, 0.0, 0.0);
    Double3D up = Double3D(0.0, 1.0, 0.0);
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
