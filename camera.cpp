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
    Double3D viewPlaneNormal = center.minus(eye);
    viewPlaneNormal.unitize();

    Double3D xaxis = viewPlaneNormal.cross(up);
    xaxis.unitize();
    up = xaxis.cross(viewPlaneNormal);
    up.unitize();

    double *rotations = MatrixOps::newIdentity();
    rotations[0] = xaxis.x;
    rotations[4] = xaxis.y;
    rotations[8] = xaxis.z;
    rotations[1] = up.x;
    rotations[5] = up.y;
    rotations[9] = up.z;
    rotations[2] = -viewPlaneNormal.x;
    rotations[6] = -viewPlaneNormal.y;
    rotations[10] = -viewPlaneNormal.z;
    double *trans = MatrixOps::newIdentity();
    trans[12] = -eye.x;
    trans[13] = -eye.y;
    trans[14] = -eye.z;
    double *viewTransform = MatrixOps::multMat(rotations, trans);

    delete []rotations;
    delete []trans;

    return viewTransform;
}
