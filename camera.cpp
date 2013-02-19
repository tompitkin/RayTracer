#include "camera.h"

Camera::Camera()
{
    windowWidth = windowRight - windowLeft;
    windowHeight = windowTop - windowBottom;
    viewportWidth = viewportRight - viewportLeft;
    viewportHeight = viewportTop - viewportBottom;
    viewMat = makeViewingTransform();
    projMat = makePersepctiveTransform();
    viewMatUniform = new Matrix4Uniform(nullptr, "viewMat");
    projMatUniform = new Matrix4Uniform(nullptr, "projMat");
    invCamUniform = new Matrix4Uniform(nullptr, "mInverseCamera");
}

Camera::~Camera()
{
    delete []viewMat;
    delete []projMat;
    delete viewMatUniform;
    delete projMatUniform;
    delete invCamUniform;
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

double *Camera::makePersepctiveTransform()
{
    double *trans = MatrixOps::newIdentity();
    trans[0] = (2.0f * near)/(windowRight - windowLeft);
    trans[5] = (2.0f * near)/(windowTop - windowBottom);
    trans[8] = (windowRight + windowLeft)/(windowRight - windowLeft);
    trans[9] = (windowTop + windowBottom)/(windowTop - windowBottom);
    trans[10] = -(far + near)/(far - near);
    trans[11] = -1.0f;
    trans[14] = (-2.0f * far * near)/(far - near);
    trans[15] = 0.0f;
    return trans;
}
