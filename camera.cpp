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

void Camera::updateCamera(ShaderProgram *shaderProg)
{
    if (frustumChanged || perspectiveChanged)
    {
        projMat = makePersepctiveTransform();
        if (projMatUniform->theBuffer == nullptr)
        {
            projMatUniform->theBuffer = new GLfloat[16];
            copy(projMat, projMat+16, projMatUniform->theBuffer);
        }
        else
            copy(projMat, projMat+16, projMatUniform->theBuffer);
        projMatUniform->needsUpdate = true;
        frustumChanged = false;
        perspectiveChanged = false;
    }
    if (cameraMoved)
    {
        viewMat = makeViewingTransform();
        if (viewMatUniform->theBuffer == nullptr)
        {
            viewMatUniform->theBuffer = new GLfloat[16];
            copy(viewMat, viewMat+16, viewMatUniform->theBuffer);
        }
        else
            copy(viewMat, viewMat+16, viewMatUniform->theBuffer);
        viewMatUniform->needsUpdate = true;
        cameraMoved = false;
    }
}

void Camera::setViewport(double left, double right, double top, double bottom)
{
    viewportLeft = left;
    viewportRight = right;
    viewportTop = top;
    viewportBottom = bottom;
    aspectRatio = (right-left)/(top-bottom);
    cameraMoved = true;
}

void Camera::setFrustum(double left, double right, double bottom, double top, double nr, double fr)
{
    windowLeft = left;
    windowRight = right;
    windowBottom = bottom;
    windowTop = top;
    windowWidth = right - left;
    windowHeight = top - bottom;
    near = nr;
    far = fr;
    viewingMode = FRUSTUM_MODE;
    frustumToPerspective();
    frustumChanged = true;
}

void Camera::frustumToPerspective()
{
    double difY = 0.0;
    if (windowTop < 0 && windowBottom < 0)
        difY = abs(windowBottom) - abs(windowTop);
    else if (windowTop > 0 && windowBottom < 0)
        difY = windowTop + abs(windowBottom);
    else
        difY = windowTop - windowBottom;
    double difX = 0.0;
    if (windowRight < 0 && windowLeft < 0)
        difX = abs(windowLeft) - abs(windowRight);
    else if (windowRight > 0 && windowLeft < 0)
        difX = windowRight + abs(windowLeft);
    else
        difX = windowRight - windowLeft;
    fov = atan((difY/2)/near) * 180 / M_PI;
    fov = 2 * fov;
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

double Camera::getWindowHeight()
{
    return windowTop - windowBottom;
}

double Camera::getWindowWidth()
{
    return windowRight - windowLeft;
}
