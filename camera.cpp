#include "camera.h"

Camera::Camera()
{
    windowWidth = windowRight - windowLeft;
    windowHeight = windowTop - windowBottom;
    viewportWidth = viewportRight - viewportLeft;
    viewportHeight = viewportTop - viewportBottom;
    viewMat = makeViewingTransform();
    projMat = makePersepctiveTransform();
    viewMatUniform = shared_ptr<Matrix4Uniform>(new Matrix4Uniform(nullptr, "viewMat"));
    projMatUniform = shared_ptr<Matrix4Uniform>(new Matrix4Uniform(nullptr, "projMat"));
    invCamUniform = shared_ptr<Matrix4Uniform>(new Matrix4Uniform(nullptr, "mInverseCamera"));
}

Camera::~Camera()
{
    viewMatUniform.reset();
    projMatUniform.reset();
    invCamUniform.reset();
}

void Camera::updateCamera()
{
    if (frustumChanged || perspectiveChanged)
    {
        projMat = makePersepctiveTransform();
        if (projMatUniform->theBuffer == nullptr)
            projMatUniform->theBuffer = new GLfloat[16];
        copy(projMat.begin(), projMat.begin()+16, projMatUniform->theBuffer);
        projMatUniform->needsUpdate = true;
        frustumChanged = false;
        perspectiveChanged = false;
    }
    if (cameraMoved)
    {
        viewMat = makeViewingTransform();
        if (viewMatUniform->theBuffer == nullptr)
            viewMatUniform->theBuffer = new GLfloat[16];
        copy(viewMat.begin(), viewMat.begin()+16, viewMatUniform->theBuffer);
        viewMatUniform->needsUpdate = true;

        if (!invCamUniform.get()->off)
        {
            if (invCamUniform.get()->theBuffer == nullptr)
            {
                inverseCamera = makeInverseCamera();
                invCamUniform.get()->theBuffer = new GLfloat[16];
            }
            copy(inverseCamera.begin(), inverseCamera.begin()+16, invCamUniform.get()->theBuffer);
        }
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
    fov = atan((difY/2)/near) * 180 / M_PI;
    fov = 2 * fov;
}

vector<double> Camera::makeViewingTransform()
{
    Double3D viewPlaneNormal = center.minus(&eye);
    viewPlaneNormal.unitize();

    Double3D xaxis = viewPlaneNormal.cross(&up);
    xaxis.unitize();
    up = xaxis.cross(viewPlaneNormal);
    up.unitize();

    vector<double> rotations = MatrixOps::newIdentity();
    rotations[0] = xaxis.x;
    rotations[4] = xaxis.y;
    rotations[8] = xaxis.z;
    rotations[1] = up.x;
    rotations[5] = up.y;
    rotations[9] = up.z;
    rotations[2] = -viewPlaneNormal.x;
    rotations[6] = -viewPlaneNormal.y;
    rotations[10] = -viewPlaneNormal.z;
    vector<double> trans = MatrixOps::newIdentity();
    trans[12] = -eye.x;
    trans[13] = -eye.y;
    trans[14] = -eye.z;
    vector<double> viewTransform = MatrixOps::multMat(rotations, trans);

    return viewTransform;
}

vector<double> Camera::makePersepctiveTransform()
{
    vector<double> trans = MatrixOps::newIdentity();
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

vector<double> Camera::makeInverseCamera()
{
    vector<double> invCam;
    vector<double> rotView(16, 0);
    for (int i = 0; i < 12; i++)
        rotView[i] = viewMat[i];
    rotView[12] = rotView[13] = rotView[14] = 0.0;
    rotView[15] = 1.0;
    invCam = MatrixOps::convertToMatrix1D(MatrixOps::inverse(MatrixOps::convertToMatrix2D(rotView)));
    return invCam;
}

double Camera::getWindowHeight()
{
    return windowTop - windowBottom;
}

double Camera::getWindowWidth()
{
    return windowRight - windowLeft;
}
