#include "double3d.h"

Double3D::Double3D()
{
    x = 0.0f;
    y = 0.0f;
    z = 0.0f;
}

Double3D::Double3D(double nX, double nY, double nZ)
{
    x = nX;
    y = nY;
    z = nZ;
}

Double3D::Double3D(const Double3D *from)
{
    x = from->x;
    y = from->y;
    z = from->z;
}

Double3D Double3D::minus(Double3D t1)
{
    Double3D ans;
    ans.x = x - t1.x;
    ans.y = y - t1.y;
    ans.z = z - t1.z;
    return ans;
}

Double3D Double3D::cross(Double3D t1)
{
    Double3D ans;
    ans.x = (y)*(t1.z)-(t1.y)*(z);
    ans.y = (z)*(t1.x)-(t1.z)*(x);
    ans.z = (x)*(t1.y)-(t1.x)*(y);
    return ans;
}

Double3D Double3D::plus(Double3D t1)
{
    Double3D ans;

    ans.x = x + t1.x;
    ans.y = y + t1.y;
    ans.z = z + t1.z;

    return ans;
}

Double3D Double3D::sDiv(double s)
{
    Double3D ans;
    ans.x = x/s;
    ans.y = y/s;
    ans.z = z/s;
    return ans;
}

Double3D Double3D::sMult(double s)
{
    Double3D ans;
    ans.x = s*x;
    ans.y = s*y;
    ans.z = s*z;
    return ans;
}

Double3D Double3D::preMultiplyMatrix(vector<double> m)
{
    Double3D t;
    t.x = (m[0] * x + m[4] * y + m[8] * z + m[12]);
    t.y = (m[1] * x + m[5] * y + m[9] * z + m[13]);
    t.z = (m[2] * x + m[6] * y + m[10] * z + m[14]);
    return t;
}

Double3D Double3D::getUnit()
{
    Double3D unit;
    double s = sqrt(x*x+y*y+z*z);
    if (s > 0)
    {
        unit.x = x / s;
        unit.y = y / s;
        unit.z = z / s;
    }
    return unit;
}

double Double3D::dot(Double3D t1)
{
    return (x)*(t1.x) + (y)*(t1.y) + (z)*(t1.z);
}

float Double3D::distanceTo(Double3D point)
{
    Double3D newVect = this->minus(point);
    float s = (float)sqrt(newVect.x * newVect.x + newVect.y * newVect.y + newVect.z + newVect.z);
    return s;
}

void Double3D::unitize()
{
    float s = sqrt(x*x+y*y+z*z);
    if (s > 0)
    {
        x = x/s;
        y = y/s;
        z = z/s;
    }
}
