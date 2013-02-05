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
