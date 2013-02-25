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

Double3D::Double3D(Double3D *from)
{
    x = from->x;
    y = from->y;
    z = from->z;
}

Double3D *Double3D::minus(Double3D *t1)
{
    Double3D *ans = new Double3D();
    ans->x = x - t1->x;
    ans->y = y - t1->y;
    ans->z = z - t1->z;
    return ans;
}

Double3D *Double3D::cross(Double3D *t1)
{
    Double3D *ans = new Double3D();
    ans->x = (y)*(t1->z)-(t1->y)*(z);
    ans->y = (z)*(t1->x)-(t1->z)*(x);
    ans->z = (x)*(t1->y)-(t1->x)*(y);
    return ans;
}

Double3D *Double3D::plus(Double3D *t1)
{
    Double3D *ans = new Double3D();

    ans->x = x + t1->x;
    ans->y = y + t1->y;
    ans->z = z + t1->z;

    return ans;
}

Double3D *Double3D::getUnit()
{
    Double3D *unit = new Double3D();
    double s = sqrt(x*x+y*y+z*z);
    if (s > 0)
    {
        unit->x = x / s;
        unit->y = y / s;
        unit->z = z / s;
    }
    return unit;
}

float Double3D::distanceTo(Double3D *point)
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
