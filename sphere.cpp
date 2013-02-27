#include "sphere.h"
#include "pmesh.h"

Sphere::Sphere(Double3D cent, double rad, PMesh *ownerObj)
{
    center = cent;
    radius = rad;
    radiusSq = radius * radius;
    theObj = ownerObj;
}
