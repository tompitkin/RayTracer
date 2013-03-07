#include "doublecolor.h"

DoubleColor::DoubleColor()
{
    r = 0.0f;
    g = 0.0f;
    b = 0.0f;
    a = 1.0;
}

DoubleColor::DoubleColor(double nR, double nG, double nB, double nA)
{
    r = nR;
    g = nG;
    b = nB;
    a = nA;
}

void DoubleColor::plus(DoubleColor other)
{
    r = r + other.r;
    g = g + other.g;
    b = b + other.b;
}

void DoubleColor::scale(double scaleValue)
{
    r *= scaleValue;
    g *= scaleValue;
    b *= scaleValue;
}

float *DoubleColor::toFloatv()
{
    float *rtn = new float[4];
    rtn[0] = r;
    rtn[1] = g;
    rtn[2] = b;
    rtn[3] = a;
    return rtn;
}
