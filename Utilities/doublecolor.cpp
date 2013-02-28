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

float *DoubleColor::toFloatv()
{
    float *rtn = new float[4];
    rtn[0] = r;
    rtn[1] = g;
    rtn[2] = b;
    rtn[3] = a;
    return rtn;
}
