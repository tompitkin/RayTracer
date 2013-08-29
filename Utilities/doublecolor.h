#ifndef DOUBLECOLOR_H
#define DOUBLECOLOR_H

#include <cuda_runtime.h>

class DoubleColor
{
public:

    double r;
    double g;
    double b;
    double a;

    __host__ __device__ DoubleColor()
    {
        r = 0.0f;
        g = 0.0f;
        b = 0.0f;
        a = 1.0;
    }

    __host__ __device__ DoubleColor(double nR, double nG, double nB, double nA)
    {
        r = nR;
        g = nG;
        b = nB;
        a = nA;
    }

    __host__ __device__ void plus(DoubleColor other)
    {
        r = r + other.r;
        g = g + other.g;
        b = b + other.b;
    }

    __host__ __device__ void scale(double scaleValue)
    {
        r *= scaleValue;
        g *= scaleValue;
        b *= scaleValue;
    }

    __host__ __device__ float *toFloatv()
    {
        float *rtn = new float[4];
        rtn[0] = r;
        rtn[1] = g;
        rtn[2] = b;
        rtn[3] = a;
        return rtn;
    }
};

#endif // DOUBLECOLOR_H
