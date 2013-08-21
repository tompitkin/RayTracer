#ifndef CUDAKERNEL_H
#define CUDAKERNEL_H

#include <cuda_runtime.h>
#include <stdio.h>
#include "MatrixManipulation/double3d.h"

struct Bitmap
{
    unsigned char *data;
    double pixelWidth, pixelHeight;
    int width, height;
    Double3D firstPixel;

    Bitmap(double viewWidth, double viewHeight, double windowWidth, double windowHeight, double windowLeft, double windowBottom, double nearPlane)
    {
        width = viewWidth;
        height = viewHeight;
        pixelWidth = windowWidth/viewWidth;
        pixelHeight = windowHeight/viewHeight;
        data = new unsigned char[width * height * 3];
        firstPixel = Double3D(windowLeft+width/2, windowBottom+height/2, nearPlane);
    }

    ~Bitmap()
    {
        delete [] data;
    }
};

struct Ray
{
    Double3D Rd;
    Double3D Ro;
    int flags;
};

void cudaStart(Bitmap *bitmap);
void checkErrors(cudaError_t *error, const char *file, int line);

#endif // CUDAKERNEL_H
