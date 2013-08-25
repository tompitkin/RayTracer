#ifndef CUDAKERNEL_H
#define CUDAKERNEL_H

#include <cuda_runtime.h>
#include <stdio.h>
#include "MatrixManipulation/double3d.h"

#define CHECK_ERROR(err) checkError(err, __FILE__, __LINE__)

struct Bitmap
{
    unsigned char *data;
    double pixelWidth, pixelHeight;
    int width, height;
    Double3D firstPixel;
};

struct Ray
{
    Double3D Rd;
    Double3D Ro;
    int flags;
};

void cudaStart(Bitmap *bitmap);
void checkError(cudaError_t error, const char *file, int line);

#endif // CUDAKERNEL_H
