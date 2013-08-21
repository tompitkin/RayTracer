#include "cudaKernel.h"

__global__ void kernel(unsigned char *data, int width, int height)
{
    //Map from threadIdx & blockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    if (offset < (width * height))
    {
        //Set the pixel to red
        data[offset*3 + 0] = 255;
        data[offset*3 + 1] = 0;
        data[offset*3 + 2] = 0;
    }
}

void cudaStart(Bitmap *bitmap)
{
    unsigned char *dev_bitmap;
    cudaError_t error;

    error = cudaMalloc((void**)&dev_bitmap, bitmap->width * bitmap->height * 3);
    checkErrors(&error, __FILE__, __LINE__);

    dim3 blocks((bitmap->width+15)/16, (bitmap->height+15)/16);
    dim3 threads(16, 16);
    kernel<<<blocks, threads>>>(dev_bitmap, bitmap->width, bitmap->height);

    error = cudaMemcpy(bitmap->data, dev_bitmap, bitmap->width * bitmap->height * 3, cudaMemcpyDeviceToHost);
    checkErrors(&error, __FILE__, __LINE__);

    error = cudaFree(dev_bitmap);
    checkErrors(&error, __FILE__, __LINE__);
}

void checkErrors(cudaError_t *error, const char *file, int line)
{
    if (*error != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(*error), file, line);
        exit(EXIT_FAILURE);
    }
}
