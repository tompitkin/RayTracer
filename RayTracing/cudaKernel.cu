#include "cudaKernel.h"

__global__ void kernel(Bitmap bitmap)
{
    //Map from threadIdx & blockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    if (offset < (bitmap.width * bitmap.height))
    {
        //Set the pixel to red
        bitmap.data[offset*3 + 0] = 255;
        bitmap.data[offset*3 + 1] = 0;
        bitmap.data[offset*3 + 2] = 0;
    }
}

void cudaStart(Bitmap *bitmap)
{
    unsigned char *d_bitmap;
    unsigned char *h_bitmap;

    CHECK_ERROR(cudaMalloc((void**)&d_bitmap, bitmap->width * bitmap->height * 3));
    h_bitmap = (unsigned char*)malloc(sizeof(unsigned char) * (bitmap->width * bitmap->height * 3));

    bitmap->data = d_bitmap;

    dim3 blocks((bitmap->width+15)/16, (bitmap->height+15)/16);
    dim3 threads(16, 16);
    kernel<<<blocks, threads>>>(*bitmap);

    CHECK_ERROR(cudaMemcpy(h_bitmap, d_bitmap, bitmap->width * bitmap->height * 3, cudaMemcpyDeviceToHost));

    CHECK_ERROR(cudaFree(d_bitmap));

    bitmap->data = h_bitmap;
}

void checkError(cudaError_t error, const char *file, int line)
{
    if (error != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(error), file, line);
        exit(EXIT_FAILURE);
    }
}
