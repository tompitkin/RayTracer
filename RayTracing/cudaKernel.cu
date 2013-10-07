#include "cudaKernel.h"
#include "stdio.h"

texture<float4, 1, cudaReadModeElementType> triangleTex;

void cudaStart(Bitmap *bitmap, int numObjects, BoundingSphere *spheres, Material *materials, int *numMaterials, float4 **triangles, int *numVerts, LightCuda *lights, int numLights, Options *options)
{
    int numRays = bitmap->width * bitmap->height;
    int matCount = 0;
    unsigned char *d_bitmap;
    unsigned char *h_bitmap = (unsigned char*)malloc(sizeof(unsigned char) * (bitmap->width * bitmap->height * 3));
    unsigned char *layers[options->maxRecursiveDepth + 1];
    float4 *d_triangles[numObjects];
    Material *d_materials;
    LightCuda *d_lights;
    Ray *rays;
    Intersect *intersects;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

    triangleTex.normalized = false;
    triangleTex.filterMode = cudaFilterModePoint;
    triangleTex.addressMode[0] = cudaAddressModeWrap;

    CHECK_ERROR(cudaMalloc((void**)&d_bitmap, sizeof(unsigned char) * bitmap->width * bitmap->height * 3));
    CHECK_ERROR(cudaMalloc((void**)&d_lights, sizeof(LightCuda) * numLights));
    CHECK_ERROR(cudaMalloc((void**)&rays, sizeof(Ray) * numRays));
    CHECK_ERROR(cudaMalloc((void**)&intersects, sizeof(Intersect) * numRays));
    CHECK_ERROR(cudaMalloc((void**)&d_materials, sizeof(Material)));
    for (int x = 0; x < numObjects; x++)
    {
        matCount += numMaterials[x];
        CHECK_ERROR(cudaMalloc((void**)&d_triangles[x], sizeof(float4) * numVerts[x]));
    }
    CHECK_ERROR(cudaMalloc((void **)&d_materials, sizeof(Material) * matCount));
    for(int x = 0; x <= options->maxRecursiveDepth; x++)
        CHECK_ERROR(cudaMalloc((void**)&layers[x], sizeof(unsigned char) * (bitmap->width * bitmap->height * 4)));

    CHECK_ERROR(cudaMemcpy(d_lights, lights, sizeof(LightCuda) * numLights, cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(d_materials, materials, sizeof(Material) * matCount, cudaMemcpyHostToDevice));
    for (int x = 0; x < numObjects; x++)
        CHECK_ERROR(cudaMemcpy(d_triangles[x], triangles[x], sizeof(float4) * numVerts[x], cudaMemcpyHostToDevice));
    for(int x = 0; x <= options->maxRecursiveDepth; x++)
        CHECK_ERROR(cudaMemset(layers[x], 0, sizeof(unsigned char) * bitmap->width * bitmap->height * 4));

    bitmap->data = d_bitmap;

    cudaEvent_t start, stop;
    CHECK_ERROR(cudaEventCreate(&start));
    CHECK_ERROR(cudaEventCreate(&stop));
    CHECK_ERROR(cudaEventRecord(start, 0));

    dim3 blocks((bitmap->width+15)/16, (bitmap->height+15)/16);
    dim3 threads(16, 16);

    baseKrnl<<<blocks, threads>>>(rays, *bitmap);
    for(int pass = 0; pass <= options->maxRecursiveDepth; pass++)
    {
        initIntersectKrnl<<<blocks, threads>>>(numRays, intersects);
        for (int obj = 0; obj < numObjects; obj++)
        {
            cudaBindTexture(0, triangleTex, d_triangles[obj], channelDesc, sizeof(float4) * numVerts[obj]);
            intersectKrnl<<<blocks, threads>>>(rays, numRays, spheres[obj].radius, spheres[obj].center, options->spheresOnly, intersects, numVerts[obj]);
        }
        //shadeKrnl<<<blocks, threads>>>(rays, numRays, intersects, d_materials, layers[pass], d_lights, numLights, pass == options->maxRecursiveDepth ? true : false);
        composeKrnl<<<blocks, threads>>>(*bitmap, layers[pass], pass == options->maxRecursiveDepth ? true : false);
    }

    CHECK_ERROR(cudaMemcpy(h_bitmap, d_bitmap, bitmap->width * bitmap->height * 3, cudaMemcpyDeviceToHost));

    CHECK_ERROR(cudaEventRecord(stop, 0));
    CHECK_ERROR(cudaEventSynchronize(stop));

    float time;
    CHECK_ERROR(cudaEventElapsedTime(&time, start, stop));

    printf("(CUDA) Ray Trace total time: %3.1f ms\n", time);

    CHECK_ERROR(cudaEventDestroy(start));
    CHECK_ERROR(cudaEventDestroy(stop));

    CHECK_ERROR_FREE(cudaFree(d_bitmap), &d_bitmap);
    CHECK_ERROR_FREE(cudaFree(d_lights), &d_lights);
    CHECK_ERROR_FREE(cudaFree(rays), &rays);
    CHECK_ERROR_FREE(cudaFree(intersects), &intersects);
    CHECK_ERROR_FREE(cudaFree(d_materials), &d_materials);
    for(int i = 0; i <= options->maxRecursiveDepth; i++)
        CHECK_ERROR_FREE(cudaFree(layers[i]), &layers[i]);

    bitmap->data = h_bitmap;
}

void checkError(cudaError_t error, const char *file, int line, void **nullObject)
{
    if (nullObject != NULL)
        nullObject = NULL;

    if (error != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(error), file, line);
        exit(EXIT_FAILURE);
    }
}

__global__ void baseKrnl(Ray *rays, Bitmap bitmap)
{
    //Map from threadIdx & blockIdx to pixel position
    int offset = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;

    if (offset < (bitmap.width * bitmap.height))
    {
        float3 point(bitmap.firstPixel);
        point.x += (offset % bitmap.width) * bitmap.pixelWidth;
        point.y += ((offset - (threadIdx.x + blockIdx.x * blockDim.x)) / bitmap.width) * bitmap.pixelHeight;
        rays[offset].Rd = normalize(point);
        rays[offset].Ro = make_float3(0.0, 0.0, 0.0);
        rays[offset].flags = EYE;
    }
}

__global__ void initIntersectKrnl(int numIntrs, Intersect *intrs)
{
    int offset = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;

    if (offset < numIntrs)
    {
        intrs[offset].hit = false;
        intrs[offset].distance = 100000000.0;
    }
}

__global__ void intersectKrnl( Ray *rays, const int numRays, const float sRadius, const float3 sCenter, const bool spheresOnly, Intersect *intrs, const int numVerts)
{
    //Map from threadIdx & blockIdx to pixel position
    const int offset = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;

    __shared__ Ray R[16][16];

    if (offset < numRays)
    {
        R[threadIdx.y][threadIdx.x] = rays[offset];

        float t = 0.0;
        float intersectDist = 0.0;
        float minDist = 100000000.0;
        float3 intersectPt;
        float3 normal;
        float4 v, e1, e2;

        bool sphereHit = (intersectSphere(R[threadIdx.y][threadIdx.x], sRadius, sCenter, t) && abs(t) >= EPSILON);

        if (sphereHit && !spheresOnly)
        {
            for (int i = 0; i < (numVerts/3); i++)
            {
                v = tex1Dfetch(triangleTex, i*3);
                e1 = tex1Dfetch(triangleTex, i*3+1);
                e2 = tex1Dfetch(triangleTex, i*3+2);

                t = intersectTriangle(R[threadIdx.y][threadIdx.x], make_float3(v.x, v.y, v.z), make_float3(e1.x, e1.y, e1.z), make_float3(e2.x, e2.y, e2.z));
            }
        }

        intersectPt = make_float3((R[threadIdx.y][threadIdx.x].Ro.x+(R[threadIdx.y][threadIdx.x].Rd.x*t)), (R[threadIdx.y][threadIdx.x].Ro.y+(R[threadIdx.y][threadIdx.x].Rd.y*t)), (R[threadIdx.y][threadIdx.x].Ro.z+(R[threadIdx.y][threadIdx.x].Rd.z*t)));
        intersectDist = length(make_float3(0.0, 0.0, 0.0) - intersectPt);

        if (spheresOnly && intersectDist < minDist)
        {
            normal = ((intersectPt - sCenter) / sRadius);
            normal = normalize(normal);
            intrs[offset] = Intersect(0, true, intersectPt, normal, minDist);
        }
        else if (!spheresOnly && intersectDist < minDist)
        {
            normal = cross(make_float3(e1.x, e1.y, e2.z), make_float3(e2.x, e2.y, e2.z));
            normal = normalize(normal);
            intrs[offset] = Intersect(v.w, true, intersectPt, normal, minDist);
        }
    }
}

__device__ bool intersectSphere(const Ray &ray, const float radius, const float3 viewCenter, float &t)
{
    float B, C, discrim;
    float3 RoMinusSc = ray.Ro - viewCenter;

    B = dot(RoMinusSc, ray.Rd);
    C = dot(RoMinusSc, RoMinusSc) - (radius * radius);

    discrim = (B*B) - C;

    if (discrim >= EPSILON)
    {
        float E = sqrt(discrim);
        float t0 = -B-E;
        if (t0 < EPSILON)
            t = -B+E;
        else
            t = min(-B-E, -B+E);
        return true;
    }
    return false;
}

__device__ float intersectTriangle(const Ray &ray, const float3 &v1, const float3 &e1, const float3 &e2)
{
    float3 tvec = ray.Ro - v1;
    float3 pvec = cross(ray.Rd, e2);
    float det = dot(e1, pvec);

    det = __fdividef(1.0f, det);

    float u = dot(tvec, pvec) * det;

    if (u < 0.0f || u > 1.0f)
        return -1.0f;

    float3 qvec = cross(tvec, e1);

    float v = dot(ray.Rd, qvec) * det;

    if (v < 0.0f || (u + v) > 1.0f)
        return -1.0f;

    return dot(e2, qvec) * det;
}

__global__ void shadeKrnl(Ray *rays, const int numRays, Intersect *intrs, Material *materials, unsigned char *layer, LightCuda *lights, int numLights, const bool finalPass)
{
    //Map from threadIdx & blockIdx to pixel position
    const int offset = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;

    if (offset < numRays && intrs[offset].hit == true)
    {
        const int materialIndex = intrs[offset].materialIndex;
        const float3 Ka = materials[materialIndex].ka;
        const float3 Kd = materials[materialIndex].kd;
        const float3 Ks = materials[materialIndex].ks;
        float3 shadeColor = make_float3(0.0, 0.0, 0.0);
        float3 ambColor = make_float3(0.0, 0.0, 0.0);
        const float3 point = intrs[offset].point;
        float3 trueNormal = make_float3(0.0, 0.0, 0.0);
        float3 R = make_float3(0.0, 0.0, 0.0);
        float3 L = make_float3(0.0, 0.0, 0.0);
        float3 V = make_float3(0.0, 0.0, 0.0);

        layer[offset*4 + 3] = (int) (materials[materialIndex].reflectivity.x * 255);

        ambColor.x = Ka.x * lights[0].ambient.x;
        ambColor.y = Ka.y * lights[0].ambient.y;
        ambColor.z = Ka.z * lights[0].ambient.z;

        shadeColor = shadeColor + ambColor;
        V = make_float3(0.0, 0.0, 0.0) - point;
        V = normalize(V);

        if (rays[offset].flags == EYE)
            trueNormal = intrs[offset].normal * -1.0;
        else
            trueNormal = intrs[offset].normal;

        LightCuda *curLight;
        for (int i = 0; i < numLights; i++)
        {
            curLight = &lights[i];

            L = curLight->viewPosition - point;
            L = normalize(L);
            float LdotN = dot(L, trueNormal);
            LdotN = max(0.0, LdotN);
            float3 diffComponent = make_float3(0.0, 0.0, 0.0);
            if (LdotN > 0.0)
                diffComponent = diffComponent + (make_float3(curLight->diffuse.x*Kd.x*LdotN, curLight->diffuse.y*Kd.y*LdotN, curLight->diffuse.z*Kd.z*LdotN));
            shadeColor = shadeColor + diffComponent;

            float3 Pr = trueNormal * LdotN;
            float3 sub = Pr * 2.0;
            R = L * -1.0 + sub;
            R = normalize(R);
            float RdotV = dot(R, V);
            RdotV = max(0.0, RdotV);
            float cosPhiPower = 0.0;
            if (RdotV > 0.0)
                cosPhiPower = pow(RdotV, materials[materialIndex].shiny);
            float3 specComponent = make_float3(curLight->specular.x*Ks.x*cosPhiPower, curLight->specular.y*Ks.y*cosPhiPower, curLight->specular.z*Ks.z*cosPhiPower);
            shadeColor = shadeColor + specComponent;
        }

        shadeColor.x = shadeColor.x < 0.0 ? 0.0 : (shadeColor.x > 1.0 ? 1.0 : shadeColor.x);
        shadeColor.y = shadeColor.y < 0.0 ? 0.0 : (shadeColor.y > 1.0 ? 1.0 : shadeColor.y);
        shadeColor.z = shadeColor.z < 0.0 ? 0.0 : (shadeColor.z > 1.0 ? 1.0 : shadeColor.z);

        if (finalPass)
        {
            layer[offset*4 + 0] = shadeColor.x * 255;
            layer[offset*4 + 1] = shadeColor.y * 255;
            layer[offset*4 + 2] = shadeColor.z * 255;
        }
    }
}

__global__ void composeKrnl(Bitmap bitmap, unsigned char *layer, bool finalPass)
{
    //Map from threadIdx & blockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    if (offset < (bitmap.width * bitmap.height))
    {
        if (finalPass)
        {
            bitmap.data[offset*3 + 0] = layer[offset*4 + 0];
            bitmap.data[offset*3 + 1] = layer[offset*4 + 1];
            bitmap.data[offset*3 + 2] = layer[offset*4 + 2];
        }
        else
        {

        }
    }

}
