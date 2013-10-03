#include "cudaKernel.h"
#include "stdio.h"

__device__ bool intersectSphere(Ray *ray, float radiusSq, float3 viewCenter, float *t)
{
    const float EPS = 0.00001;
    float t0=0.0, t1=0.0, A=0.0, B=0.0, C=0.0, discrim=0.0;
    float3 RoMinusSc = ray->Ro - viewCenter;
    float fourAC = 0.0;

    A = dot(ray->Rd, ray->Rd);
    B = 2.0 * dot(ray->Rd, RoMinusSc);
    C = dot(RoMinusSc, RoMinusSc) - radiusSq;
    fourAC = (4*A*C);

    discrim = (B*B) - fourAC;

    if (discrim < EPS)
        return false;
    else
    {
        t0 = ((-B) - sqrt(discrim))/(2.0*A);
        t1 = ((-B) + sqrt(discrim))/(2.0*A);

        if (t0 < EPS)
        {
            if (t1 < EPS)
            {
                *t = 0.0;
                return false;
            }
            else
            {
                *t = t1;
                return true;
            }
        }
        else if (t1 < EPS)
        {
            *t = t0;
            return true;
        }
        else if (t0 < t1)
        {
            *t = t0;
            return true;
        }
        else
        {
            *t = t1;
            return true;
        }
    }
}

__device__ bool intersectTriangle(Ray *ray, float3 *v, float3 *n, HitRecord *hrec)
{
    float3 edges[2];
    float3 pvec, qvec, tvec;
    float det, inv_det;
    const float EPSILON = 0.000001;

    edges[0] = v[1] - v[0];
    edges[1] = v[2] - v[0];
    pvec = cross(ray->Rd, edges[1]);
    det = dot(edges[0], pvec);

    if (det > -EPSILON && det < EPSILON)
        return false;
    inv_det = 1.0/det;
    tvec = ray->Ro - v[0];
    hrec->u = dot(tvec, pvec) * inv_det;
    if (hrec->u < 0.0 || hrec->u > 1.0)
        return false;
    qvec = cross(tvec, edges[0]);
    hrec->v = dot(ray->Rd, qvec) * inv_det;
    if (hrec->v < 0.0 || hrec->u + hrec->v > 1.0)
        return false;
    if (det < -EPSILON)
        hrec->backfacing = true;
    else
        hrec->backfacing = false;
    hrec->t = dot(edges[1], qvec) * inv_det;

    if (hrec->t < EPSILON)
        return false;
    else
    {
        hrec->intersectPoint = make_float3((ray->Ro.x + (ray->Rd.x * hrec->t)), (ray->Ro.y + (ray->Rd.y * hrec->t)), (ray->Ro.z + (ray->Rd.z * hrec->t)));
        float w = 1.0 - hrec->u - hrec->v;

        float3 sumNorms = n[2] * hrec->v;
        sumNorms = n[1] * hrec->u + sumNorms;
        sumNorms = n[0] * w + sumNorms;
        hrec->normal = sumNorms;
        hrec->normal = normalize(hrec->normal);
        return true;
    }
}

void cudaStart(Bitmap *bitmap, Mesh *objects, int numObjects, LightCuda *lights, int numLights, Options *options)
{
    int numRays;
    unsigned char *d_bitmap;
    unsigned char *h_bitmap;
    unsigned char *layers[options->maxRecursiveDepth + 1];
    bool *hits;
    Mesh *d_objects;
    Mesh *h_objects;
    LightCuda *d_lights;
    Ray *rays;
    Intersect *intersects;

    cudaEvent_t start, stop;
    CHECK_ERROR(cudaEventCreate(&start));
    CHECK_ERROR(cudaEventCreate(&stop));
    CHECK_ERROR(cudaEventRecord(start, 0));

    CHECK_ERROR(cudaMalloc((void**)&d_bitmap, bitmap->width * bitmap->height * 3));
    h_bitmap = (unsigned char*)malloc(sizeof(unsigned char) * (bitmap->width * bitmap->height * 3));

    bitmap->data = d_bitmap;

    h_objects = (Mesh *)malloc(sizeof(Mesh) * numObjects);
    memcpy(h_objects, objects, sizeof(Mesh) * numObjects);

    for (int x = 0; x < numObjects; x++)
    {
        h_objects[x].surfaces = new Surface[h_objects[x].numSurfs];
        memcpy(h_objects[x].surfaces, objects[x].surfaces, sizeof(Surface) * objects[x].numSurfs);
        for (int y = 0; y < h_objects[x].numSurfs; y++)
        {
            CHECK_ERROR(cudaMalloc((void**)&h_objects[x].surfaces[y].vertArray, sizeof(float3) * h_objects[x].surfaces[y].numVerts));
            CHECK_ERROR(cudaMemcpy(h_objects[x].surfaces[y].vertArray, objects[x].surfaces[y].vertArray, sizeof(float3) * h_objects[x].surfaces[y].numVerts, cudaMemcpyHostToDevice));
            CHECK_ERROR(cudaMalloc((void**)&h_objects[x].surfaces[y].viewNormArray, sizeof(float3) * h_objects[x].surfaces[y].numVerts));
            CHECK_ERROR(cudaMemcpy(h_objects[x].surfaces[y].viewNormArray, objects[x].surfaces[y].viewNormArray, sizeof(float3) * h_objects[x].surfaces[y].numVerts, cudaMemcpyHostToDevice));
            delete [] objects[x].surfaces[y].vertArray;
            delete [] objects[x].surfaces[y].viewNormArray;
            objects[x].surfaces[y].vertArray = h_objects[x].surfaces[y].vertArray;
            objects[x].surfaces[y].viewNormArray = h_objects[x].surfaces[y].viewNormArray;
            h_objects[x].surfaces[y].vertArray = NULL;
            h_objects[x].surfaces[y].viewNormArray = NULL;
        }

        delete [] h_objects[x].surfaces;
        CHECK_ERROR(cudaMalloc((void **)&h_objects[x].surfaces, sizeof(Surface) * h_objects[x].numSurfs));
        CHECK_ERROR(cudaMemcpy(h_objects[x].surfaces, objects[x].surfaces, sizeof(Surface) * h_objects[x].numSurfs, cudaMemcpyHostToDevice));
        CHECK_ERROR(cudaMalloc((void **)&h_objects[x].materials, sizeof(Material) * h_objects[x].numMats));
        CHECK_ERROR(cudaMemcpy(h_objects[x].materials, objects[x].materials, sizeof(Material) * h_objects[x].numMats, cudaMemcpyHostToDevice));
    }

    CHECK_ERROR(cudaMalloc((void**)&d_objects, sizeof(Mesh) * numObjects));
    CHECK_ERROR(cudaMemcpy(d_objects, h_objects, sizeof(Mesh) * numObjects, cudaMemcpyHostToDevice));

    CHECK_ERROR(cudaMalloc((void**)&d_lights, sizeof(LightCuda) * numLights));
    CHECK_ERROR(cudaMemcpy(d_lights, lights, sizeof(LightCuda) * numLights, cudaMemcpyHostToDevice));

    numRays = bitmap->width * bitmap->height;
    CHECK_ERROR(cudaMalloc((void**)&rays, sizeof(Ray) * numRays));

    CHECK_ERROR(cudaMalloc((void**)&intersects, sizeof(Intersect) * numRays));

    CHECK_ERROR(cudaMalloc((void**)&hits, sizeof(bool) * numRays));

    for(int i = 0; i <= options->maxRecursiveDepth; i++)
    {
        CHECK_ERROR(cudaMalloc((void**)&layers[i], sizeof(unsigned char) * (bitmap->width * bitmap->height * 4)));
        CHECK_ERROR(cudaMemset(layers[i], 0, sizeof(unsigned char) * bitmap->width * bitmap->height * 4));
    }

    dim3 blocks((bitmap->width+15)/16, (bitmap->height+15)/16);
    dim3 threads(16, 16);
    baseKrnl<<<blocks, threads>>>(rays, *bitmap);
    for(int pass = 0; pass <= options->maxRecursiveDepth; pass++)
    {
        initIntersectKrnl<<<blocks, threads>>>(numRays, intersects);
        intersectSphereKrnl<<<blocks, threads>>>(rays, numRays, d_objects, numObjects, options->spheresOnly, intersects, hits);
        if (!options->spheresOnly)
        {
            for (int obj = 0; obj < numObjects; obj++)
            {
                for (int surf = 0; surf < objects[obj].numSurfs; surf++)
                {
                    for (int offset = 0; offset < (int)ceil(((float)objects[obj].surfaces[surf].numVerts) / CHUNK); offset++)
                    {
                        intersectTriangleKrnl<<<blocks, threads>>>(rays, numRays, intersects, hits, &d_objects[obj], &objects[obj].surfaces[surf].vertArray[offset * CHUNK], &objects[obj].surfaces[surf].viewNormArray[offset * CHUNK], (objects[obj].surfaces[surf].numVerts - offset * CHUNK) < CHUNK ? (objects[obj].surfaces[surf].numVerts - offset * CHUNK) : CHUNK , objects[obj].surfaces[surf].material);
                    }
                }
            }
        }
        shadeKrnl<<<blocks, threads>>>(rays, numRays, intersects, layers[pass], d_lights, numLights, *options, pass == options->maxRecursiveDepth ? true : false);
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

    for (int x = 0; x < numObjects; x++)
    {
        for (int y = 0; y < h_objects[x].numSurfs; y++)
        {
            CHECK_ERROR(cudaFree(objects[x].surfaces[y].vertArray));
            CHECK_ERROR(cudaFree(objects[x].surfaces[y].viewNormArray));
            objects[x].surfaces[y].vertArray = NULL;
            objects[x].surfaces[y].viewNormArray = NULL;
        }
        CHECK_ERROR_FREE(cudaFree(h_objects[x].surfaces), &h_objects[x].surfaces);
        CHECK_ERROR_FREE(cudaFree(h_objects[x].materials), &h_objects[x].materials);
    }
    CHECK_ERROR_FREE(cudaFree(d_objects), &d_objects);

    CHECK_ERROR_FREE(cudaFree(d_lights), &d_lights);

    CHECK_ERROR_FREE(cudaFree(rays), &rays);

    CHECK_ERROR_FREE(cudaFree(intersects), &intersects);

    CHECK_ERROR_FREE(cudaFree(hits), &hits);

    for(int i = 0; i <= options->maxRecursiveDepth; i++)
        CHECK_ERROR_FREE(cudaFree(layers[i]), &layers[i]);

    bitmap->data = h_bitmap;

    free(h_objects);
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
        intrs[offset].theObj = NULL;
        intrs[offset].distance = 100000000.0;
    }
}

__global__ void intersectSphereKrnl(Ray *rays, int numRays, Mesh *objects, int numObjects, bool spheresOnly, Intersect *intrs, bool *hits)
{
    //Map from threadIdx & blockIdx to pixel position
    int offset = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;

    __shared__ Ray R[16][16];

    if (offset < numRays)
    {
        R[threadIdx.y][threadIdx.x] = rays[offset];

        float t = 0.0;
        float intersectDist = 0.0;
        float minDist = 100000000.0;
        float3 intersectPt;
        float3 normal;
        Mesh *theObj;

        for (int obj = 0; obj < numObjects; obj++)
        {
            theObj = &objects[obj];
            if (intersectSphere(&R[threadIdx.y][threadIdx.x], theObj->boundingSphere.radiusSq, theObj->viewCenter, &t))
            {
                if (abs(t) < 0.0001)
                    continue;
                if (spheresOnly)
                {
                    intersectPt = make_float3((R[threadIdx.y][threadIdx.x].Ro.x+(R[threadIdx.y][threadIdx.x].Rd.x*t)), (R[threadIdx.y][threadIdx.x].Ro.y+(R[threadIdx.y][threadIdx.x].Rd.y*t)), (R[threadIdx.y][threadIdx.x].Ro.z+(R[threadIdx.y][threadIdx.x].Rd.z*t)));
                    normal = ((intersectPt - theObj->viewCenter) / theObj->boundingSphere.radius);
                    normal = normalize(normal);
                    intersectDist = length(make_float3(0.0, 0.0, 0.0) - intersectPt);
                    if (intersectDist < minDist)
                    {
                        minDist = intersectDist;
                        intrs[offset] = Intersect(0, false, theObj, intersectPt, normal, minDist);
                    }
                }
                else
                {
                    hits[offset] = true;
                }
            }
            else if (!spheresOnly)
            {
                hits[offset] = false;
            }
        }
    }
}

__global__ void intersectTriangleKrnl(Ray *rays, int numRays, Intersect *intrs, bool *hits, Mesh *theObj, float3 *verts, float3 *norms, int numVerts, int mat)
{
    int offset = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;

    __shared__ float3 V[CHUNK];
    __shared__ float3 N[CHUNK];
    __shared__ Ray R[16][16];

    if (offset < numRays)
    {
        R[threadIdx.y][threadIdx.x] = rays[offset];
        float intersectDist = 0.0;
        float minDist = intrs[offset].distance;
        int index = (threadIdx.y + threadIdx.x * 16) % numVerts;

        V[index] = verts[index];
        N[index] = norms[index];

        __syncthreads();

        if (hits[offset])
        {
            for (int i =  0; i < (numVerts / 3); i++)
            {
                HitRecord hrec;
                if (intersectTriangle(&R[threadIdx.y][threadIdx.x], &V[i * 3], &N[i * 3], &hrec))
                {
                    intersectDist = length(R[threadIdx.y][threadIdx.x].Ro - hrec.intersectPoint);
                    if (intersectDist < minDist)
                    {
                        minDist = intersectDist;
                        intrs[offset] = Intersect(mat, hrec.backfacing, theObj, hrec.intersectPoint, hrec.normal, minDist);
                    }
                }
            }
        }
    }
}

__global__ void shadeKrnl(Ray *rays, int numRays, Intersect *intrs, unsigned char *layer, LightCuda *lights, int numLights, Options options, bool finalPass)
{
    //Map from threadIdx & blockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    if (offset < numRays && intrs[offset].theObj != NULL)
    {
        int materialIndex = intrs[offset].materialIndex;
        Mesh *theObj = intrs[offset].theObj;
        float3 Ka = make_float3(0.0, 0.0, 0.0);
        float3 Kd = make_float3(0.0, 0.0, 0.0);
        float3 Ks = make_float3(0.0, 0.0, 0.0);
        float3 shadeColor = make_float3(0.0, 0.0, 0.0);
        float3 ambColor = make_float3(0.0, 0.0, 0.0);
        float3 point = intrs[offset].point;
        float3 trueNormal = make_float3(0.0, 0.0, 0.0);
        float3 inv_normal = intrs[offset].normal * -1.0;
        float3 R = make_float3(0.0, 0.0, 0.0);
        float3 L = make_float3(0.0, 0.0, 0.0);
        float3 V = make_float3(0.0, 0.0, 0.0);

        layer[offset*4 + 3] = (int) (theObj->materials[materialIndex].reflectivity.x * 255);

        Ka = theObj->materials[materialIndex].ka;
        Kd = theObj->materials[materialIndex].kd;
        Ks = theObj->materials[materialIndex].ks;

        ambColor.x = Ka.x * lights[0].ambient.x;
        ambColor.y = Ka.y * lights[0].ambient.y;
        ambColor.z = Ka.z * lights[0].ambient.z;

        shadeColor = shadeColor + ambColor;
        V = make_float3(0.0, 0.0, 0.0) - point;
        V = normalize(V);

        if (rays[offset].flags == EYE && intrs[offset].backFacing)
            trueNormal = inv_normal;
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
                cosPhiPower = pow(RdotV, theObj->materials[materialIndex].shiny);
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
