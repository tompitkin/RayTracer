#include "cudaKernel.h"
#include "stdio.h"

__device__ bool intersectSphere(Ray *ray, float radiusSq, Float3D viewCenter, float *t)
{
    const float EPS = 0.00001;
    float t0=0.0, t1=0.0, A=0.0, B=0.0, C=0.0, discrim=0.0;
    Float3D RoMinusSc = ray->Ro.minus(&viewCenter);
    float fourAC = 0.0;

    A = ray->Rd.dot(&ray->Rd);
    B = 2.0 * (ray->Rd.dot(&RoMinusSc));
    C = RoMinusSc.dot(&RoMinusSc) - radiusSq;
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

__device__ bool intersectTriangle(Ray *ray, Float3D *v, Float3D *n, HitRecord *hrec)
{
    Float3D edges[2];
    Float3D pvec, qvec, tvec;
    float det, inv_det;
    const float EPSILON = 0.000001;

    edges[0] = v[1].minus(&v[0]);
    edges[1] = v[2].minus(&v[0]);
    pvec = ray->Rd.cross(&edges[1]);
    det = edges[0].dot(&pvec);

    if (det > -EPSILON && det < EPSILON)
        return false;
    inv_det = 1.0/det;
    tvec = ray->Ro.minus(v);
    hrec->u = tvec.dot(&pvec) * inv_det;
    if (hrec->u < 0.0 || hrec->u > 1.0)
        return false;
    qvec = tvec.cross(&edges[0]);
    hrec->v = ray->Rd.dot(&qvec) * inv_det;
    if (hrec->v < 0.0 || hrec->u + hrec->v > 1.0)
        return false;
    if (det < -EPSILON)
        hrec->backfacing = true;
    else
        hrec->backfacing = false;
    hrec->t = edges[1].dot(&qvec) * inv_det;

    if (hrec->t < EPSILON)
        return false;
    else
    {
        hrec->intersectPoint = Float3D((ray->Ro.x + (ray->Rd.x * hrec->t)), (ray->Ro.y + (ray->Rd.y * hrec->t)), (ray->Ro.z + (ray->Rd.z * hrec->t)));
        float w = 1.0 - hrec->u - hrec->v;

        Float3D sumNorms = n[2].sMult(hrec->v);
        sumNorms = n[1].sMult(hrec->u).plus(&sumNorms);
        sumNorms = n[0].sMult(w).plus(&sumNorms);
        hrec->normal = sumNorms;
        hrec->normal.unitize();
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
            CHECK_ERROR(cudaMalloc((void**)&h_objects[x].surfaces[y].vertArray, sizeof(Float3D) * h_objects[x].surfaces[y].numVerts));
            CHECK_ERROR(cudaMemcpy(h_objects[x].surfaces[y].vertArray, objects[x].surfaces[y].vertArray, sizeof(Float3D) * h_objects[x].surfaces[y].numVerts, cudaMemcpyHostToDevice));
            CHECK_ERROR(cudaMalloc((void**)&h_objects[x].surfaces[y].viewNormArray, sizeof(Float3D) * h_objects[x].surfaces[y].numVerts));
            CHECK_ERROR(cudaMemcpy(h_objects[x].surfaces[y].viewNormArray, objects[x].surfaces[y].viewNormArray, sizeof(Float3D) * h_objects[x].surfaces[y].numVerts, cudaMemcpyHostToDevice));
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
        Float3D point(bitmap.firstPixel);
        point.x += (offset % bitmap.width) * bitmap.pixelWidth;
        point.y += ((offset - (threadIdx.x + blockIdx.x * blockDim.x)) / bitmap.width) * bitmap.pixelHeight;
        rays[offset].Rd = point.getUnit();
        rays[offset].Ro = Float3D(0.0, 0.0, 0.0);
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
        Float3D intersectPt;
        Float3D normal;
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
                    intersectPt = Float3D((R[threadIdx.y][threadIdx.x].Ro.x+(R[threadIdx.y][threadIdx.x].Rd.x*t)), (R[threadIdx.y][threadIdx.x].Ro.y+(R[threadIdx.y][threadIdx.x].Rd.y*t)), (R[threadIdx.y][threadIdx.x].Ro.z+(R[threadIdx.y][threadIdx.x].Rd.z*t)));
                    normal = (intersectPt.minus(&theObj->viewCenter).sDiv(theObj->boundingSphere.radius));
                    normal.unitize();
                    intersectDist = Float3D(0.0, 0.0, 0.0).distanceTo(&intersectPt);
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

__global__ void intersectTriangleKrnl(Ray *rays, int numRays, Intersect *intrs, bool *hits, Mesh *theObj, Float3D *verts, Float3D *norms, int numVerts, int mat)
{
    int offset = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;

    __shared__ Float3D V[CHUNK];
    __shared__ Float3D N[CHUNK];
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
                    intersectDist = R[threadIdx.y][threadIdx.x].Ro.distanceTo(&hrec.intersectPoint);
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
        FloatColor Ka(0.0, 0.0, 0.0, 1.0);
        FloatColor Kd(0.0, 0.0, 0.0, 1.0);
        FloatColor Ks(0.0, 0.0, 0.0, 1.0);
        FloatColor shadeColor(0.0, 0.0, 0.0, 1.0);
        FloatColor ambColor(0.0, 0.0, 0.0, 1.0);
        Float3D point = intrs[offset].point;
        Float3D trueNormal(0.0, 0.0, 0.0);
        Float3D inv_normal = intrs[offset].normal.sMult(-1.0);
        Float3D R(0.0, 0.0, 0.0);
        Float3D L(0.0, 0.0, 0.0);
        Float3D V(0.0, 0.0, 0.0);

        layer[offset*4 + 3] = (int) (theObj->materials[materialIndex].reflectivity.r * 255);

        Ka = theObj->materials[materialIndex].ka;
        Kd = theObj->materials[materialIndex].kd;
        Ks = theObj->materials[materialIndex].ks;

        ambColor.r = Ka.r * lights[0].ambient.r;
        ambColor.g = Ka.g * lights[0].ambient.g;
        ambColor.b = Ka.b * lights[0].ambient.b;

        shadeColor.plus(ambColor);
        V = Float3D(0.0, 0.0, 0.0).minus(&point);
        V.unitize();

        if (rays[offset].flags == EYE && intrs[offset].backFacing)
            trueNormal = inv_normal;
        else
            trueNormal = intrs[offset].normal;

        LightCuda *curLight;
        for (int i = 0; i < numLights; i++)
        {
            curLight = &lights[i];

            L = curLight->viewPosition.minus(&point);
            L.unitize();
            float LdotN = L.dot(&trueNormal);
            LdotN = max(0.0, LdotN);
            FloatColor diffComponent(0.0, 0.0, 0.0, 1.0);
            if (LdotN > 0.0)
                diffComponent.plus(FloatColor(curLight->diffuse.r*Kd.r*LdotN, curLight->diffuse.g*Kd.g*LdotN, curLight->diffuse.b*Kd.b*LdotN, 1.0));
            shadeColor.plus(diffComponent);

            Float3D Pr = trueNormal.sMult(LdotN);
            Float3D sub = Pr.sMult(2.0);
            R = L.sMult(-1.0).plus(&sub);
            R.unitize();
            float RdotV = R.dot(&V);
            RdotV = max(0.0, RdotV);
            float cosPhiPower = 0.0;
            if (RdotV > 0.0)
                cosPhiPower = pow(RdotV, theObj->materials[materialIndex].shiny);
            FloatColor specComponent(curLight->specular.r*Ks.r*cosPhiPower, curLight->specular.g*Ks.g*cosPhiPower, curLight->specular.b*Ks.b*cosPhiPower, 1.0);
            shadeColor.plus(specComponent);
        }

        shadeColor.r = shadeColor.r < 0.0 ? 0.0 : (shadeColor.r > 1.0 ? 1.0 : shadeColor.r);
        shadeColor.g = shadeColor.g < 0.0 ? 0.0 : (shadeColor.g > 1.0 ? 1.0 : shadeColor.g);
        shadeColor.b = shadeColor.b < 0.0 ? 0.0 : (shadeColor.b > 1.0 ? 1.0 : shadeColor.b);

        if (finalPass)
        {
            layer[offset*4 + 0] = shadeColor.r * 255;
            layer[offset*4 + 1] = shadeColor.g * 255;
            layer[offset*4 + 2] = shadeColor.b * 255;
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
