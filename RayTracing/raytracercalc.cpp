#include "raytracercalc.h"
#include "scene.h"
#include "RayTracing/raytracer.h"

RayTracerCalc::RayTracerCalc(Scene *theScene, RayTracer *rayTracer)
{
    QObject::moveToThread(this);
    this->theScene = theScene;
    this->rayTracer = rayTracer;
}

RayTracerCalc::~RayTracerCalc()
{
    theScene = nullptr;
    rayTracer = nullptr;
}

DoubleColor RayTracerCalc::trace(RayTracerCalc::Ray ray, int numRecurs)
{
    double t = 0.0;
    double intersectDist = 0.0;
    double minDist = 100000000.0;
    Double3D minIntPt;
    Double3D minNormal;
    PMesh *minObj = nullptr;
    int minMatIndex = 0;
    bool minBackfacing = false;
    Double3D intersectPt;
    Double3D normal;
    Double3D origin;

    for (int obj = 0; obj < (int)theScene->objects.size(); obj++)
    {
        PMesh *theObj = theScene->objects.at(obj).get();
        if (ray.intersectSphere(theObj, &t))
        {
            if (abs(t) < 0.00001)
                continue;
            if (rayTracer->spheresOnly)
            {
                intersectPt = Double3D((ray.Ro.x+(ray.Rd.x*t)), (ray.Ro.y+(ray.Rd.y*t)), (ray.Ro.z+(ray.Rd.z*t)));
                normal = (intersectPt.minus(theObj->viewCenter).sDiv(theObj->boundingSphere->radius));
                normal.unitize();
                intersectDist = origin.distanceTo(intersectPt);
                if (intersectDist < minDist)
                {
                    minDist = intersectDist;
                    minObj = theObj;
                    minIntPt = Double3D(intersectPt);
                    minNormal = Double3D(normal);
                }
            }
            else
            {
                PMesh::SurfCell *surf = theObj->surfHead.get();
                while (surf != nullptr)
                {
                    int i = 0;
                    for (PMesh::PolyCell *poly = surf->polyHead.get(); i < surf->numPoly; i++, poly = poly->next.get())
                    {
                        HitRecord hrec;
                        if (ray.intersectTriangle(theObj, poly, &hrec, false))
                        {
                            if (!(ray.flags == EYE && hrec.backfacing && theScene->cull) || ray.flags == REFLECT || ray.flags == EXTERNAL_REFRACT)
                            {
                                intersectDist = ray.Ro.distanceTo(hrec.intersectPoint);
                                if (intersectDist < minDist)
                                {
                                    minDist = intersectDist;
                                    minObj = theObj;
                                    minIntPt = hrec.intersectPoint;
                                    minNormal = hrec.normal;
                                    minMatIndex = surf->material;
                                    minBackfacing = hrec.backfacing;
                                }
                            }
                        }
                    }
                    surf = surf->next.get();
                }
            }
        }
    }
    if (minObj != nullptr)
        return shade(minObj, minIntPt, minNormal, minMatIndex, minBackfacing, ray, numRecurs);
    else if (rayTracer->checkerBackground)
    {
        double checkerSize = rayTracer->checkerSize;
        double tFar, xFar, yFar;
        Double3D farPlaneNormal(0.0, 0.0, 1.0);
        double Vo = -1.0 * (ray.Rd.dot(farPlaneNormal) + theScene->camera->far);
        double Vd = ray.Rd.dot(farPlaneNormal);
        tFar = Vo/Vd;
        if (tFar < 0.0)
            return DoubleColor(0.0, 0.0, 0.0, 1.0);
        xFar = ray.Ro.x + (ray.Rd.x * tFar);
        yFar = ray.Ro.y + (ray.Rd.y * tFar);
        if (xFar < farLeft || xFar > farRight || yFar < farBottom || yFar > farTop)
            return DoubleColor(0.0, 0.0, 0.0, 1.0);

        double c = 0.0;
        double u = fmod(abs(xFar), checkerSize);
        double v = fmod(abs(yFar), checkerSize);
        if ((xFar > 0.0 && yFar < 0.0) || (xFar < 0.0 && yFar > 0.0))
        {
            if ((u >= checkerSize/2 && v < checkerSize/2) || (u < checkerSize/2 && v>= checkerSize/2))
                c = 0;
            else c = 1;
        }
        else
        {
            if ((u >= checkerSize/2 && v < checkerSize/2) || (u < checkerSize/2 && v>= checkerSize/2))
                c = 1;
            else
                c = 0;
        }
        return DoubleColor(c, c, c, 1.0);
    }
    else
        return DoubleColor(0.0, 0.0, 0.0, 1.0);
}

DoubleColor RayTracerCalc::shade(PMesh *theObj, Double3D point, Double3D normal, int materialIndex, bool backFacing, RayTracerCalc::Ray ray, int numRecurs)
{
    DoubleColor ambColor;
    DoubleColor shadeColor;
    DoubleColor reflColor;
    DoubleColor refrColor;
    Double3D inv_normal = normal.sMult(-1.0);
    Double3D trueNormal;
    bool reflections;
    bool refractions;
    double reflectivity;
    double refractivity;

    if (ray.flags == INTERNAL_REFRACT)
        reflections = false;
    else
        reflections = rayTracer->reflections;
    refractions = rayTracer->refractions;
    reflectivity = theObj->materials[materialIndex].reflectivity.r;
    refractivity = theObj->materials[materialIndex].refractivity.r;

    Ka = theObj->materials[materialIndex].ka;
    Kd = theObj->materials[materialIndex].kd;
    Ks = theObj->materials[materialIndex].ks;

    Double3D lightPos;
    Double3D lightViewPos;
    Double3D R;
    Double3D L;
    Double3D V;
    ambColor.r = Ka.r * theScene->lights->lights[0].ambient[0];
    ambColor.g = Ka.g * theScene->lights->lights[0].ambient[1];
    ambColor.b = Ka.b * theScene->lights->lights[0].ambient[2];
    shadeColor.plus(ambColor);

    V = Double3D(0.0, 0.0, 0.0).minus(point);
    V.unitize();

    if (ray.flags == EYE && backFacing && !theScene->cull)
        trueNormal = inv_normal;
    else if (ray.flags == INTERNAL_REFRACT && backFacing)
        trueNormal = inv_normal;
    else
        trueNormal = normal;

    Lights::Light *curLight;
    for (int i = 0; i < 8; i++)
    {
        bool obstructed = false;
        curLight = &theScene->lights->lights[i];
        if (curLight->lightSwitch == 0)
            continue;
        lightPos = Double3D(curLight->position[0], curLight->position[1], curLight->position[2]);
        lightViewPos = lightPos.preMultiplyMatrix(theScene->camera->viewMat);
        if(rayTracer->shadows)
        {
            Double3D Rd(lightViewPos.minus(point));
            Rd.unitize();
            Ray shadowRay = Ray(Double3D(Rd), Double3D(point));
            if (traceLightRay(shadowRay, theObj))
                obstructed = true;
        }
        if (obstructed)
            continue;

        L = lightViewPos.minus(point);
        L.unitize();
        double LdotN = L.dot(trueNormal);
        LdotN = max(0.0, LdotN);
        DoubleColor diffComponent(0.0, 0.0, 0.0, 1.0);
        if (LdotN > 0.0)
            diffComponent.plus(DoubleColor(curLight->diffuse[0]*Kd.r*LdotN, curLight->diffuse[1]*Kd.g*LdotN, curLight->diffuse[2]*Kd.b*LdotN, 1.0));
        shadeColor.plus(diffComponent);

        Double3D Pr = trueNormal.sMult(LdotN);
        Double3D sub = Pr.sMult(2.0);
        R = L.sMult(-1.0).plus(sub);
        R.unitize();
        double RdotV = R.dot(V);
        RdotV = max(0.0, RdotV);
        if (RdotV > 1.0)
            fprintf(stdout, "RdotV: %f\n", RdotV);
        double cosPhiPower = 0.0;
        if (RdotV > 0.0)
            cosPhiPower = pow(RdotV, theObj->materials[materialIndex].shiny);
        DoubleColor specComponent(curLight->specular[0]*Ks.r*cosPhiPower, curLight->specular[1]*Ks.g*cosPhiPower, curLight->specular[2]*Ks.b*cosPhiPower, 1.0);
        shadeColor.plus(specComponent);
    }
    if (numRecurs >= rayTracer->maxRecursiveDepth)
        return shadeColor;

    if (refractions)
    {
        double rhoNew, rhoOld;
        Double3D norm;
        if (ray.flags == INTERNAL_REFRACT)
        {
            rhoOld = theObj->materials[theObj->objNumber].refractiveIndex;
            rhoNew = rhoAIR;
            norm = Double3D(inv_normal);
        }
        else
        {
            rhoNew = theObj->materials[theObj->objNumber].refractiveIndex;
            rhoOld = rhoAIR;
            norm = Double3D(normal);
        }
        double rhoOldSq = rhoOld * rhoOld;
        double rhoNewSq = rhoNew * rhoNew;
        Double3D d = ray.Rd;
        double dDotn = d.dot(norm);
        Double3D term1 = d.minus(norm.sMult(dDotn)).sMult(rhoOld);
        term1 = term1.sDiv(rhoNew);
        double sqrtOp = 1.0 - ((rhoOldSq*(1.0 - dDotn * dDotn))/rhoNewSq);
        if (sqrtOp < 0.0)
        {
            reflectivity = reflectivity + refractivity;
            reflections = true;
            refractions = false;
        }
        else
        {
            double root = sqrt(sqrtOp);
            Double3D term2 = norm.sMult(root);
            Double3D t = term1.minus(term2);
            t.unitize();
            Ray newRay = Ray(Double3D(), Double3D());
            if (ray.flags == INTERNAL_REFRACT)
                newRay = Ray(t, point, EXTERNAL_REFRACT);
            else
                newRay = Ray(t, point, INTERNAL_REFRACT);
            refrColor = trace(newRay, numRecurs+1);
        }
    }

    if (reflections)
    {
        Double3D Pr = trueNormal.sMult(ray.Rd.dot(trueNormal));
        Double3D sub = Pr.sMult(2.0);
        Double3D refVect = ray.Rd.minus(sub);
        refVect.unitize();

        Ray reflRay = Ray(refVect, point, REFLECT);
        reflColor = trace(reflRay, numRecurs+1);
    }

    DoubleColor rtnColor;

    if (reflections && !refractions)
    {
        shadeWeight = 1.0 - reflectivity;
        reflColor.scale(reflectivity);
        shadeColor.scale(shadeWeight);
        rtnColor.plus(shadeColor);
        rtnColor.plus(reflColor);
        return rtnColor;
    }
    else if (reflections && refractions)
    {
        shadeWeight = 1.0 - (reflectivity + refractivity);
        reflColor.scale(refractivity);
        reflColor.scale(reflectivity);
        shadeColor.scale(shadeWeight);
        rtnColor.plus(refrColor);
        rtnColor.plus(shadeColor);
        rtnColor.plus(reflColor);
        return rtnColor;
    }
    else if (!reflections && refractions)
    {
        shadeWeight = 1.0 - refractivity;
        reflColor.scale(refractivity);
        shadeColor.scale(shadeWeight);
        rtnColor.plus(refrColor);
        rtnColor.plus(shadeColor);
        return rtnColor;
    }
    else
        return shadeColor;
}

void RayTracerCalc::calcBounds()
{
    Double3D farPlaneNormal(0.0, 0.0, 1.0);

    Double3D raydir(0.0, theScene->camera->windowTop, -theScene->camera->near);
    Ray boundaryRay(raydir, Double3D());
    double Vo = -1.0 * (boundaryRay.Rd.dot(farPlaneNormal) + theScene->camera->far);
    double Vd = boundaryRay.Rd.dot(farPlaneNormal);
    double tFar = Vo/Vd;
    farTop = boundaryRay.Ro.y + (boundaryRay.Rd.y * tFar);

    raydir = Double3D(0.0, theScene->camera->windowBottom, -theScene->camera->near);
    boundaryRay = Ray(raydir, Double3D());
    Vo = -1.0 * (boundaryRay.Rd.dot(farPlaneNormal) + theScene->camera->far);
    Vd = boundaryRay.Rd.dot(farPlaneNormal);
    tFar = Vo/Vd;
    farBottom = boundaryRay.Ro.y + (boundaryRay.Rd.y * tFar);

    raydir = Double3D(theScene->camera->windowLeft, 0.0, -theScene->camera->near);
    boundaryRay = Ray(raydir, Double3D());
    Vo = -1.0 * (boundaryRay.Rd.dot(farPlaneNormal) + theScene->camera->far);
    Vd = boundaryRay.Rd.dot(farPlaneNormal);
    tFar = Vo/Vd;
    farLeft = boundaryRay.Ro.x + (boundaryRay.Rd.x * tFar);

    raydir = Double3D(theScene->camera->windowRight, 0.0, -theScene->camera->near);
    boundaryRay = Ray(raydir, Double3D());
    Vo = -1.0 * (boundaryRay.Rd.dot(farPlaneNormal) + theScene->camera->far);
    Vd = boundaryRay.Rd.dot(farPlaneNormal);
    tFar = Vo/Vd;
    farRight = boundaryRay.Ro.x + (boundaryRay.Rd.x * tFar);
}

bool RayTracerCalc::traceLightRay(RayTracerCalc::Ray ray, PMesh *fromObj)
{
    double t = 0.0;
    for (int obj = 0; obj < (int)theScene->objects.size(); obj++)
    {
        PMesh *theObj = theScene->objects.at(obj).get();
        if (ray.intersectSphere(theObj, &t))
        {
            if (abs(t) < 0.0001)
                return false;
            else
                return true;
        }
    }
    return false;
}

void RayTracerCalc::doViewTrans()
{
    vector<double> modelViewInvTranspose;
    for (int obj = 0; obj < (int)theScene->objects.size(); obj++)
    {
        PMesh *thisObj = theScene->objects.at(obj).get();
        vector<double> modelView = MatrixOps::newIdentity();
        modelView = MatrixOps::multMat(modelView, thisObj->modelMat);
        modelView = MatrixOps::multMat(theScene->camera->viewMat, modelView);
        modelViewInvTranspose = MatrixOps::inverseTranspose(modelView);
        Double3D transNorm;
        for (int vert = 0; vert < thisObj->numVerts; vert++)
        {
            thisObj->vertArray.at(vert)->viewPos = thisObj->vertArray.at(vert)->worldPos.preMultiplyMatrix(modelView);
            transNorm = thisObj->vertNormArray.at(vert)->preMultiplyMatrix(modelViewInvTranspose);
            thisObj->viewNormArray.insert(thisObj->viewNormArray.begin()+vert, transNorm);
        }
        thisObj->viewCenter = thisObj->center.preMultiplyMatrix(theScene->camera->viewMat);
        thisObj->calcViewPolyNorms();
    }
}

void RayTracerCalc::run()
{
    height = heightOfPixel();
    width = widthOfPixel();

    calcBounds();
    doViewTrans();
    if (rayTracer->data != nullptr)
    {
        free(rayTracer->data);
        rayTracer->data = nullptr;
    }
    rayTracer->data = castRays();

    this->quit();
}

double RayTracerCalc::heightOfPixel()
{
    return theScene->camera->getWindowHeight()/theScene->camera->getViewportHeight();
}

double RayTracerCalc::widthOfPixel()
{
    return theScene->camera->getWindowWidth()/theScene->camera->getViewportWidth();
}

RayTracerCalc::Ray::Ray(Double3D dir, Double3D origin)
{
    Rd = dir;
    Ro = origin;
    flags = 0;
}

RayTracerCalc::Ray::Ray(Double3D dir, Double3D origin, int type)
{
    Rd = dir;
    Ro = origin;
    flags = type;
}

bool RayTracerCalc::Ray::intersectSphere(PMesh *theObj, double *t)
{
    const double EPS = 0.00001;
    double t0=0.0, t1=0.0, A=0.0, B=0.0, C=0.0, discrim=0.0;
    Sphere *theSphere = theObj->boundingSphere;
    Double3D RoMinusSc = Ro.minus(theObj->viewCenter);
    double fourAC = 0.0;

    A = Rd.dot(Rd);
    B = 2.0 * (Rd.dot(RoMinusSc));
    C = RoMinusSc.dot(RoMinusSc) - theSphere->radiusSq;
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

bool RayTracerCalc::Ray::intersectTriangle(PMesh *theObj, PMesh::PolyCell *thePoly, RayTracerCalc::HitRecord *hrec, bool cull)
{
    if (thePoly->numVerts != 3)
    {
        fprintf(stderr, "PolyCell.intersectTriangle: PolyCell is not a triangle.\n");
        return false;
    }
    Double3D verts[3];
    Double3D edges[2];
    Double3D vnorms[3];
    Double3D pvec, qvec, tvec;
    double det, inv_det;
    double EPSILON = 0.000001;
    PMesh::VertListCell *curV;
    int vindex = 0;

    for (curV = thePoly->vert.get(); curV != nullptr; curV = curV->next.get(), vindex++)
    {
        verts[vindex] = theObj->vertArray.at(curV->vert)->viewPos;
        vnorms[vindex] = theObj->viewNormArray.at(curV->vert);
    }
    edges[0] = verts[1].minus(verts[0]);
    edges[1] = verts[2].minus(verts[0]);
    pvec = Rd.cross(edges[1]);
    det = edges[0].dot(pvec);
    if(cull)
    {
        if (det < EPSILON)
            return false;
        tvec = Ro.minus(verts[0]);
        hrec->u = tvec.dot(pvec);
        if (hrec->u < 0.0 || hrec->u > det)
            return false;
        qvec = tvec.cross(edges[0]);
        hrec->v = Rd.dot(qvec);
        if (hrec->v < 0.0 || hrec->u + hrec->v > det)
            return false;
        hrec->t = edges[1].dot(qvec);
        inv_det = 1.0/det;
        hrec->t *= inv_det;
        hrec->u *= inv_det;
        hrec->v *= inv_det;
    }
    else
    {
        if (det > -EPSILON && det < EPSILON)
            return false;
        inv_det = 1.0/det;
        tvec = Ro.minus(verts[0]);
        hrec->u = tvec.dot(pvec) * inv_det;
        if (hrec->u < 0.0 || hrec->u > 1.0)
            return false;
        qvec = tvec.cross(edges[0]);
        hrec->v = Rd.dot(qvec) * inv_det;
        if (hrec->v < 0.0 || hrec->u + hrec->v > 1.0)
            return false;
        if (det < -EPSILON)
            hrec->backfacing = true;
        else
            hrec->backfacing = false;
        hrec->t = edges[1].dot(qvec) * inv_det;
    }
    if (hrec->t < EPSILON)
        return false;
    else
    {
        hrec->intersectPoint = Double3D((Ro.x + (Rd.x * hrec->t)), (Ro.y + (Rd.y * hrec->t)), (Ro.z + (Rd.z * hrec->t)));
        double w = 1.0 - hrec->u - hrec->v;
        Double3D sumNorms;
        vnorms[0] = vnorms[0].sMult(w);
        vnorms[1] = vnorms[1].sMult(hrec->u);
        vnorms[2] = vnorms[2].sMult(hrec->v);
        sumNorms = vnorms[0].plus(vnorms[1].plus(vnorms[2]));
        hrec->normal = sumNorms;
        hrec->normal.unitize();
        hrec->thisPoly = thePoly;
        return true;
    }
}

unsigned char *RayTracerCalc::castRays()
{
    unsigned char *data = (unsigned char*)malloc(sizeof(unsigned char)*(((int)theScene->camera->viewportTop) * ((int)theScene->camera->getViewportWidth()))*3);
    firstPixel.x = theScene->camera->windowLeft+width/2;
    firstPixel.y = theScene->camera->windowBottom+height/2;
    firstPixel.z = -theScene->camera->near;

    Double3D point(firstPixel);
    Double3D origin;
    DoubleColor rgb;
    Ray ray = Ray(Double3D(0.0, 0.0, -1.0), origin, EYE);

    cancelRayTrace = false;
    int i = 0;
    for (int y = 0; y < (int)theScene->camera->viewportTop; y++)
    {
        if (cancelRayTrace)
            break;
        emit percentageComplete(((y*theScene->camera->getViewportWidth())/(theScene->camera->viewportTop*theScene->camera->getViewportWidth()))*100.0);
        for (int x = 0; x < (int)theScene->camera->getViewportWidth(); x++)
        {
            ray = Ray(point.getUnit(), origin, EYE);
            rgb = trace(ray, 0);
            rgb.r = rgb.r < 0.0 ? 0.0 : (rgb.r > 1.0 ? 1.0 : rgb.r);
            rgb.g = rgb.g < 0.0 ? 0.0 : (rgb.g > 1.0 ? 1.0 : rgb.g);
            rgb.b = rgb.b < 0.0 ? 0.0 : (rgb.b > 1.0 ? 1.0 : rgb.b);
            data[i] = rgb.r*255;
            data[i+1] = rgb.g*255;
            data[i+2] = rgb.b*255;
            i+=3;
            point.x += width;
        }
        point.x = firstPixel.x;
        point.y += height;
    }
    emit percentageComplete(100);
    return data;
}


RayTracerCalc::HitRecord::HitRecord()
{
    t = 0.0;
    u = 0.0;
    v = 0.0;
    backfacing = false;
}

RayTracerCalc::HitRecord::HitRecord(double newT, double newU, double newV, Double3D newIntersect, Double3D newNormal, bool newBackFacing)
{
    t = newT;
    u = newU;
    v = newV;
    intersectPoint = new Double3D(newIntersect);
    normal = new Double3D(newNormal);
    backfacing = newBackFacing;
}
