#include "raytracer.h"
#include "scene.h"

RayTracer::RayTracer(Scene *theScene)
{
    this->theScene = theScene;
    /*rayTracerShaderProg = shared_ptr<ShaderProgram>(new ShaderProgram());
    rayTracerShaderProg->vertexShader = new ShaderProgram::Shader("raytrace.vert", false, false, -1, GL_VERTEX_SHADER, rayTracerShaderProg);
    rayTracerShaderProg->vertexShader = new ShaderProgram::Shader("raytrace.frag", false, false, -1, GL_FRAGMENT_SHADER, rayTracerShaderProg);
    theScene->shUtil.setupShaders(rayTracerShaderProg);*/
}

DoubleColor RayTracer::trace(RayTracer::Ray ray, int numRecurs)
{
    double t[2] = {0, 0};
    double intersectDist = 0.0;
    double minDist = 100000000.0;
    Double3D minIntPt;
    Double3D minNormal;
    shared_ptr<PMesh> minObj = nullptr;
    int minMatIndex = 0;
    bool minBackfacing = false;
    Double3D intersectPt;
    Double3D normal;
    Double3D origin;

    for (int obj = 0; obj < (int)theScene->objects.size(); obj++)
    {
        shared_ptr<PMesh> theObj = theScene->objects.at(obj);
        if (ray.intersectSphere(theObj, t))
        {
            if (abs(t[0]) < 0.00001)
                continue;
            if (spheresOnly)
            {
                intersectPt = Double3D((ray.Ro.x+(ray.Rd.x*t[0])), (ray.Ro.y+(ray.Rd.y*t[0])), (ray.Ro.z+(ray.Rd.z*t[0])));
                normal = (intersectPt.minus(theObj->viewCenter)).sDiv(theObj->boundingSphere->radius);
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
        }
    }
    if (minObj != nullptr)
        return shade(minObj, minIntPt, minNormal, minMatIndex, minBackfacing, ray, numRecurs);
    else if (checkerBackground)
    {
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
        double u = fmod(abs(xFar), 100);
        double v = fmod(abs(yFar), 100);
        if ((xFar > 0.0 && yFar < 0.0) || (xFar < 0.0 && yFar > 0.0))
        {
            if ((u >= 100/2 && v < 100/2) || (u < 100/2 && v>= 100/2))
                c = 0;
            else c = 1;
        }
        else
        {
            if ((u >= 100/2 && v < 100/2) || (u < 100/2 && v>= 100/2))
                c = 1;
            else
                c = 0;
        }
        return DoubleColor(c, c, c, 1.0);
    }
    else
        return DoubleColor(0.0, 0.0, 0.0, 1.0);
}

DoubleColor RayTracer::shade(shared_ptr<PMesh> theObj, Double3D point, Double3D normal, int materialIndex, bool backFacing, RayTracer::Ray ray, int numRecurs)
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
        reflections = this->reflections;
    refractions = this->refractions;
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
        lightPos = Double3D(curLight->position[0], curLight->position[1], curLight->position[2]);
        lightViewPos = lightPos.preMultiplyMatrix(theScene->camera->viewMat);
        if(shadows)
        {
            Double3D Rd(lightViewPos.minus(point));
            Rd.unitize();
            Ray shadowRay = Ray(Double3D(Rd), Double3D(point));
            if (traceLightRay(shadowRay, theObj))
            {
                fprintf(stdout, " light#%d\n", i);
                obstructed = true;
            }
        }
        if (obstructed)
            continue;

        L = lightViewPos.minus(point);
        L.unitize();
        double LdotN = L.dot(trueNormal);
        DoubleColor diffComponent(0.0, 0.0, 0.0, 1.0);
        if (LdotN > 0.0)
            diffComponent.plus(DoubleColor(curLight->diffuse[0]*Kd.r*LdotN, curLight->diffuse[1]*Kd.g*LdotN, curLight->diffuse[2]*Kd.b*LdotN, 1.0));
        shadeColor.plus(diffComponent);

        Double3D Pr = trueNormal.sMult(LdotN);
        Double3D sub = Pr.sMult(2.0);
        R = L.sMult(-1.0).plus(sub);
        R.unitize();
        double RdotV = R.dot(V);
        if (RdotV > 1.0)
            fprintf(stdout, "RdotV: %f\n", RdotV);
        double cosPhiPower = 0.0;
        if (RdotV > 0.0)
            cosPhiPower = pow(RdotV, theObj->materials[materialIndex].shiny);
        DoubleColor specComponent(curLight->specular[0]*Ks.r*cosPhiPower, curLight->specular[1]*Ks.g*cosPhiPower, curLight->specular[2]*Ks.b*cosPhiPower, 1.0);
        shadeColor.plus(specComponent);
    }
    if (numRecurs >= maxRecursiveDepth)
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
        fprintf(stdout, "Leaving shade recursive depth: %d\n", numRecurs);
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

void RayTracer::render()
{
    height = heightOfPixel();
    width = widthOfPixel();

    calcBounds();
    doViewTrans();
    GLfloat *data = castRays();
    GLuint tex;
    glGenTextures(1, &tex);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, theScene->camera->getViewportWidth(), theScene->camera->getViewportHeight(), 0, GL_RGB, GL_FLOAT, data);

    glUseProgram(rayTracerShaderProg->progID);

    free(data);
}

double RayTracer::heightOfPixel()
{
    return theScene->camera->getWindowHeight()/theScene->camera->getViewportHeight();
}

double RayTracer::widthOfPixel()
{
    return theScene->camera->getWindowWidth()/theScene->camera->getViewportWidth();
}

bool RayTracer::traceLightRay(RayTracer::Ray ray, shared_ptr<PMesh> fromObj)
{
    double t[2]={0.0, 0.0};
    for (int obj = 0; obj < (int)theScene->objects.size(); obj++)
    {
        shared_ptr<PMesh> theObj = theScene->objects.at(obj);
        if (ray.intersectSphere(theObj, t))
        {
            if (abs(t[0]) < 0.0001)
                return false;
            else
                return true;
        }
    }
    return false;
}

RayTracer::Ray::Ray(Double3D dir, Double3D origin)
{
    Rd = dir;
    Ro = origin;
    flags = 0;
}

RayTracer::Ray::Ray(Double3D dir, Double3D origin, int type)
{
    Rd = dir;
    Ro = origin;
    flags = type;
}

bool RayTracer::Ray::intersectSphere(shared_ptr<PMesh> theObj, double *t)
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
                t[0] = 0.0;
                return false;
            }
            else
            {
                t[0] = t1;
                return true;
            }
        }
        else if (t1 < EPS)
        {
            t[0] = t0;
            return true;
        }
        else if (t0 < t1)
        {
            t[0] = t0;
            return true;
        }
        else
        {
            t[0] = t1;
            return true;
        }
    }
}

void RayTracer::calcBounds()
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

void RayTracer::doViewTrans()
{
    vector<double> modelViewInvTranspose;
    for (int obj = 0; obj < (int)theScene->objects.size(); obj++)
    {
        shared_ptr<PMesh> thisObj = theScene->objects.at(obj);
        vector<double> modelView = MatrixOps::newIdentity();
        modelView = MatrixOps::multMat(modelView, thisObj->modelMat);
        modelView = MatrixOps::multMat(theScene->camera->viewMat, modelView);
        modelViewInvTranspose = MatrixOps::inverseTranspose(modelView);
        thisObj->viewCenter = thisObj->center.preMultiplyMatrix(theScene->camera->viewMat);
    }
}

GLfloat *RayTracer::castRays()
{
    GLfloat *data = (GLfloat*)malloc(sizeof(GLfloat)*(theScene->camera->viewportTop * theScene->camera->viewportWidth)*3);
    firstPixel.x = theScene->camera->windowLeft+width/2;
    firstPixel.y = theScene->camera->windowBottom+height/2;
    firstPixel.z = -theScene->camera->near;

    Double3D point(firstPixel);
    Double3D origin;
    DoubleColor rgb;
    Ray ray = Ray(Double3D(0.0, 0.0, -1.0), origin, EYE);
    int xCount = 0, yCount = 0;

    for (double y = 0; y < theScene->camera->viewportTop; y++)
    {
        for (double x = 0; x < theScene->camera->viewportWidth; x++)
        {
            ray = Ray(point.getUnit(), origin, EYE);
            rgb = trace(ray, 0);
            data[((yCount*(int)theScene->camera->viewportWidth+xCount)*3)] = rgb.r;
            data[((yCount*(int)theScene->camera->viewportWidth+xCount)*3)+1] = rgb.g;
            data[((yCount*(int)theScene->camera->viewportWidth+xCount)*3)+2] = rgb.b;
            point.x += width;
        }
        point.x = firstPixel.x;
        point.y += height;
    }
    return data;
}
