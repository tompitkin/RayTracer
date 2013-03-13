#include "raytracer.h"
#include "scene.h"

RayTracer::RayTracer(Scene *theScene)
{
    this->theScene = theScene;
    rayTracerShaderProg = shared_ptr<ShaderProgram>(new ShaderProgram());
    rayTracerShaderProg->vertexShader = new ShaderProgram::Shader("raytrace.vert", false, false, -1, GL_VERTEX_SHADER, rayTracerShaderProg);
    rayTracerShaderProg->fragmentShader = new ShaderProgram::Shader("raytrace.frag", false, false, -1, GL_FRAGMENT_SHADER, rayTracerShaderProg);
    theScene->shUtil.setupShaders(rayTracerShaderProg);
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

void RayTracer::writeBMP(const char *fname, int w,int h,unsigned char *img)
{
    FILE *f = fopen(fname,"wb");

    unsigned char bfh[54] = {0x42, 0x4d,
    /* bfSize [2]*/ 54, 0, 0, 0, /**/
    /* reserved [6]*/ 0, 0, 0, 0, /**/
    /* biOffBits [10]*/ 54, 0, 0, 0, /**/
    /* biSize [14]*/ 40, 0, 0, 0, /**/
    /* width [18]*/ 0, 0, 0, 0, /**/
    /* height [22]*/ 0, 0, 0, 0, /**/
    /* planes [26]*/ 1, 0, /**/
    /* bitcount [28]*/ 24, 0,/**/
    /* compression [30]*/ 0, 0, 0, 0, /**/
    /* size image [34]*/ 0, 0, 0, 0, /**/
    /* xpermeter [38]*/ 0, 0, 0, 0, /**/
    /* ypermeter [42]*/ 0, 0, 0, 0, /**/
    /* clrused [46]*/ 0, 0, 0, 0, /**/
    /* clrimportant [50]*/ 0, 0, 0, 0 /**/};
    int realw = w * 3, rem = w % 4, isz = (realw + rem) * h, fsz = isz + 54;
    //bfh.bfSize = fsz;
    bfh[2] = (fsz & 0xFF); bfh[3] = (fsz >> 8) & 0xFF; bfh[4] = (fsz >> 16) & 0xFF; bfh[5] = (fsz >> 24) & 0xFF;
    //bfh.biSize = isz
    bfh[34] = (isz & 0xFF); bfh[35] = (isz >> 8) & 0xFF; bfh[36] = (isz >> 16) & 0xFF; bfh[37] = (isz >> 24) & 0xFF;
    //bfh.biWidth = w;
    bfh[18] = (w & 0xFF); bfh[19] = (w >> 8) & 0xFF; bfh[20] = (w >> 16) & 0xFF; bfh[21] = (w >> 24) & 0xFF;
    //bfh.biHeight = h;
    bfh[22] = (h & 0xFF); bfh[23] = (h >> 8) & 0xFF; bfh[24] = (h >> 16) & 0xFF; bfh[25] = (h >> 24) & 0xFF;

    // xpels/ypels
    // bfh[38] = 19; bfh[39] = 11;
    // bfh[42] = 19; bfh[43] = 11;

    fwrite((void*)bfh, 54, 1, f);

    unsigned char* bstr = new unsigned char[realw], *remstr = 0;
    if(rem != 0) { remstr = new unsigned char[rem]; memset(remstr,0,rem); }

    for(int j = h-1 ; j > -1 ; j--){
            for(int i = 0 ; i < w ; i++)
                    for(int k = 0 ; k < 3 ; k++) { bstr[i*3+k] = img[(j*realw+i*3)+(2-k)]; }
            fwrite(bstr,realw,1,f); if (rem != 0) { fwrite(remstr,rem,1,f); }
    }

    delete [] bstr; if(remstr) delete [] remstr;

    fclose(f);
}

void RayTracer::render()
{
    height = heightOfPixel();
    width = widthOfPixel();

    calcBounds();
    doViewTrans();
    GLbyte *data = castRays();
    /*unsigned char *data = new unsigned char[600 * 600 * 3];
    for (int x = 0; x < 600*600*3; x++)
        data[x] = rand()%100+1;*/
    /*GLbyte *data = new GLbyte[1920 * 1080 * 3];
    for (int x = 0; x < 1920*1080*3; x++)
        data[x] = rand()%256;*/
    writeBMP("/home/tom/Desktop/test.bmp", (int)theScene->camera->viewportTop, (int)theScene->camera->viewportWidth, (unsigned char*)data);

    static const GLfloat position[] = {
        -1.0f, -1.0f,
         1.0f, -1.0f,
        -1.0f,  1.0f,
         1.0f,  1.0f
    };

    glUseProgram(rayTracerShaderProg->progID);
    glGenBuffers(1, &buffer);
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(position), position, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(GLfloat)*2, 0);
    glActiveTexture(GL_TEXTURE0);
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, (int)theScene->camera->viewportTop, (int)theScene->camera->viewportWidth, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
    GLuint texLoc = glGetUniformLocation(rayTracerShaderProg->progID, "tex");
    glUniform1i(texLoc, 0);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    delete []data;
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

GLbyte *RayTracer::castRays()
{
    GLbyte *data = (GLbyte*)malloc(sizeof(GLbyte)*(((int)theScene->camera->viewportTop) * ((int)theScene->camera->viewportWidth))*3);
    firstPixel.x = theScene->camera->windowLeft+width/2;
    firstPixel.y = theScene->camera->windowBottom+height/2;
    firstPixel.z = -theScene->camera->near;

    Double3D point(firstPixel);
    Double3D origin;
    DoubleColor rgb;
    Ray ray = Ray(Double3D(0.0, 0.0, -1.0), origin, EYE);

    int i = 0;
    for (int y = 0; y < (int)theScene->camera->viewportTop; y++)
    {
        for (int x = 0; x < (int)theScene->camera->viewportWidth; x++)
        {
            ray = Ray(point.getUnit(), origin, EYE);
            rgb = trace(ray, 0);
            data[i] = rgb.r*255;
            data[i+1] = rgb.g*255;
            data[i+2] = rgb.b*255;
            i+=3;
            point.x += width;
        }
        point.x = firstPixel.x;
        point.y += height;
    }
    return data;
}
