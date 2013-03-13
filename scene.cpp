#include "scene.h"

Scene::Scene(QWidget *parent) :
    QGLWidget(parent)
{
    //This is a work around for a bug in QT 5.0 that won't allow you
    //to use any OpenGL version >= 3.0
    QGLFormat format;
    format.setVersion(4, 3);
    QGLFormat::setDefaultFormat(format);
    this->setFormat(format);

    makeCurrent();
    GLenum err = glewInit();
    if (err != GLEW_OK)
    {
        fprintf(stderr, "GLEW Error: %s\n", glewGetErrorString(err));
    }
    fprintf(stdout, "Driver Version String: %s\n", glGetString(GL_VERSION));
    if (glewIsSupported("GL_VERSION_3_0"))
        fprintf(stdout, "GL3 is available\n");
    else
    {
        fprintf(stdout, "GL3 is NOT available\n");
    }

    camera = new Camera();
    curObject = nullptr;
    lights = new Lights(this);
    updateLight = true;
    updateLights = vector<bool>(8);
    for (int i = 0; i < (int)updateLights.size(); i++)
    {
        lights->updateLight(i);
        updateLights[i] = false;
    }
    updateLight = false;

    shaderProg = shared_ptr<ShaderProgram>(new ShaderProgram());
    if (vertShaderName.empty())
        shaderProg->vertexShader = new ShaderProgram::Shader(defaultVertShader,false, false, -1, GL_VERTEX_SHADER, shaderProg);
    else
        shaderProg->vertexShader = new ShaderProgram::Shader(vertShaderName, false, false, -1, GL_VERTEX_SHADER, shaderProg);
    if (fragShaderName.empty())
        shaderProg->fragmentShader = new ShaderProgram::Shader(defaultFragShader, false, false, -1, GL_FRAGMENT_SHADER, shaderProg);
    else
        shaderProg->fragmentShader = new ShaderProgram::Shader(fragShaderName, false, false, -1, GL_FRAGMENT_SHADER, shaderProg);

    setupUniforms(shaderProg);

    shaders.push_back(shaderProg);
    drawAxis = true;
    cull = false;
}

Scene::~Scene()
{
    shaderProg.reset();
    curObject.reset();
    delete camera;
    camera = nullptr;
    delete lights;
    lights = nullptr;
    delete axes;
    axes = nullptr;
}

void Scene::setupUniforms(shared_ptr<ShaderProgram> theShader)
{
    theShader->addUniform(camera->viewMatUniform);
    theShader->addUniform(camera->invCamUniform);
    theShader->addUniform(camera->projMatUniform);
    for(int i =0; i < 8;i++)
    {
        theShader->addUniform(lights->lightSwitch[i]);
        theShader->addUniform(lights->lightPosition[i]);
        theShader->addUniform(lights->lightDiff[i]);
        theShader->addUniform(lights->lightAmb[i]);
        theShader->addUniform(lights->lightSpec[i]);
        theShader->addUniform(lights->spotCutoff[i]);
        theShader->addUniform(lights->spotDirection[i]);
        theShader->addUniform(lights->spotExponent[i]);
    }

    camera->viewMatUniform->needsUpdate = true;
    camera->invCamUniform->off = false;
    camera->invCamUniform->needsUpdate = true;
    camera->projMatUniform->needsUpdate = true;
    for(int i =0; i < 8;i++)
    {
        lights->lightSwitch[i]->needsUpdate=true;
        lights->lightDiff[i]->needsUpdate = true;
        lights->lightPosition[i]->needsUpdate = true;

        lights->lightAmb[i]->off = true;
        lights->lightAmb[i]->needsUpdate = false;
        lights->lightSpec[i]->off = true;
        lights->lightSpec[i]->needsUpdate = false;
    }
}

void Scene::adjustWindowAspect()
{
    double viewportAspect = camera->aspectRatio;
    double winWidth = camera->getWindowWidth();
    double winHeight = camera->getWindowHeight();
    double winLeft = camera->windowLeft;
    double winRight = camera->windowRight;
    double winTop = camera->windowTop;
    double winBottom = camera->windowBottom;
    double newWinWidth = viewportAspect * winHeight;
    double widthChange = newWinWidth - winWidth;
    widthChange /= 2.0;
    winLeft -= widthChange;
    winRight += widthChange;
    camera->setFrustum(winLeft, winRight, winBottom, winTop, camera->near, camera->far);
    camera->frustumChanged = true;
}

void Scene::addObject(QString fileName, int fileType)
{
    shared_ptr<PMesh> newObj;
    switch(fileType)
    {
    case ObjectTypes::TYPE_OBJ:
        newObj = shared_ptr<ObjLoaderBuffer>(new ObjLoaderBuffer(this));
        newObj->objNumber = ++numLoaded;
        break;
    default:
        fprintf(stderr, "Scene.addObject : undefined object type %d\n", fileType);
    }

    if (newObj->load(fileName))
    {
        this->curObject = newObj;
        this->objects.push_back(newObj);
    }
    else
        fprintf(stderr, "Error loading Object\n");
    repaint();
}

void Scene::deleteObject(int index)
{
    objects.erase(objects.begin()+index);
    curObject.reset();
    repaint();
}

void Scene::initializeGL()
{
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    axes = new Axes(this);
    rayTracer = new RayTracer(this);
}

void Scene::resizeGL(int x, int h)
{
    fprintf(stdout, "ResizeGL: width: %d height: %d\n", x, h);
    glViewport(0, 0, x, h);
    camera->setViewport(0.0, x, h, 0.0);
    adjustWindowAspect();
}

void Scene::paintGL()
{
    if (rayTrace)
    {
        glClearColor(clearColorR, clearColorG, clearColorB, clearColorA);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        rayTracer->render();
        fprintf(stdout, "End RayTrace\n");
        rayTrace = false;
    }
    else
    {
        glClearColor(clearColorR, clearColorG, clearColorB, clearColorA);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (updateShaders)
        {
            shUtil.setupShaders(shaderProg);
            updateShaders = false;
        }

        if (updateLight)
        {
            for (int i = 0; i < (int)updateLights.size(); i++)
            {
                if (updateLights[i])
                {
                    lights->updateLight(i);
                    updateLights[i] = false;
                }
            }
            updateLight = false;
        }

        camera->updateCamera();

        if (drawAxis)
            axes->draw();

        glUseProgram(shaderProg->progID);

        shared_ptr<PMesh> curObj;
        for (int i = 0; i < (int)objects.size(); i++)
        {
            curObj = objects.at(i);
            curObj->updateUniforms();
            curObj->draw(camera);
        }
    }
}
