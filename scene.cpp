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
    objects = new vector<PMesh>;
    curObject = NULL;
    lights = new Lights(this);
    updateLight = true;
    updateLights = vector<bool>(8);
    for (int i = 0; i < (int)updateLights.size(); i++)
    {
        lights->updateLight(i);
        updateLights[i] = false;
    }
    updateLight = false;

    shaderProg = new ShaderProgram();
    if (vertShaderName.compare(""))
        shaderProg->vertexShader = new ShaderProgram::Shader(defaultVertShader,false, false, -1, GL_VERTEX_SHADER, shaderProg);
    else
        shaderProg->vertexShader = new ShaderProgram::Shader(vertShaderName, false, false, -1, GL_VERTEX_SHADER, shaderProg);
    if (fragShaderName.compare(""))
        shaderProg->fragmentShader = new ShaderProgram::Shader(defaultFragShader, false, false, -1, GL_FRAGMENT_SHADER, shaderProg);
    else
        shaderProg->fragmentShader = new ShaderProgram::Shader(fragShaderName, false, false, -1, GL_FRAGMENT_SHADER, shaderProg);

    setupUniforms(shaderProg);

    shaders.push_back(shaderProg);
    drawAxis = true;
}

Scene::~Scene()
{
    delete camera;
    delete objects;
    delete lights;
    delete shaderProg;
    for (int x = 0; x < (int)shaders.size(); x++)
        shaders[x] = nullptr;
    delete axes;
}

void Scene::setupUniforms(ShaderProgram *theShader)
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

void Scene::initializeGL()
{
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    axes = new Axes(this);
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
    glClearColor(clearColorR, clearColorG, clearColorB, clearColorA);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

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

    camera->updateCamera(shaderProg);

    if (drawAxis)
        axes->draw();
}
