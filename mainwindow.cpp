#include <QFileDialog>
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "Loaders/objecttypes.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_DrawAxis_toggled(bool checked)
{
    ui->Canvas->drawAxis = checked;
    ui->Canvas->repaint();
}

void MainWindow::on_LoadObject_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Load Object"), "./Objects", tr("Objects (*.obj)"));
    if (!fileName.isEmpty())
    {
        ui->Canvas->addObject(fileName, ObjectTypes::TYPE_OBJ);
        if (!ui->Canvas->objects.empty())
        {
            QString itemName((*ui->Canvas->objects.back()).objName);
            ui->CurrentObject->addItem(itemName);
        }
    }
}

void MainWindow::on_Delete_clicked()
{
    int index = ui->CurrentObject->currentIndex();
    ui->Canvas->deleteObject(index);
    ui->CurrentObject->removeItem(index);
}

void MainWindow::on_RayTraceButton_clicked()
{
    fprintf(stdout, "Ray Tracing\n");
    ui->Canvas->rayTrace = true;
    ui->Canvas->repaint();
}

void MainWindow::on_RenderSpheres_toggled(bool checked)
{
    ui->Canvas->rayTracer->spheresOnly = checked;
}

void MainWindow::on_CheckeredBackground_toggled(bool checked)
{
    ui->Canvas->rayTracer->checkerBackground = checked;
}

void MainWindow::on_CheckerSize_valueChanged(double arg1)
{
    ui->Canvas->rayTracer->checkerSize = arg1;
}

void MainWindow::on_Reflections_toggled(bool checked)
{
    ui->Canvas->rayTracer->reflections = checked;
}

void MainWindow::on_Refractions_toggled(bool checked)
{
    ui->Canvas->rayTracer->refractions = checked;
}

void MainWindow::on_Shadows_toggled(bool checked)
{
    ui->Canvas->rayTracer->shadows = checked;
}

void MainWindow::on_CurrentObject_currentIndexChanged(int index)
{
    ui->Canvas->curObject = ui->Canvas->objects[index];
}

void MainWindow::on_spinBox_valueChanged(int arg1)
{
    ui->Canvas->rayTracer->maxRecursiveDepth = arg1;
}
