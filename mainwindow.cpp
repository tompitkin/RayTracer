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
            ui->CurrentObject->repaint();
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

void MainWindow::on_checkBox_toggled(bool checked)
{
    ui->Canvas->rayTracer->spheresOnly = checked;
}
