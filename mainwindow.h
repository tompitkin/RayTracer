#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT
    
public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    
private slots:
    void on_DrawAxis_toggled(bool checked);

    void on_LoadObject_clicked();

    void on_Delete_clicked();

    void on_RayTraceButton_clicked();

    void on_RenderSpheres_toggled(bool checked);

    void on_CheckeredBackground_toggled(bool checked);

    void on_CheckerSize_valueChanged(double arg1);

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
