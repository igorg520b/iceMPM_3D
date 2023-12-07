#include <QFileDialog>
#include <QList>
#include <QPointF>
#include <QCloseEvent>
#include <QStringList>
#include <algorithm>
#include <cmath>
#include <fstream>
#include "mainwindow.h"
#include "./ui_mainwindow.h"

MainWindow::~MainWindow() {delete ui;}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    params = new ParamsWrapper(&model.prms);
    worker = new BackgroundWorker(&model);
    snapshot.model = &model;
    model.gpu.initialize();
    representation.model = &model;

    // VTK
    qt_vtk_widget = new QVTKOpenGLNativeWidget();
    qt_vtk_widget->setRenderWindow(renderWindow);

    renderer->SetBackground(1.0,1.0,1.0);
    renderWindow->AddRenderer(renderer);
    renderWindow->GetInteractor()->SetInteractorStyle(pointSelector);

    pointSelector->clicked_on_a_point = [&](double x, double y) { point_selection(x,y);};


    // property browser
    pbrowser = new ObjectPropertyBrowser(this);

    // splitter
    splitter = new QSplitter(Qt::Orientation::Horizontal);
    splitter->addWidget(pbrowser);
    splitter->addWidget(qt_vtk_widget);
    splitter->setSizes(QList<int>({100, 500}));
    setCentralWidget(splitter);

    // toolbar - combobox
    comboBox_visualizations = new QComboBox();
    ui->toolBar->addWidget(comboBox_visualizations);

    // double spin box
    qdsbValRange = new QDoubleSpinBox();
    qdsbValRange->setRange(-10, 10);
    qdsbValRange->setValue(-2);
    qdsbValRange->setDecimals(2);
    qdsbValRange->setSingleStep(0.25);
    ui->toolBar->addWidget(qdsbValRange);

    // slider
    ui->toolBar->addSeparator();
    slider1 = new QSlider(Qt::Horizontal);
    ui->toolBar->addWidget(slider1);
    slider1->setTracking(true);
    slider1->setRange(0,0);
    connect(slider1, SIGNAL(valueChanged(int)), this, SLOT(sliderValueChanged(int)));


    // statusbar
    statusLabel = new QLabel();
    labelElapsedTime = new QLabel();
    labelStepCount = new QLabel();

    QSizePolicy sp;
    const int status_width = 80;
    sp.setHorizontalPolicy(QSizePolicy::Fixed);
    labelStepCount->setSizePolicy(sp);
    labelStepCount->setFixedWidth(status_width);
    labelElapsedTime->setSizePolicy(sp);
    labelElapsedTime->setFixedWidth(status_width);

    ui->statusbar->addWidget(statusLabel);
    ui->statusbar->addPermanentWidget(labelElapsedTime);
    ui->statusbar->addPermanentWidget(labelStepCount);

// anything that includes the Model


    renderer->AddActor(representation.actor_points);
    renderer->AddActor(representation.actor_grid);
    renderer->AddActor(representation.actor_indenter);
    renderer->AddActor(representation.actorText);
    renderer->AddActor(representation.scalarBar);


    // populate combobox
    QMetaEnum qme = QMetaEnum::fromType<icy::VisualRepresentation::VisOpt>();
    for(int i=0;i<qme.keyCount();i++) comboBox_visualizations->addItem(qme.key(i));

    connect(comboBox_visualizations, QOverload<int>::of(&QComboBox::currentIndexChanged),
            [&](int index){ comboboxIndexChanged_visualizations(index); });

    // read/restore saved settings
    settingsFileName = QDir::currentPath() + "/cm.ini";
    QFileInfo fi(settingsFileName);

    if(fi.exists())
    {
        QSettings settings(settingsFileName,QSettings::IniFormat);
        QVariant var;

        vtkCamera* camera = renderer->GetActiveCamera();
        renderer->ResetCamera();
        camera->ParallelProjectionOn();

        var = settings.value("camData");
        if(!var.isNull())
        {
            double *vec = (double*)var.toByteArray().constData();
            camera->SetClippingRange(1e-1,1e4);
            camera->SetViewUp(0.0, 1.0, 0.0);
            camera->SetPosition(vec[0],vec[1],vec[2]);
            camera->SetFocalPoint(vec[3],vec[4],vec[5]);
            camera->SetParallelScale(vec[6]);
            camera->Modified();
        }

        var = settings.value("visualization_ranges");
        if(!var.isNull())
        {
            QByteArray ba = var.toByteArray();
            memcpy(representation.ranges, ba.constData(), ba.size());
        }

        var = settings.value("lastParameterFile");
        if(!var.isNull())
        {
            qLastParameterFile = var.toString();
            QFile paramFile(qLastParameterFile);
            if(paramFile.exists())
            {
                this->outputDirectory = model.prms.ParseFile(qLastParameterFile.toStdString());
                this->setWindowTitle(qLastParameterFile);
                model.Reset();
            }
        }

        comboBox_visualizations->setCurrentIndex(settings.value("vis_option").toInt());

        var = settings.value("take_screenshots");
        if(!var.isNull()) ui->actionTake_Screenshots->setChecked(var.toBool());

        var = settings.value("save_binary_data");
        if(!var.isNull()) ui->actionSave_Binary_Data->setChecked(var.toBool());

        var = settings.value("splitter_size_0");
        if(!var.isNull())
        {
            int sz1 = var.toInt();
            int sz2 = settings.value("splitter_size_1").toInt();
            splitter->setSizes(QList<int>({sz1, sz2}));
        }

        var = settings.value("vis_option");
        if(!var.isNull())
        {
            comboBox_visualizations->setCurrentIndex(var.toInt());
            qdsbValRange->setValue(representation.ranges[var.toInt()]);
        }
    }
    else
    {
        cameraReset_triggered();
    }

    windowToImageFilter->SetInput(renderWindow);
    windowToImageFilter->SetScale(1); // image quality
    windowToImageFilter->SetInputBufferTypeToRGBA(); //also record the alpha (transparency) channel
    windowToImageFilter->ReadFrontBufferOn(); // read from the back buffer
    writerPNG->SetInputConnection(windowToImageFilter->GetOutputPort());

    connect(ui->action_quit, &QAction::triggered, this, &MainWindow::quit_triggered);
    connect(ui->action_camera_reset, &QAction::triggered, this, &MainWindow::cameraReset_triggered);
    connect(ui->actionOpen, &QAction::triggered, this, &MainWindow::open_snapshot_triggered);
    connect(ui->actionCreate_Video, &QAction::triggered, this, &MainWindow::createVideo_triggered);
    connect(ui->actionScreenshot, &QAction::triggered, this, &MainWindow::screenshot_triggered);
    connect(ui->actionStart_Pause, &QAction::triggered, this, &MainWindow::simulation_start_pause);
    connect(ui->actionLoad_Parameters, &QAction::triggered, this, &MainWindow::load_parameter_triggered);
    connect(ui->actionReset, &QAction::triggered, this, &MainWindow::simulation_reset_triggered);

    connect(ui->actionExport_Indenter_Forces, &QAction::triggered, this, &MainWindow::export_indenter_force_triggered);


    connect(qdsbValRange,QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &MainWindow::limits_changed);

    connect(worker, SIGNAL(workerPaused()), SLOT(background_worker_paused()));
    connect(worker, SIGNAL(stepCompleted()), SLOT(simulation_data_ready()));

    representation.SynchronizeTopology();
    pbrowser->setActiveObject(params);
    updateGUI();
}


void MainWindow::closeEvent(QCloseEvent* event)
{
    quit_triggered();
    event->accept();
}


void MainWindow::quit_triggered()
{
    qDebug() << "MainWindow::quit_triggered() ";
    worker->Finalize();
    // save settings and stop simulation
    QSettings settings(settingsFileName,QSettings::IniFormat);
    qDebug() << "MainWindow: closing main window; " << settings.fileName();

    double data[10];
    renderer->GetActiveCamera()->GetPosition(&data[0]);
    renderer->GetActiveCamera()->GetFocalPoint(&data[3]);
    data[6] = renderer->GetActiveCamera()->GetParallelScale();

    qDebug() << "cam pos " << data[0] << "," << data[1] << "," << data[2];
    qDebug() << "cam focal pt " << data[3] << "," << data[4] << "," << data[5];
    qDebug() << "cam par scale " << data[6];

    QByteArray arr((char*)data, sizeof(data));
    settings.setValue("camData", arr);

    QByteArray ranges((char*)representation.ranges, sizeof(representation.ranges));
    settings.setValue("visualization_ranges", ranges);

    settings.setValue("vis_option", comboBox_visualizations->currentIndex());

    if(!qLastParameterFile.isEmpty()) settings.setValue("lastParameterFile", qLastParameterFile);
    settings.setValue("take_screenshots", ui->actionTake_Screenshots->isChecked());
    settings.setValue("save_binary_data", ui->actionSave_Binary_Data->isChecked());

    QList<int> szs = splitter->sizes();
    settings.setValue("splitter_size_0", szs[0]);
    settings.setValue("splitter_size_1", szs[1]);
    QApplication::quit();
}



void MainWindow::comboboxIndexChanged_visualizations(int index)
{
    representation.ChangeVisualizationOption(index);
//    scalarBar->SetVisibility(index != 0);
//    renderWindow->Render();
    qdsbValRange->setValue(representation.ranges[index]);
}

void MainWindow::limits_changed(double val_)
{
//    double val = qdsbValRange->value();
//    representation.value_range = val;
    int idx = (int)representation.VisualizingVariable;
    representation.ranges[idx] = val_;
    representation.SynchronizeValues();
    renderWindow->Render();
}

void MainWindow::cameraReset_triggered()
{
    qDebug() << "MainWindow::on_action_camera_reset_triggered()";
    vtkCamera* camera = renderer->GetActiveCamera();
    renderer->ResetCamera();
    camera->ParallelProjectionOn();
    camera->SetClippingRange(1e-1,1e3);
    camera->SetFocalPoint(0, 0., 0.);
    camera->SetPosition(0.0, 0.0, 50.0);
    camera->SetViewUp(0.0, 1.0, 0.0);
    camera->SetParallelScale(2.5);

    camera->Modified();
    renderWindow->Render();
}


void MainWindow::sliderValueChanged(int val)
{
//    int val = slider1->value();
    QString stringIdx = QString{"%1"}.arg(val,5, 10, QLatin1Char('0'));
    labelStepCount->setText(stringIdx);
    QString stringFileName = stringIdx + ".h5";
    stringFileName = QString::fromStdString(snapshot.path) + "/"+stringFileName;
    OpenFile(stringFileName);
}


void MainWindow::open_snapshot_triggered()
{
    QString qFileName = QFileDialog::getOpenFileName(this, "Open Simulation Snapshot", QDir::currentPath(), "HDF5 Files (*.h5)");
    if(qFileName.isNull())return;
    int idx = OpenFile(qFileName);
    QString fileDirectory = QFileInfo(qFileName).absolutePath();
    snapshot.ReadDirectory(fileDirectory.toStdString());
    slider1->blockSignals(true);
    slider1->setEnabled(snapshot.last_file_index > 1);
    slider1->setRange(1,snapshot.last_file_index);
    slider1->setValue(idx);
    slider1->blockSignals(false);
}

void MainWindow::simulation_reset_triggered()
{
    model.ResetToStep0();
    representation.SynchronizeTopology();
    updateGUI();
    renderWindow->Render();
}

int MainWindow::OpenFile(QString fileName)
{
    int idx = snapshot.ReadSnapshot(fileName.toStdString());
    representation.SynchronizeTopology();
    updateGUI();
    pbrowser->setActiveObject(params);
    return idx;
}

void MainWindow::load_parameter_triggered()
{
    QString qFileName = QFileDialog::getOpenFileName(this, "Load Parameters", QDir::currentPath(), "JSON Files (*.json)");
    if(qFileName.isNull())return;
    this->outputDirectory = model.prms.ParseFile(qFileName.toStdString());
    this->qLastParameterFile = qFileName;
    this->setWindowTitle(qLastParameterFile);
    model.Reset();
    representation.SynchronizeTopology();
//    if(ui->actionSave_Binary_Data->isChecked() && model.prms.SimulationStep == 0) save_binary_data();
    pbrowser->setActiveObject(params);
    updateGUI();
}



void MainWindow::createVideo_triggered()
{
    QString filePath = QDir::currentPath()+ "/video";

    QDir videoDir(filePath);
    if(!videoDir.exists()) videoDir.mkdir(filePath);

    renderWindow->DoubleBufferOff();
    windowToImageFilter->SetInputBufferTypeToRGBA(); //also record the alpha (transparency) channel

    int nFrames = snapshot.last_file_index;
    for(int i=1; i<=nFrames; i++)
    {
        QString stringIdx = QString{"%1"}.arg(i,5, 10, QLatin1Char('0'));
        labelStepCount->setText(stringIdx);
        QString stringFileName = stringIdx + ".h5";
        stringFileName = QString::fromStdString(snapshot.path) + "/"+stringFileName;
        OpenFile(stringFileName);
        renderWindow->WaitForCompletion();

        QString outputPath = filePath + "/" + stringIdx + ".png";

        windowToImageFilter->Update();
        windowToImageFilter->Modified();

        writerPNG->Modified();
        writerPNG->SetFileName(outputPath.toUtf8().constData());
        writerPNG->Write();
    }
    renderWindow->DoubleBufferOn();

    std::string ffmpegCommand = "ffmpeg -y -r 60 -f image2 -start_number 1 -i \"" + filePath.toStdString() +
                                "/%05d.png\" -vframes " + std::to_string(nFrames) +
                                " -vcodec libx264 -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" -crf 25  -pix_fmt yuv420p "+
        filePath.toStdString() + "/result.mp4\n";
    qDebug() << ffmpegCommand.c_str();
    int result = std::system(ffmpegCommand.c_str());
}







void MainWindow::screenshot_triggered()
{
    if(model.prms.SimulationStep % model.prms.UpdateEveryNthStep) return;
    int screenshot_number = model.prms.SimulationStep / model.prms.UpdateEveryNthStep;
    QString outputPath = QDir::currentPath()+ "/" + screenshot_directory.c_str() + "/" +
            QString::number(screenshot_number).rightJustified(5, '0') + ".png";

    QDir pngDir(QDir::currentPath()+ "/"+ screenshot_directory.c_str());
    if(!pngDir.exists()) pngDir.mkdir(QDir::currentPath()+ "/"+ screenshot_directory.c_str());

    renderWindow->DoubleBufferOff();
    renderWindow->Render();
    windowToImageFilter->SetInputBufferTypeToRGBA(); //also record the alpha (transparency) channel
    renderWindow->WaitForCompletion();


    windowToImageFilter->Update();
    windowToImageFilter->Modified();

    writerPNG->Modified();
    writerPNG->SetFileName(outputPath.toUtf8().constData());
    writerPNG->Write();
    renderWindow->DoubleBufferOn();

}

void MainWindow::save_binary_data()
{
    QDir pngDir(QDir::currentPath()+ "/"+ outputDirectory.c_str());
    if(!pngDir.exists()) pngDir.mkdir(QDir::currentPath()+ "/"+ outputDirectory.c_str());

    int snapshot_number = model.prms.SimulationStep / model.prms.UpdateEveryNthStep;
    QString outputPathSnapshot = QDir::currentPath()+ "/"+outputDirectory.c_str() + "/" +
                         QString::number(snapshot_number).rightJustified(5, '0') + ".h5";
    snapshot.SaveSnapshot(outputPathSnapshot.toStdString());
}


void MainWindow::simulation_data_ready()
{
    updateGUI();
    if(worker->running && ui->actionTake_Screenshots->isChecked()) screenshot_triggered();
    if(worker->running && ui->actionSave_Binary_Data->isChecked()) save_binary_data();
    model.UnlockCycleMutex();
}


void MainWindow::updateGUI()
{
 //   if(worker->running) statusLabel->setText("simulation is running");
 //   else statusLabel->setText("simulation is stopped");
    labelStepCount->setText(QString::number(model.prms.SimulationStep));
    labelElapsedTime->setText(QString("%1 s").arg(model.prms.SimulationTime,0,'f',3));
    statusLabel->setText(QString("per cycle: %1 ms").arg(model.compute_time_per_cycle,0,'f',3));

    representation.SynchronizeValues();
    renderWindow->Render();

    worker->visual_update_requested = false;
}

void MainWindow::simulation_start_pause(bool checked)
{
    if(!worker->running && checked)
    {
        qDebug() << "starting simulation via GUI";
        statusLabel->setText("starting simulation");
        worker->Resume();
    }
    else if(worker->running && !checked)
    {
        qDebug() << "pausing simulation via GUI";
        statusLabel->setText("pausing simulation");
        worker->Pause();
        ui->actionStart_Pause->setEnabled(false);
    }
}

void MainWindow::background_worker_paused()
{
    ui->actionStart_Pause->blockSignals(true);
    ui->actionStart_Pause->setEnabled(true);
    ui->actionStart_Pause->setChecked(false);
    ui->actionStart_Pause->blockSignals(false);
    statusLabel->setText("simulation stopped");
}

void MainWindow::point_selection(double x, double y)
{
    /*
    qDebug() << QString("clicked %1; %2").arg(x).arg(y);
    int idx = representation.FindPoint(x,y);
    qDebug() << QString("found point index %1").arg(idx);
    snapshot.DumpPointData(idx);
*/
}

void MainWindow::export_indenter_force_triggered()
{
    std::ofstream ofs("indenter_force.csv", std::ofstream::out | std::ofstream::trunc);
    ofs << "fx,fy,F_total\n";
    for(int i=0;i<model.indenter_force_history.size();i++)
    {
        Vector2r v = model.indenter_force_history[i];
        ofs << v[0] << ',' << v[1] << ',' << v.norm() << '\n';
    }
    ofs.close();
    qDebug() << "export_indenter_force_triggered()";
}

