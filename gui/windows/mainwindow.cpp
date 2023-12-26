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

    spdlog::info("MainWindow constructor start");
    model.prms.Reset();
    params = new ParamsWrapper(&model.prms);
    worker = new BackgroundWorker(&model);
    snapshot.model = &model;
    representation.model = &model;
    model.gpu.initialize();

    // VTK
    qt_vtk_widget = new QVTKOpenGLNativeWidget();
    qt_vtk_widget->setRenderWindow(renderWindow);

    renderer->SetBackground(1.0,1.0,1.0);
    renderWindow->AddRenderer(renderer);
//    renderWindow->GetInteractor()->SetInteractorStyle(pointSelector);
    // pointSelector->clicked_on_a_point = [&](double x, double y) { point_selection(x,y);};


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
    slider1->setTracking(false);
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


    renderer->AddActor(representation.actor_points);
    renderer->AddActor(representation.actor_grid);
    renderer->AddActor(representation.actor_indenter);
    renderer->AddActor(representation.actorText);
    renderer->AddActor(representation.scalarBar);
    renderer->AddActor(representation.actor_axes);

    // populate combobox
    QMetaEnum qme = QMetaEnum::fromType<icy::VisualRepresentation::VisOpt>();
    for(int i=0;i<qme.keyCount();i++) comboBox_visualizations->addItem(qme.key(i));

    connect(comboBox_visualizations, QOverload<int>::of(&QComboBox::currentIndexChanged),
            [this](int index){ this->comboboxIndexChanged_visualizations(index); });

    // read/restore saved settings
    settingsFileName = QDir::currentPath() + "/cm.ini";
    QFileInfo fi(settingsFileName);

    vtkCamera* camera = renderer->GetActiveCamera();
    renderer->ResetCamera();
    if(fi.exists())
    {
        QSettings settings(settingsFileName,QSettings::IniFormat);
        QVariant var;
        var = settings.value("camData");
        if(!var.isNull())
        {
            double *data = (double*)var.toByteArray().constData();
            camera->SetPosition(data[0],data[1],data[2]);
            camera->SetFocalPoint(data[3],data[4],data[5]);
            camera->SetViewUp(data[6],data[7],data[8]);
            camera->SetViewAngle(data[9]);
            camera->SetClippingRange(1e-3,1e5);
        }


        var = settings.value("visualization_ranges");
        if(!var.isNull())
        {
            QByteArray ba = var.toByteArray();
            memcpy(representation.ranges, ba.constData(), ba.size());
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
    connect(ui->actionSave_Snapshot_As, &QAction::triggered, this, &MainWindow::save_snapshot_triggered);
    connect(ui->actionCreate_Video, &QAction::triggered, this, &MainWindow::createVideo_triggered);
    connect(ui->actionScreenshot, &QAction::triggered, this, &MainWindow::screenshot_triggered);
    connect(ui->actionStart_Pause, &QAction::triggered, this, &MainWindow::simulation_start_pause);
    connect(ui->actionLoad_Parameters, &QAction::triggered, this, &MainWindow::load_parameter_triggered);
    connect(ui->actionReplay, &QAction::triggered, this, &MainWindow::open_replay_triggered);

    connect(qdsbValRange,QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &MainWindow::limits_changed);

    connect(worker, SIGNAL(workerPaused()), SLOT(background_worker_paused()));
    connect(worker, SIGNAL(stepCompleted()), SLOT(simulation_data_ready()));

    representation.SynchronizeTopology();
    pbrowser->setActiveObject(params);
    updateGUI();
    spdlog::info("MainWindow constructor done");
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
    renderer->GetActiveCamera()->GetViewUp(&data[6]);
    data[9]=renderer->GetActiveCamera()->GetViewAngle();
    QByteArray arr((char*)data, sizeof(data));
    settings.setValue("camData", arr);

    QByteArray ranges((char*)representation.ranges, sizeof(representation.ranges));
    settings.setValue("visualization_ranges", ranges);

    settings.setValue("vis_option", comboBox_visualizations->currentIndex());

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
    renderWindow->Render();
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
    camera->Modified();
    renderWindow->Render();
}


void MainWindow::sliderValueChanged(int val)
{
    /*
//    int val = slider1->value();
    QString stringIdx = QString{"%1"}.arg(val,5, 10, QLatin1Char('0'));
    labelStepCount->setText(stringIdx);
    QString stringFileName = stringIdx + ".h5";
    stringFileName = QString::fromStdString(snapshot.path) + "/"+stringFileName;
    OpenFile(stringFileName);
*/
}






void MainWindow::createVideo_triggered()
{
    qDebug() << "MainWindow::createVideo_triggered()";
    if(!ui->actionTake_Screenshots->isChecked()) return;

    QString filePath = QDir::currentPath()+ "/video";
    QDir videoDir(filePath);
    if(!videoDir.exists()) videoDir.mkdir(filePath);

    renderWindow->DoubleBufferOff();
    windowToImageFilter->SetInputBufferTypeToRGBA(); //also record the alpha (transparency) channel

    // make screenshots while the files are available
    int count = -1;
    do
    {
        count++;
        representation.SynchronizeTopology();
        renderWindow->Render();
        renderWindow->WaitForCompletion();
        windowToImageFilter->Update();
        windowToImageFilter->Modified();

        writerPNG->Modified();
        int frame_index = model.prms.AnimationFrameNumber();
        QString stringIdx = QString{"%1"}.arg(frame_index,5, 10, QLatin1Char('0'));
        QString outputPath = filePath + "/" + stringIdx + ".png";
        writerPNG->SetFileName(outputPath.toUtf8().constData());
        writerPNG->Write();

    } while(snapshot.ReadNextFrame());

    renderWindow->DoubleBufferOn();


    std::string ffmpegCommand = "ffmpeg -y -r 60 -f image2 -start_number 0 -i \"" + filePath.toStdString() +
                                "/%05d.png\" -vframes " + std::to_string(count) +
                                " -vcodec libx264 -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" -crf 25  -pix_fmt yuv420p "+
        filePath.toStdString() + "/result.mp4\n";
    qDebug() << ffmpegCommand.c_str();
    int result = std::system(ffmpegCommand.c_str());

}







void MainWindow::screenshot_triggered()
{
    if(!ui->actionTake_Screenshots->isChecked()) return;
    if(model.prms.SimulationStep % model.prms.UpdateEveryNthStep) return;

    int screenshot_number = model.prms.AnimationFrameNumber();
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



void MainWindow::simulation_data_ready()
{
    updateGUI();
    screenshot_triggered();
    save_binary_data();
    model.UnlockCycleMutex();
}


void MainWindow::updateGUI()
{
 //   if(worker->running) statusLabel->setText("simulation is running");
 //   else statusLabel->setText("simulation is stopped");
    labelStepCount->setText(QString::number(model.prms.AnimationFrameNumber()));
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



void MainWindow::save_snapshot_triggered()
{
    QString qFileName = QFileDialog::getSaveFileName(this, "Save Snapshot", QDir::currentPath(), "HDF5 files (*.h5)");
    qDebug() << "manually saving snapshot";
    snapshot.SaveFullSnapshot(qFileName.toStdString());
    qDebug() << "snapshot saved";
}

void MainWindow::open_snapshot_triggered()
{
    qDebug() << "MainWindow::open_snapshot_triggered()";
    QString qFileName = QFileDialog::getOpenFileName(this, "Open Simulation Snapshot", QDir::currentPath(), "HDF5 Files (*.h5)");
    if(qFileName.isNull())return;
    snapshot.ReadFullSnapshot(qFileName.toStdString());
    representation.SynchronizeTopology();
    updateGUI();
    pbrowser->setActiveObject(params);
    snapshot.AllocateMemoryForFrames();
//    save_binary_data();
}

void MainWindow::load_parameter_triggered()
{
    QString qFileName = QFileDialog::getOpenFileName(this, "Load Parameters", QDir::currentPath(), "JSON Files (*.json)");
    if(qFileName.isNull())return;
    model.Reset();

    this->setWindowTitle(qFileName);
    std::pair<std::string, std::string> p = model.prms.ParseFile(qFileName.toStdString());
    model.outputDirectory = p.first;
    if(p.second == "") snapshot.GeneratePoints();
    else snapshot.ReadRawPoints(p.second);

    representation.SynchronizeTopology();
    pbrowser->setActiveObject(params);
    updateGUI();

    snapshot.AllocateMemoryForFrames();
    if(ui->actionSave_Binary_Data->isChecked()) snapshot.SaveFrame();
}

void MainWindow::save_binary_data()
{
    constexpr int save_full_snapshot_every = 100;
    if(!ui->actionSave_Binary_Data->isChecked()) return;
    qDebug() << "MainWindow::save_binary_data()";
    snapshot.SaveFrame();

    // once in 100 frames save full data (expect 20 GB per file)
    int frame = model.prms.AnimationFrameNumber();
    if(frame%save_full_snapshot_every == 0 && frame != 0)
    {
        QString filePath = QDir::currentPath()+ "/full_snapshots";
        QDir fileDir(filePath);
        if(!fileDir.exists()) fileDir.mkdir(filePath);

        QString stringIdx = QString{"%1"}.arg(frame,5, 10, QLatin1Char('0'));
        QString outputPath = filePath + "/" + "s" + stringIdx + ".h5";
        snapshot.SaveFullSnapshot(outputPath.toStdString());
    }
}


void MainWindow::open_replay_triggered()
{
    qDebug() << "MainWindow::open_replay_triggered()";
    QString dir = QFileDialog::getExistingDirectory(this, tr("Open Directory"), QDir::currentPath(), QFileDialog::ShowDirsOnly);
    if(dir.isNull()) return;
    snapshot.ReadFirstFrame(dir.toStdString());
    this->setWindowTitle(dir);
    representation.SynchronizeTopology();
    pbrowser->setActiveObject(params);
    updateGUI();
}
