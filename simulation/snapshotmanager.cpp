#include "snapshotmanager.h"
#include "model_3d.h"

#include <spdlog/spdlog.h>
#include <H5Cpp.h>
#include "poisson_disk_sampling.h"

#include <filesystem>
#include <string>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <cstdio>
#include <cmath>

#include <vtkCellArray.h>
#include <vtkNew.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkFloatArray.h>
#include <vtkPointData.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkCylinderSource.h>
#include <vtkPolyDataMapper.h>
#include <vtkTransform.h>
#include <vtkTransformFilter.h>
#include <vtkAppendFilter.h>

void icy::SnapshotManager::SaveFullSnapshot(std::string fileName)
{
    spdlog::info("writing snapshot {}",fileName);

    H5::H5File file(fileName, H5F_ACC_TRUNC);

    hsize_t dims_params = sizeof(icy::SimParams3D);
    H5::DataSpace dataspace_params(1,&dims_params);
    H5::DataSet dataset_params = file.createDataSet("Params", H5::PredType::NATIVE_B8, dataspace_params);
    dataset_params.write(&model->prms, H5::PredType::NATIVE_B8);

    hsize_t dims_points = model->prms.nPtsPitch * icy::SimParams3D::nPtsArrays;
    hsize_t dims_unlimited = H5S_UNLIMITED;
    H5::DataSpace dataspace_points(1, &dims_points, &dims_unlimited);

    hsize_t chunk_dims = 1024*1024;
    if(chunk_dims > dims_points) chunk_dims = dims_points/10;
    H5::DSetCreatPropList proplist;
    proplist.setChunk(1, &chunk_dims);
    proplist.setDeflate(6);
    H5::DataSet dataset_points = file.createDataSet("Points", H5::PredType::NATIVE_DOUBLE, dataspace_points, proplist);
    dataset_points.write(model->gpu.tmp_transfer_buffer, H5::PredType::NATIVE_DOUBLE);

    // save indenter force history
    hsize_t dims_indenter_history[2] {model->gpu.indenter_force_history.size(), 3};
    H5::DataSpace dataspace_history(2, dims_indenter_history);
    H5::DataSet dataset_history = file.createDataSet("Indenter_Force_History", H5::PredType::NATIVE_DOUBLE, dataspace_history);
    dataset_history.write(model->gpu.indenter_force_history.data(), H5::PredType::NATIVE_DOUBLE);

    file.close();
    spdlog::info("SaveSnapshot done {}", fileName);
}

void icy::SnapshotManager::ReadFullSnapshot(std::string fileName)
{
    if(!std::filesystem::exists(fileName)) return;

    H5::H5File file(fileName, H5F_ACC_RDONLY);

    // read and process SimParams
    H5::DataSet dataset_params = file.openDataSet("Params");
    hsize_t dims_params = 0;
    dataset_params.getSpace().getSimpleExtentDims(&dims_params, NULL);
    if(dims_params != sizeof(icy::SimParams3D)) throw std::runtime_error("ReadSnapshot: SimParams3D size mismatch");
    icy::SimParams3D tmp_params = model->prms;
    dataset_params.read(&model->prms, H5::PredType::NATIVE_B8);
    if(tmp_params.nGridPitch != model->prms.nGridPitch || tmp_params.nPtsPitch != model->prms.nPtsPitch)
        model->gpu.cuda_allocate_arrays(model->prms.nGridPitch, model->prms.nPtsPitch);

    // read point data
    H5::DataSet dataset_points = file.openDataSet("Points");
    dataset_points.read(model->gpu.tmp_transfer_buffer,H5::PredType::NATIVE_DOUBLE);

    // read indenter force history
    hsize_t dims_indenter_history[2] {};
    H5::DataSet dataset_history = file.openDataSet("Indenter_Force_History");
    dataset_history.getSpace().getSimpleExtentDims(dims_indenter_history, NULL);
    model->gpu.indenter_force_history.resize(dims_indenter_history[0]);
    dataset_history.read(model->gpu.indenter_force_history.data(), H5::PredType::NATIVE_DOUBLE);

    file.close();
    previous_frame_exists = false;
    model->outputDirectory = "default_output";

    model->gpu.transfer_ponts_to_device();
    model->Prepare();
}


void icy::SnapshotManager::AllocateMemoryForFrames()
{
    int n = model->prms.nPts;
    current_frame.resize(n);
    previous_frame.resize(n);
    saved_frame.reserve(n);

    last_refresh_frame.resize(n);
    previous_frame_exists = false;
}

void icy::SnapshotManager::SaveFrame()
{
    spdlog::info("icy::SnapshotManager::SaveFrame()");
    std::filesystem::path odp(model->outputDirectory);
    if(!std::filesystem::is_directory(odp) || !std::filesystem::exists(odp)) std::filesystem::create_directory(odp);

    if(export_vtp)
    {
        ExportPointsAsVTP();
        ExportIndenterAsVTU();
    }
    if(export_h5) ExportPointsAsH5();
    if(export_force) WriteIndenterForceCSV();
}


void icy::SnapshotManager::ReadRawPoints(std::string fileName)
{
    spdlog::info("ReadRawPoints {}",fileName);
    if(!std::filesystem::exists(fileName)) throw std::runtime_error("error reading raw points file - no file");;

    spdlog::info("reading raw points file {}",fileName);
    H5::H5File file(fileName, H5F_ACC_RDONLY);

    H5::DataSet dataset = file.openDataSet("Points_Raw");
    hsize_t dims[2] = {};
    dataset.getSpace().getSimpleExtentDims(dims, NULL);
    int nPoints = dims[0];
    if(dims[1]!=3) throw std::runtime_error("error reading raw points file - dimensions mismatch");
    spdlog::info("dims[0] {}, dims[1] {}", dims[0], dims[1]);
    model->prms.nPts = nPoints;

    std::vector<std::array<float, 3>> buffer;
    buffer.resize(nPoints);
    dataset.read(buffer.data(), H5::PredType::NATIVE_FLOAT);
    file.close();

    model->gpu.cuda_allocate_arrays(model->prms.GridTotal, nPoints);

    const real &h = model->prms.cellsize;
    const real &bz = model->prms.IceBlockDimZ;
    const real box_z = model->prms.GridZ*h;

    const real z_offset = (box_z - bz)/2;

    model->prms.ParticleVolume = model->prms.bvol()/nPoints;
    model->prms.ParticleMass = model->prms.ParticleVolume * model->prms.Density;


    for(int k=0; k<nPoints; k++)
    {
        Point3D p;
        p.Reset();
        buffer[k][0] += 5*h;
        buffer[k][1] += 2*h;
        buffer[k][2] += z_offset; // center
        for(int i=0;i<3;i++) p.pos[i] = buffer[k][i];
        p.pos_initial = p.pos;
        p.TransferToBuffer(model->gpu.tmp_transfer_buffer, model->prms.nPtsPitch, k);
    }
    spdlog::info("raw points loaded");

    model->gpu.transfer_ponts_to_device();
    model->Reset();
    model->Prepare();
}

void icy::SnapshotManager::GeneratePoints()
{
    spdlog::info("icy::SnapshotManager::GeneratePoints()");
    const real &bx = model->prms.IceBlockDimX;
    const real &by = model->prms.IceBlockDimY;
    const real &bz = model->prms.IceBlockDimZ;
    const real bvol = model->prms.bvol();
    const real &h = model->prms.cellsize;
    constexpr real magic_constant = 0.5844;

    const real z_center = model->prms.GridZ*h/2;
    const real block_z_min = std::max(z_center - bz/2, 0.0);
    const real block_z_max = std::min(z_center + bz/2, model->prms.GridZ*h);
    spdlog::info("block_z_range [{}, {}]", block_z_min, block_z_max);

    const real kRadius = cbrt(magic_constant*bvol/model->prms.PointsWanted);
    const std::array<real, 3>kXMin{5.0*h, 2.0*h, block_z_min};
    const std::array<real, 3>kXMax{5.0*h+bx, 2.0*h+by, block_z_max};

    spdlog::info("starting thinks::PoissonDiskSampling");
    std::vector<std::array<real, 3>> prresult = thinks::PoissonDiskSampling(kRadius, kXMin, kXMax);
    int n = prresult.size();
    model->prms.nPts = n;
    spdlog::info("finished thinks::PoissonDiskSampling; {} ", n);
    model->gpu.cuda_allocate_arrays(model->prms.GridTotal, n);

    model->prms.ParticleVolume = bvol/n;
    model->prms.ParticleMass = model->prms.ParticleVolume * model->prms.Density;
    for(int k = 0; k<n; k++)
    {
        Point3D p;
        p.Reset();
        for(int i=0;i<3;i++) p.pos[i] = prresult[k][i];
        p.pos_initial = p.pos;
        p.TransferToBuffer(model->gpu.tmp_transfer_buffer, model->prms.nPtsPitch, k);
    }
    spdlog::info("icy::SnapshotManager::GeneratePoints() done");

    model->gpu.transfer_ponts_to_device();
    model->Reset();
    model->Prepare();
}


void icy::SnapshotManager::ExportPointsAsH5()
{
    spdlog::info("icy::SnapshotManager::ExportPointsAsH5()");

    // populate current_frame (pull and convert to float: idx, pos, vel, Jp_inv)
    for(int i=0;i<model->prms.nPts;i++)
    {
        VisualPoint &vp = current_frame[i];
        vp.id = i;
        vp.Jp_inv = (float)icy::Point3D::getJp_inv(model->gpu.tmp_transfer_buffer, model->prms.nPtsPitch, i);
        Vector3r pos = icy::Point3D::getPos(model->gpu.tmp_transfer_buffer, model->prms.nPtsPitch, i);
        Vector3r vel = icy::Point3D::getVelocity(model->gpu.tmp_transfer_buffer, model->prms.nPtsPitch, i);
        for(int j=0;j<3;j++)
        {
            vp.p[j] = (float) pos[j];
            vp.v[j] = (float) vel[j];
        }
    }

    // populate saved_frame
    const int current_frame_number = model->prms.AnimationFrameNumber();

    if(!previous_frame_exists)
    {
        // saved_frame is a full copy of the current_frame
        saved_frame = current_frame;
        std::fill(last_refresh_frame.begin(), last_refresh_frame.end(), current_frame_number);
        previous_frame_exists = true;
        previous_frame = current_frame;
        spdlog::info("saving full frame");
    }
    else
    {
        const float visual_threshold = 2e-3;
        const float Jp_inv_threshold = 1e-4;
        float dt = (float)model->prms.InitialTimeStep;
        saved_frame.clear();
        for(int i=0;i<model->prms.nPts;i++)
        {
            VisualPoint &p_c = current_frame[i];
            VisualPoint &p_p = previous_frame[i];
            int previous_update = last_refresh_frame[i];
            int elapsed_frames = current_frame_number - previous_update;
            Eigen::Vector3f predicted_position = p_p.pos() + p_p.vel()*(dt*elapsed_frames);
            Eigen::Vector3f prediction_error = p_c.pos() - predicted_position;
            float Jp_diff = abs(p_c.Jp_inv - p_p.Jp_inv);
            if(prediction_error.norm() > visual_threshold || Jp_diff > Jp_inv_threshold)
            {
                last_refresh_frame[i] = current_frame_number;
                p_p = p_c;
                saved_frame.push_back(p_c);
            }
        }
        spdlog::info("saving difference; saved_frame.size() = {}",saved_frame.size());
    }

    char fileName[20];
    snprintf(fileName, sizeof(fileName), "v%05d.h5", current_frame_number);
    std::string saveDir = model->outputDirectory + "/" + dir_points_h5;
    std::string fullFilePath = saveDir + "/" + fileName;
    spdlog::info("saving frame {} to file {}", current_frame_number, fullFilePath);

    // ensure that directory exists
    std::filesystem::path od(saveDir);
    if(!std::filesystem::is_directory(od) || !std::filesystem::exists(od)) std::filesystem::create_directory(od);

    // save!
    H5::H5File file(fullFilePath, H5F_ACC_TRUNC);

    hsize_t dims_params = sizeof(icy::SimParams3D);
    H5::DataSpace dataspace_params(1,&dims_params);
    H5::DataSet dataset_params = file.createDataSet("Params", H5::PredType::NATIVE_B8, dataspace_params);
    dataset_params.write(&model->prms, H5::PredType::NATIVE_B8);

    hsize_t dims_indenter_force = model->prms.indenter_array_size;
    hsize_t chunk_dims_indenter = 10000;
    if(chunk_dims_indenter > dims_indenter_force) chunk_dims_indenter = dims_indenter_force/10;
    H5::DSetCreatPropList proplist2;
    proplist2.setChunk(1, &chunk_dims_indenter);
    proplist2.setDeflate(5);
    H5::DataSpace dataspace_indneter_force(1, &dims_indenter_force);
    H5::DataSet dataset_indneter_force = file.createDataSet("Indenter_Force", H5::PredType::NATIVE_DOUBLE, dataspace_indneter_force, proplist2);
    dataset_indneter_force.write(model->gpu.host_side_indenter_force_accumulator, H5::PredType::NATIVE_DOUBLE);

    hsize_t dims_points = sizeof(VisualPoint)*saved_frame.size();
    hsize_t dims_unlimited = H5S_UNLIMITED;
    H5::DataSpace dataspace_points(1, &dims_points, &dims_unlimited);
    hsize_t chunk_dims = sizeof(VisualPoint)*1024;
    if(chunk_dims > dims_points) chunk_dims = dims_points/10;
    if(chunk_dims == 0) chunk_dims = 1;
    H5::DSetCreatPropList proplist;
    proplist.setChunk(1, &chunk_dims);
    proplist.setDeflate(5);
    H5::DataSet dataset_points = file.createDataSet("Points", H5::PredType::NATIVE_B8, dataspace_points, proplist);
    dataset_points.write(saved_frame.data(), H5::PredType::NATIVE_B8);

    file.close();
}


void icy::SnapshotManager::ExportPointsAsVTP()
{
    std::string saveDir = model->outputDirectory + "/" + dir_vtp;

    // ensure that directory exists
    std::filesystem::path od(saveDir);
    if(!std::filesystem::is_directory(od) || !std::filesystem::exists(od)) std::filesystem::create_directory(od);

    int frame = model->prms.AnimationFrameNumber();

    char fileName[20];
    snprintf(fileName, sizeof(fileName), "p_%05d.vtp", frame);
    std::string savePath = saveDir + "/" + fileName;
    spdlog::info("export points {} to file {}", frame, savePath);



    spdlog::info("icy::SnapshotManager::ExportPointsAsVTP()");
    spdlog::info("export_vtu {}", frame);
    int n = model->prms.nPts;

    vtkNew<vtkPoints> points;
    vtkNew<vtkFloatArray> values;
    points->SetNumberOfPoints(n);
    values->SetNumberOfValues(n);
    values->SetName("Jp_inv");

    for(int i=0;i<n;i++)
    {
        Vector3r pos = icy::Point3D::getPos(model->gpu.tmp_transfer_buffer, model->prms.nPtsPitch, i);
        double Jp_inv = icy::Point3D::getJp_inv(model->gpu.tmp_transfer_buffer, model->prms.nPtsPitch, i);
        points->SetPoint(i, pos[0], pos[1], pos[2]);
        values->SetValue(i, Jp_inv);
    }
    values->Modified();

    vtkNew<vtkPolyData> polydata;
    polydata->SetPoints(points);
    polydata->GetPointData()->AddArray(values);
    polydata->GetPointData()->SetActiveScalars("Jp_inv");

    // Write the file
    vtkNew<vtkXMLPolyDataWriter> writer;
    writer->SetFileName(savePath.c_str());
    writer->SetInputData(polydata);
    writer->Write();

    spdlog::info("export_vtk frame done {}", frame);
}

void icy::SnapshotManager::ExportIndenterAsVTU()
{
    std::string saveDir = model->outputDirectory + "/" + dir_indenter;

    // ensure that directory exists
    std::filesystem::path od(saveDir);
    if(!std::filesystem::is_directory(od) || !std::filesystem::exists(od)) std::filesystem::create_directory(od);

    int frame = model->prms.AnimationFrameNumber();

    char fileName[20];
    snprintf(fileName, sizeof(fileName), "i_%05d.vtu", frame);
    std::string savePath = saveDir + "/" + fileName;
    spdlog::info("export indenter vtu {} to file {}", frame, savePath);

    vtkNew<vtkCylinderSource> cylinder;
    vtkNew<vtkTransform> transform;
    vtkNew<vtkTransformFilter> transformFilter;
    vtkNew<vtkAppendFilter> appendFilter;
    vtkNew<vtkUnstructuredGrid> unstructuredGrid;
    vtkNew<vtkXMLUnstructuredGridWriter> writer2;

    cylinder->SetResolution(33);
    cylinder->SetRadius(model->prms.IndDiameter/2.f);
    cylinder->SetHeight(model->prms.GridZ * model->prms.cellsize);

    double indenter_x = model->prms.indenter_x;
    double indenter_y = model->prms.indenter_y;
    double indenter_z = model->prms.GridZ * model->prms.cellsize/2;
    cylinder->SetCenter(indenter_x, indenter_z, -indenter_y);
    cylinder->Update();

    transform->RotateX(90);
    transformFilter->SetTransform(transform);
    transformFilter->SetInputConnection(cylinder->GetOutputPort());
    transformFilter->Update();

    appendFilter->SetInputConnection(transformFilter->GetOutputPort());
    appendFilter->Update();

    unstructuredGrid->ShallowCopy(appendFilter->GetOutput());

    // Write the unstructured grid.
    writer2->SetFileName(savePath.c_str());
    writer2->SetInputData(unstructuredGrid);
    writer2->Write();
    }

void icy::SnapshotManager::WriteIndenterForceCSV()
{
    spdlog::info("icy::SnapshotManager::WriteIndenterForceCSV()");

    std::string fullFilePath = model->outputDirectory + "/indenter_force.csv";
    std::ofstream ofs(fullFilePath, std::ofstream::out | std::ofstream::trunc);
    ofs << "fx,fy,fz,F_total\n";
    for(int i=0;i<model->gpu.indenter_force_history.size();i++)
    {
        Vector3r &v = model->gpu.indenter_force_history[i];
        ofs << v[0] << ',' << v[1] << ',' << v[2] << ',' << v.norm() << '\n';
    }
    ofs.close();
}


std::vector<std::array<float, 3>> icy::SnapshotManager::GenerateBlock(float bx, float by, float bz, int n)
{
    constexpr float magic_constant = 0.58;
    const float bvol = bx*by*bz;
    const float kRadius = cbrt(magic_constant*bvol/n);

    const std::array<float, 3>kXMin{0, 0, 0};
    const std::array<float, 3>kXMax{bx, by, bz};
    std::vector<std::array<float, 3>> prresult = thinks::PoissonDiskSampling(kRadius, kXMin, kXMax);
    return prresult;
}



/*
attributes save/load
//    hsize_t att_dim = 1;
//    H5::DataSpace att_dspace(1, &att_dim);
//    H5::Attribute att = dataset_points.createAttribute("full_data", H5::PredType::NATIVE_INT,att_dspace);
//    att.write(H5::PredType::NATIVE_INT, &full_data);


//    H5::Attribute att = dataset_points.openAttribute("full_data");
//    int full_data;
//    att.read(H5::PredType::NATIVE_INT, &full_data);

*/
