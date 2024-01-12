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
    spdlog::info("writing full snapshot {}",fileName);

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
    spdlog::info("SnapshotManager: reading full snapshot {}", fileName);

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


void icy::SnapshotManager::SaveFrame()
{
    spdlog::info("icy::SnapshotManager::SaveFrame()");
    std::filesystem::path odp(model->outputDirectory);
    if(!std::filesystem::is_directory(odp) || !std::filesystem::exists(odp)) std::filesystem::create_directory(odp);

    if(export_h5) ExportPointsAsH5();
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
        buffer[k][0] += (model->prms.BlockOffsetX+5)*h;
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

void icy::SnapshotManager::PopulateVisualPoint(VisualPoint &vp, int idx)
{
    vp.id = idx;
    vp.Jp_inv = (float)icy::Point3D::getJp_inv(model->gpu.tmp_transfer_buffer, model->prms.nPtsPitch, idx);
    Vector3r pos = icy::Point3D::getPos(model->gpu.tmp_transfer_buffer, model->prms.nPtsPitch, idx);
    Vector3r vel = icy::Point3D::getVelocity(model->gpu.tmp_transfer_buffer, model->prms.nPtsPitch, idx);
    for(int j=0;j<3;j++)
    {
        vp.p[j] = (float)pos[j];
        vp.v[j] = (float)vel[j];
    }
}

void icy::SnapshotManager::AllocateMemoryForFrames()
{
    int n = model->prms.nPts;
    current_frame.resize(n);
    saved_frame.reserve(n);
    last_refresh_frame.resize(n);
    previous_frame_exists = false;
}


void icy::SnapshotManager::ExportPointsAsH5()
{
    spdlog::info("icy::SnapshotManager::ExportPointsAsH5()");

    // populate saved_frame
    const int current_frame_number = model->prms.AnimationFrameNumber();

    if(!previous_frame_exists)
    {
        for(int i=0;i<model->prms.nPts;i++) PopulateVisualPoint(current_frame[i], i);
        // saved_frame is a full copy of the current_frame
        saved_frame = current_frame;
        std::fill(last_refresh_frame.begin(), last_refresh_frame.end(), current_frame_number);
        previous_frame_exists = true;
    }
    else
    {
        const float visual_threshold = 1e-3;
        const float Jp_inv_threshold = 1e-3;
        float dt = (float)model->prms.InitialTimeStep;
        saved_frame.clear();
        for(int i=0;i<model->prms.nPts;i++)
        {
            VisualPoint p_c;
            PopulateVisualPoint(p_c, i);
            VisualPoint &p_p = current_frame[i];
            int previous_update = last_refresh_frame[i];
            int elapsed_frames = current_frame_number - previous_update;
            Eigen::Vector3f predicted_position = p_p.pos() + p_p.vel()*(dt*elapsed_frames);
            Eigen::Vector3f prediction_error = p_c.pos() - predicted_position;
            float Jp_diff = abs(p_c.Jp_inv - p_p.Jp_inv);
            bool writeJp = (p_p.Jp_inv == 1.0 && p_c.Jp_inv != 1.0);
            if(prediction_error.norm() > visual_threshold || Jp_diff > Jp_inv_threshold || writeJp)
            {
                last_refresh_frame[i] = current_frame_number;
                p_p = p_c;
                saved_frame.push_back(p_c);
            }
        }
    }
    spdlog::info("saved_frame size is {}",saved_frame.size());

    char fileName[20];
    snprintf(fileName, sizeof(fileName), "v%05d.h5", current_frame_number);
    std::string saveDir = model->outputDirectory + "/" + dir_points_h5;
    std::string fullFilePath = saveDir + "/" + fileName;
    spdlog::info("saving frame {} to file {}", current_frame_number, fullFilePath);

    // ensure that directory exists
    std::filesystem::path od(saveDir);
    if(!std::filesystem::is_directory(od) || !std::filesystem::exists(od)) std::filesystem::create_directory(od);

    // save
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

    // save some additional data as attributes
    hsize_t att_dim = 1;
    H5::DataSpace att_dspace(1, &att_dim);
    H5::Attribute att_indenter_x = dataset_indneter_force.createAttribute("indenter_x", H5::PredType::NATIVE_DOUBLE, att_dspace);
    H5::Attribute att_indenter_y = dataset_indneter_force.createAttribute("indenter_y", H5::PredType::NATIVE_DOUBLE, att_dspace);
    H5::Attribute att_SimulationTime = dataset_indneter_force.createAttribute("SimulationTime", H5::PredType::NATIVE_DOUBLE, att_dspace);
    att_indenter_x.write(H5::PredType::NATIVE_DOUBLE, &model->prms.indenter_x);
    att_indenter_y.write(H5::PredType::NATIVE_DOUBLE, &model->prms.indenter_y);
    att_SimulationTime.write(H5::PredType::NATIVE_DOUBLE, &model->prms.SimulationTime);

    if(current_frame_number == 1)
    {
        H5::Attribute att_GridZ = dataset_indneter_force.createAttribute("GridZ", H5::PredType::NATIVE_INT, att_dspace);
        H5::Attribute att_nPts = dataset_indneter_force.createAttribute("nPts", H5::PredType::NATIVE_INT, att_dspace);
        H5::Attribute att_UpdateEveryNthStep = dataset_indneter_force.createAttribute("UpdateEveryNthStep", H5::PredType::NATIVE_INT, att_dspace);
        H5::Attribute att_n_indenter_subdivisions_angular = dataset_indneter_force.createAttribute("n_indenter_subdivisions_angular", H5::PredType::NATIVE_INT, att_dspace);

        att_GridZ.write(H5::PredType::NATIVE_INT, &model->prms.GridZ);
        att_nPts.write(H5::PredType::NATIVE_INT, &model->prms.nPts);
        att_UpdateEveryNthStep.write(H5::PredType::NATIVE_INT, &model->prms.UpdateEveryNthStep);
        att_n_indenter_subdivisions_angular.write(H5::PredType::NATIVE_INT, &model->prms.n_indenter_subdivisions_angular);

        H5::Attribute att_cellsize = dataset_indneter_force.createAttribute("cellsize", H5::PredType::NATIVE_DOUBLE, att_dspace);
        H5::Attribute att_IceBlockDimZ = dataset_indneter_force.createAttribute("IceBlockDimZ", H5::PredType::NATIVE_DOUBLE, att_dspace);
        H5::Attribute att_IndDiameter = dataset_indneter_force.createAttribute("IndDiameter", H5::PredType::NATIVE_DOUBLE, att_dspace);
        H5::Attribute att_InitialTimeStep = dataset_indneter_force.createAttribute("InitialTimeStep", H5::PredType::NATIVE_DOUBLE, att_dspace);

        att_cellsize.write(H5::PredType::NATIVE_DOUBLE, &model->prms.cellsize);
        att_IceBlockDimZ.write(H5::PredType::NATIVE_DOUBLE, &model->prms.IceBlockDimZ);
        att_IndDiameter.write(H5::PredType::NATIVE_DOUBLE, &model->prms.IndDiameter);
        att_InitialTimeStep.write(H5::PredType::NATIVE_DOUBLE, &model->prms.InitialTimeStep);
    }

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

