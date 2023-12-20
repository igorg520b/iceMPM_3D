#include "snapshotmanager.h"
#include "model_3d.h"

#include <spdlog/spdlog.h>
#include <H5Cpp.h>

#include <filesystem>
#include <string>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <cstdio>

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
    H5::DSetCreatPropList proplist;
    proplist.setChunk(1, &chunk_dims);
    proplist.setDeflate(5);
    H5::DataSet dataset_points = file.createDataSet("Points", H5::PredType::NATIVE_DOUBLE, dataspace_points, proplist);
    dataset_points.write(model->gpu.tmp_transfer_buffer, H5::PredType::NATIVE_DOUBLE);

/*
    hsize_t dims_indneter_force = icy::SimParams3D::indenter_array_size;
    H5::DataSpace dataspace_indneter_force(1, &dims_indneter_force);
    H5::DataSet dataset_indneter_force = file.createDataSet("Indenter_Force", H5::PredType::NATIVE_DOUBLE, dataspace_indneter_force);
    dataset_indneter_force.write(model->gpu.host_side_indenter_force_accumulator, H5::PredType::NATIVE_DOUBLE);
*/

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
    model->prms.ParticleViewSize = tmp_params.ParticleViewSize;
    model->prms.SphereViewSize = tmp_params.SphereViewSize;

    // read point data
    H5::DataSet dataset_points = file.openDataSet("Points");
    dataset_points.read(model->gpu.tmp_transfer_buffer,H5::PredType::NATIVE_DOUBLE);

    /*
    H5::DataSet dataset_indneter_force = file.openDataSet("Indenter_Force");
    dataset_indneter_force.read(model->gpu.host_side_indenter_force_accumulator, H5::PredType::NATIVE_DOUBLE);
      */

    //model->gpu.transfer_ponts_to_host_finalize(model->points);
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

    indenter_force_buffer.resize(icy::SimParams3D::indenter_array_size);

    last_refresh_frame.resize(n);
    previous_frame_exists = false;
}

void icy::SnapshotManager::SaveFrame()
{
    spdlog::info("icy::SnapshotManager::SaveFrame()");
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
            if(prediction_error.norm() > visual_threshold)
            {
                last_refresh_frame[i] = current_frame_number;
                p_p = p_c;
                saved_frame.push_back(p_c);
            }
        }
        spdlog::info("saving difference; saved_frame.size() = {}",saved_frame.size());
    }

    // indenter buffer
    for(int i=0;i<icy::SimParams3D::indenter_array_size;i++)
        indenter_force_buffer[i] = (float) model->gpu.host_side_indenter_force_accumulator[i];

    char fileName[20];
    snprintf(fileName, sizeof(fileName), "v%05d.h5", current_frame_number);
    std::string savePath = model->outputDirectory + "/" + fileName;
    spdlog::info("saving visual frame {} to file {}", current_frame_number, savePath);

    // ensure that directory exists
    std::filesystem::path outputDirectory(model->outputDirectory);
    if(!std::filesystem::is_directory(outputDirectory) || !std::filesystem::exists(outputDirectory))
        std::filesystem::create_directory(outputDirectory);

    // save!
    H5::H5File file(savePath, H5F_ACC_TRUNC);

    hsize_t dims_params = sizeof(icy::SimParams3D);
    H5::DataSpace dataspace_params(1,&dims_params);
    H5::DataSet dataset_params = file.createDataSet("Params", H5::PredType::NATIVE_B8, dataspace_params);
    dataset_params.write(&model->prms, H5::PredType::NATIVE_B8);

    hsize_t chunk_dims_indenter = 10000;
    H5::DSetCreatPropList proplist2;
    proplist2.setChunk(1, &chunk_dims_indenter);
    proplist2.setDeflate(5);
    hsize_t dims_indneter_force = icy::SimParams3D::indenter_array_size;
    H5::DataSpace dataspace_indneter_force(1, &dims_indneter_force);
    H5::DataSet dataset_indneter_force = file.createDataSet("Indenter_Force", H5::PredType::NATIVE_FLOAT, dataspace_indneter_force, proplist2);
    dataset_indneter_force.write(indenter_force_buffer.data(), H5::PredType::NATIVE_FLOAT);


    hsize_t dims_points = sizeof(VisualPoint)*saved_frame.size();
    hsize_t dims_unlimited = H5S_UNLIMITED;
    H5::DataSpace dataspace_points(1, &dims_points, &dims_unlimited);
    hsize_t chunk_dims = sizeof(VisualPoint)*1024;
    H5::DSetCreatPropList proplist;
    proplist.setChunk(1, &chunk_dims);
    proplist.setDeflate(5);
    H5::DataSet dataset_points = file.createDataSet("Points", H5::PredType::NATIVE_B8, dataspace_points, proplist);
    dataset_points.write(saved_frame.data(), H5::PredType::NATIVE_B8);

    file.close();
    spdlog::info("saved_frame.size {}; {}", saved_frame.size(), fileName);
}


void icy::SnapshotManager::ReadFirstFrame(std::string directory)
{
    path = directory;
    char fileName[20];
    int current_frame_number = 0;
    snprintf(fileName, sizeof(fileName), "v%05d.h5", current_frame_number);
    std::string filePath = path + "/" + fileName;
    spdlog::info("reading visual frame {} to file {}", current_frame_number, filePath);

    if(!std::filesystem::exists(filePath))
    {
        spdlog::critical("file {} does not exist",filePath);
        return;
    }

    H5::H5File file(filePath, H5F_ACC_RDONLY);

    spdlog::info("reading Params");
    // read params
    H5::DataSet dataset_params = file.openDataSet("Params");
    hsize_t dims_params = 0;
    dataset_params.getSpace().getSimpleExtentDims(&dims_params, NULL);
    if(dims_params != sizeof(icy::SimParams3D)) throw std::runtime_error("SimParams3D size mismatch");
    dataset_params.read(&model->prms, H5::PredType::NATIVE_B8);
    model->gpu.cuda_allocate_arrays(model->prms.nGridPitch, model->prms.nPtsPitch);

    // allocate arrays for snapshot
    AllocateMemoryForFrames();

    // read indenter data
    spdlog::info("reading indenter data");
    H5::DataSet dataset_indenter = file.openDataSet("Indenter_Force");
    dataset_indenter.read(indenter_force_buffer.data(), H5::PredType::NATIVE_FLOAT);
    for(int i=0;i<icy::SimParams3D::indenter_array_size;i++)
    model->gpu.host_side_indenter_force_accumulator[i] = indenter_force_buffer[i];

    // read points
    spdlog::info("reading Points");
    H5::DataSet dataset_points = file.openDataSet("Points");
    hsize_t dims_points = 0;
    dataset_points.getSpace().getSimpleExtentDims(&dims_points, NULL);
    int nPoints = dims_points/sizeof(VisualPoint);
    current_frame.resize(nPoints);
    dataset_points.read(current_frame.data(), H5::PredType::NATIVE_B8);
    file.close();

    spdlog::info("transferring points to host buffer");
    for(int i=0;i<current_frame.size();i++)
    {
        VisualPoint &vp = current_frame[i];
        int pt_idx = vp.id;
        if(pt_idx >= model->prms.nPtsPitch)
        {
            spdlog::critical("pt_idx {} out of bounds {}", pt_idx, model->prms.nPtsPitch);
            throw std::out_of_range("read error");
        }
        icy::Point3D::setPos_Q_Jpinv(vp.pos(), vp.Jp_inv,
                                     model->gpu.tmp_transfer_buffer, model->prms.nPtsPitch, pt_idx);
    }
    spdlog::info("icy::SnapshotManager::ReadFirstFrame done");
}


bool icy::SnapshotManager::ReadNextFrame()
{
    // this is different from ReadFirstFrame - the updates are iterative
    int next_frame = model->prms.AnimationFrameNumber()+1;

    char fileName[20];
    snprintf(fileName, sizeof(fileName), "v%05d.h5", next_frame);
    std::string filePath = path + "/" + fileName;
    spdlog::info("reading file {}", filePath);

    if(!std::filesystem::exists(filePath)) return false;

    H5::H5File file(filePath, H5F_ACC_RDONLY);

    // read params
    H5::DataSet dataset_params = file.openDataSet("Params");
    hsize_t dims_params = 0;
    dataset_params.getSpace().getSimpleExtentDims(&dims_params, NULL);
    if(dims_params != sizeof(icy::SimParams3D)) throw std::runtime_error("SimParams3D size mismatch");
    dataset_params.read(&model->prms, H5::PredType::NATIVE_B8);

    // read indenter data
    H5::DataSet dataset_indenter = file.openDataSet("Indenter_Force");
    dataset_indenter.read(indenter_force_buffer.data(), H5::PredType::NATIVE_FLOAT);
    for(int i=0;i<icy::SimParams3D::indenter_array_size;i++)
        model->gpu.host_side_indenter_force_accumulator[i] = indenter_force_buffer[i];

    H5::DataSet dataset_points = file.openDataSet("Points");
    hsize_t dims_points = 0;
    dataset_points.getSpace().getSimpleExtentDims(&dims_points, NULL);
    int nPoints = dims_points/sizeof(VisualPoint);
    saved_frame.resize(nPoints);
    dataset_points.read(saved_frame.data(), H5::PredType::NATIVE_B8);
    file.close();

    // advance "current_frame" one step forward

    for(int i=0; i<model->prms.nPts;i++)
    {
        VisualPoint &vp = current_frame[i];
        Eigen::Vector3f updated_pos = vp.pos() + vp.vel()*model->prms.InitialTimeStep;
        for(int j=0;j<3;j++) vp.p[j] = updated_pos[j];
    }

    // update select points
    for(int i=0; i<saved_frame.size(); i++)
    {
        VisualPoint &vp = saved_frame[i];
        current_frame[vp.id] = vp;
    }

    // transfer
    for(int i=0;i<current_frame.size();i++)
    {
        VisualPoint &vp = current_frame[i];
        icy::Point3D::setPos_Q_Jpinv(vp.pos(), vp.Jp_inv,
                                     model->gpu.tmp_transfer_buffer, model->prms.nPtsPitch, i);
    }

    return true;
}




/*
void icy::SnapshotManager::ReadDirectory(std::string directoryPath)
{
    path = directoryPath;
    // set last_file_index
    for (const auto & entry : std::filesystem::directory_iterator(directoryPath))
    {
        std::string fileName = entry.path();
        std::string extension = fileName.substr(fileName.length()-3,3);
        if(extension != ".h5") continue;
        std::string numbers = fileName.substr(fileName.length()-8,5);
        int idx = std::stoi(numbers);
        if(idx > last_file_index) last_file_index = idx;

//        std::cout << fileName << ", " << extension << ", " << numbers << std::endl;
    }
    spdlog::info("directory scanned; last_file_index is {}", last_file_index);

}

*/


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
