#include "snapshotmanager.h"
#include "model_3d.h"

#include <spdlog/spdlog.h>
#include <H5Cpp.h>
#include <filesystem>
#include <string>
#include <sstream>
#include <iomanip>
#include <fstream>

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
    real ParticleViewSize = model->prms.ParticleViewSize;
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
