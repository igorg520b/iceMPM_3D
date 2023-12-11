#include "snapshotmanager.h"
#include "model.h"

#include <spdlog/spdlog.h>
#include <H5Cpp.h>
#include <filesystem>
#include <string>
#include <sstream>
#include <iomanip>
#include <fstream>

/*

void icy::SnapshotManager::SaveSnapshot(std::string fileName)
{
    spdlog::info("writing snapshot {}",fileName);

    H5::H5File file(fileName, H5F_ACC_TRUNC);

    hsize_t dims_params = sizeof(icy::SimParams);
    H5::DataSpace dataspace_params(1,&dims_params);
    H5::DataSet dataset_params = file.createDataSet("Params", H5::PredType::NATIVE_B8, dataspace_params);
    dataset_params.write(&model->prms, H5::PredType::NATIVE_B8);

    int fullDataArrays = icy::SimParams::nPtsArrays;
    hsize_t nPtsPitched = model->prms.nPtsPitch;
    hsize_t dims_points = nPtsPitched*fullDataArrays;

    H5::DataSpace dataspace_points(1, &dims_points);
    H5::DataSet dataset_points = file.createDataSet("Points", H5::PredType::NATIVE_DOUBLE, dataspace_points);
    dataset_points.write(model->gpu.tmp_transfer_buffer, H5::PredType::NATIVE_DOUBLE);

//    hsize_t att_dim = 1;
//    H5::DataSpace att_dspace(1, &att_dim);
//    H5::Attribute att = dataset_points.createAttribute("full_data", H5::PredType::NATIVE_INT,att_dspace);
//    att.write(H5::PredType::NATIVE_INT, &full_data);


    file.close();
    spdlog::info("SaveSnapshot done {}", fileName);
}

int icy::SnapshotManager::ReadSnapshot(std::string fileName)
{
    if(!std::filesystem::exists(fileName)) return -1;

    std::string numbers = fileName.substr(fileName.length()-8,5);
    int idx = std::stoi(numbers);
    spdlog::info("reading snapshot {}", idx);

    H5::H5File file(fileName, H5F_ACC_RDONLY);

    // read and process SimParams
    H5::DataSet dataset_params = file.openDataSet("Params");
    hsize_t dims_params = 0;
    dataset_params.getSpace().getSimpleExtentDims(&dims_params, NULL);
    if(dims_params != sizeof(icy::SimParams)) throw std::runtime_error("ReadSnapshot: SimParams size mismatch");

    icy::SimParams tmp_params;
    dataset_params.read(&tmp_params, H5::PredType::NATIVE_B8);

    if(tmp_params.nGridPitch != model->prms.nGridPitch || tmp_params.nPtsPitch != model->prms.nPtsPitch)
        model->gpu.cuda_allocate_arrays(tmp_params.nGridPitch,tmp_params.nPtsPitch);
    real ParticleViewSize = model->prms.ParticleViewSize;
    model->prms = tmp_params;
    model->prms.ParticleViewSize = ParticleViewSize;

    // read point data
    H5::DataSet dataset_points = file.openDataSet("Points");
//    H5::Attribute att = dataset_points.openAttribute("full_data");
//    int full_data;
//    att.read(H5::PredType::NATIVE_INT, &full_data);

    dataset_points.read(model->gpu.tmp_transfer_buffer,H5::PredType::NATIVE_DOUBLE);

    model->gpu.transfer_ponts_to_host_finalize(model->points);
    file.close();
    return idx;
}

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

void icy::SnapshotManager::DumpPointData(int pt_idx)
{
    std::vector<double> p0, p, q, q_limit, Jp, c;

    for(int i=1; i<=last_file_index; i++)
    {
        std::stringstream ss;
        ss << std::setw(5) << std::setfill('0') << i;
        std::string s = ss.str();
        std::string fileName = this->path + "/" + s + ".h5";
        std::cout << "reading " << fileName << std::endl;


        if(!std::filesystem::exists(fileName)) throw std::runtime_error("saved file not found");
        H5::H5File file(fileName, H5F_ACC_RDONLY);
        H5::DataSet dataset_points = file.openDataSet("Points");

        //DATASPACE
        H5::DataSpace space1 = dataset_points.getSpace();
        double tmp[6];
        hsize_t dimsm[1] {6};
        H5::DataSpace memspace(1, dimsm);

        hsize_t count[1] {1};
        hsize_t offset[1] {icy::SimParams::idx_p0 * model->prms.nPtsPitch + pt_idx};
        space1.selectHyperslab(H5S_SELECT_SET, count, offset);

        hsize_t moffset[1] {0};
        memspace.selectHyperslab(H5S_SELECT_SET, count, moffset);
        dataset_points.read(&tmp, H5::PredType::NATIVE_DOUBLE, memspace, space1);



        offset[0] = icy::SimParams::idx_p * model->prms.nPtsPitch + pt_idx;
        space1.selectHyperslab(H5S_SELECT_SET, count, offset);
        moffset[0] = 1;
        memspace.selectHyperslab(H5S_SELECT_SET, count, moffset);
        dataset_points.read(&tmp, H5::PredType::NATIVE_DOUBLE, memspace, space1);

        offset[0] = icy::SimParams::idx_q * model->prms.nPtsPitch + pt_idx;
        space1.selectHyperslab(H5S_SELECT_SET, count, offset);
        moffset[0] = 2;
        memspace.selectHyperslab(H5S_SELECT_SET, count, moffset);
        dataset_points.read(&tmp, H5::PredType::NATIVE_DOUBLE, memspace, space1);

        offset[0] = icy::SimParams::idx_Jp * model->prms.nPtsPitch + pt_idx;
        space1.selectHyperslab(H5S_SELECT_SET, count, offset);
        moffset[0] = 3;
        memspace.selectHyperslab(H5S_SELECT_SET, count, moffset);
        dataset_points.read(&tmp, H5::PredType::NATIVE_DOUBLE, memspace, space1);

        offset[0] = icy::SimParams::idx_case * model->prms.nPtsPitch + pt_idx;
        space1.selectHyperslab(H5S_SELECT_SET, count, offset);
        moffset[0] = 4;
        memspace.selectHyperslab(H5S_SELECT_SET, count, moffset);
        dataset_points.read(&tmp, H5::PredType::NATIVE_DOUBLE, memspace, space1);

        offset[0] = icy::SimParams::idx_q_limit * model->prms.nPtsPitch + pt_idx;
        space1.selectHyperslab(H5S_SELECT_SET, count, offset);
        moffset[0] = 5;
        memspace.selectHyperslab(H5S_SELECT_SET, count, moffset);
        dataset_points.read(&tmp, H5::PredType::NATIVE_DOUBLE, memspace, space1);


        p0.push_back(tmp[0]);
        p.push_back(tmp[1]);
        q.push_back(tmp[2]);
        Jp.push_back(tmp[3]);
        c.push_back(tmp[4]);
        q_limit.push_back((tmp[5]));
    }

    std::string outputFile = std::to_string(pt_idx) + ".csv";
    std::ofstream ofs(outputFile, std::ofstream::out);
    ofs << "p0, pc, p, q, q_limit, Jp, c\n";
    for(int i=0;i<last_file_index;i++)
    {
        double pc = p0[i]*(1-model->prms.NACC_beta)*0.5;
        ofs << p0[i] << ',' << pc << ',' << p[i] << ',' << q[i] << ',' << q_limit[i] << ',' << Jp[i] << ',' << c[i];
        ofs << '\n';
    }
    ofs.close();


}
*/
