#include <spdlog/spdlog.h>
#include <H5Cpp.h>

#include <vector>
#include <array>

#include "snapshotmanager.h"

void generate_points(float bx, float by, float bz, int n, std::string fileName)
{
    spdlog::info("generate_points: {}, {}", n, fileName);
    std::vector<std::array<float, 3>> pts = icy::SnapshotManager::GenerateBlock(bx, by, bz, n);
    H5::H5File file(fileName, H5F_ACC_TRUNC);

    hsize_t dims_points[2] = {pts.size(), 3};
    H5::DataSpace dataspace_points(2, dims_points);

    hsize_t chunk_dims[2] = {1024*1024,3};
    if(chunk_dims[0] > pts.size()) chunk_dims[0] = pts.size()/10;
    if(chunk_dims[0] == 0) chunk_dims[0] = 1;
    spdlog::info("chunk {}; dims_pts {}", chunk_dims[0], dims_points[0]);
    H5::DSetCreatPropList proplist;
    proplist.setChunk(2, chunk_dims);
    proplist.setDeflate(7);
    H5::DataSet dataset_points = file.createDataSet("Points_Raw", H5::PredType::NATIVE_FLOAT, dataspace_points, proplist);
    dataset_points.write(pts.data(), H5::PredType::NATIVE_FLOAT);

    file.close();
    spdlog::info("generating and saving done");
}
