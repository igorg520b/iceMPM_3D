#include <spdlog/spdlog.h>
#include <H5Cpp.h>

#include <vector>
#include <algorithm>
#include <array>
#include <cmath>

#include "snapshotmanager.h"

void save_hdf5(std::vector<std::array<float, 3>> &pts, std::string fileName)
{
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
}

void generate_block(float bx, float by, float bz, int n, std::string fileName)
{
    spdlog::info("generate_points: {}, {}", n, fileName);
    std::vector<std::array<float, 3>> pts = icy::SnapshotManager::GenerateBlock(bx, by, bz, n);
    save_hdf5(pts, fileName);
    spdlog::info("generating and saving done");
}

void generate_cone(float diameter, float top, float angle, float height, int n, std::string fileName)
{
    spdlog::info("generate_cone; diam {}; top {}; angle {}; height {}; n {}", diameter, top, angle, height, n);
    float vBlock = diameter*diameter*height;
    float tan_alpha = tan(M_PI*angle/180.);
    float R = diameter/2;
    float r = top/2;
    float ht = (R-r)*tan_alpha;
    float htt = R*tan_alpha;
    float &H = height;
    float hb = H-ht;

    float vResult = M_PI*R*R*(hb+htt/3) - M_PI*r*r*r*tan_alpha/3;
    spdlog::info("generating block {} x {} x {}", diameter, height, diameter);
    std::vector<std::array<float, 3>> pts = icy::SnapshotManager::GenerateBlock(diameter, height, diameter, n*vBlock/vResult);
    spdlog::info("points generated {}",pts.size());

    pts.erase(
    std::remove_if(pts.begin(),pts.end(),[hb,R,tan_alpha](std::array<float,3> &p){
        float ph = p[1];
        float pr = sqrt((p[0]-R)*(p[0]-R)+(p[2]-R)*(p[2]-R));
        if(ph < hb) return (pr > R);
        float h_limit = (R-pr)*tan_alpha;
        return ((ph-hb) > h_limit);
        }),
        pts.end());

    spdlog::info("points after filtering {}", pts.size());

    save_hdf5(pts, fileName);
    spdlog::info("cone generating and saving done");
}

