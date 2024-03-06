#ifndef SNAPSHOTMANAGER_H
#define SNAPSHOTMANAGER_H

#include <string>
#include <vector>
#include <array>
#include <Eigen/Core>
#include <H5Cpp.h>

#include "parameters_sim_3d.h"

namespace icy {class SnapshotManager; class Model3D;}


class icy::SnapshotManager
{
public:
    icy::Model3D *model;
    bool previous_frame_exists = false;     // false when starting/restarting the simulation
    std::string path;
    bool export_h5_raw = false;

    void ReadRawPoints(std::string fileName);

    void SaveFullSnapshot(std::string fileName);
    void ReadFullSnapshot(std::string fileName);

    void AllocateMemoryForFrames();
    void SaveFrame();   // export in H5, VTP, VTU, CSV

    struct VisualPoint
    {
        int id;
        float p[3], v[3];
        float Jp_inv;

        Eigen::Vector3f pos() {return Eigen::Vector3f(p[0],p[1],p[2]);}
        Eigen::Vector3f vel() {return Eigen::Vector3f(v[0],v[1],v[2]);}
    };

//    static void H5Raw_to_Paraview(std::string directory);

private:
    const std::string dir_vtp = "output_vtp";
    const std::string dir_indenter = "output_indenter";
    const std::string dir_points_h5 = "output_h5";
    const std::string dir_h5_raw = "raw_h5";

    std::vector<VisualPoint> current_frame, saved_frame;
    std::vector<int> last_refresh_frame;

    void ExportPointsAsH5();
    void ExportPointsAsH5_Raw();
    void SaveParametersAsAttributes(H5::DataSet &dataset);
    static void ReadParametersAsAttributes(icy::SimParams3D &prms, const H5::DataSet &dataset);

    void PopulateVisualPoint(VisualPoint &vp, int idx);

    // functions that export data in bgeo / paraview formats
    static void export_bgeo_f(int frame, std::vector<icy::SnapshotManager::VisualPoint> &current_frame);
};

#endif // SNAPSHOTWRITER_H
