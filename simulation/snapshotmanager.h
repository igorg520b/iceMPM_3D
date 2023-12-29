#ifndef SNAPSHOTMANAGER_H
#define SNAPSHOTMANAGER_H

#include <string>
#include <vector>
#include <Eigen/Core>

namespace icy {class SnapshotManager; class Model3D;}


class icy::SnapshotManager
{
public:
    icy::Model3D *model;
    bool previous_frame_exists = false;     // false when starting/restarting the simulation
    std::string path;
    bool export_vtp, export_h5, export_force;

    void ReadRawPoints(std::string fileName);
    void GeneratePoints();

    void SaveFullSnapshot(std::string fileName);
    void ReadFullSnapshot(std::string fileName);

    void AllocateMemoryForFrames();
    void SaveFrame();   // export in H5, VTP, VTU, CSV

//    void ReadFirstFrame(std::string directory);
//    bool ReadNextFrame();  // false if reached the end

private:
    const std::string dir_vtp = "output_vtp";
    const std::string dir_indenter = "output_indenter";
    const std::string dir_points_h5 = "output_h5";

    struct VisualPoint
    {
        int id;
        float p[3], v[3];
        float Jp_inv;

        Eigen::Vector3f pos() {return Eigen::Vector3f(p[0],p[1],p[2]);}
        Eigen::Vector3f vel() {return Eigen::Vector3f(v[0],v[1],v[2]);}
    };
    std::vector<VisualPoint> current_frame, previous_frame, saved_frame;
    std::vector<int> last_refresh_frame;

    void ExportPointsAsH5();
    void ExportPointsAsVTP();
    void ExportIndenterAsVTU();
    void WriteIndenterForceCSV();
};

#endif // SNAPSHOTWRITER_H
