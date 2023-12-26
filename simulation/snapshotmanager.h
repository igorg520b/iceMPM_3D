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

    int last_file_index = 0;
    bool previous_frame_exists = false;     // false when starting/restarting the simulation
    std::string path;

    void ReadRawPoints(std::string fileName);
    void GeneratePoints();

    void SaveFullSnapshot(std::string fileName);
    void ReadFullSnapshot(std::string fileName);

    void AllocateMemoryForFrames();
    void SaveFrame();

    void ReadFirstFrame(std::string directory);
    bool ReadNextFrame();  // false if reached the end



private:
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
    std::vector<float> indenter_force_buffer;
};

#endif // SNAPSHOTWRITER_H
