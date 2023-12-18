#ifndef SNAPSHOTMANAGER_H
#define SNAPSHOTMANAGER_H

#include <string>

namespace icy {class SnapshotManager; class Model3D;}


class icy::SnapshotManager
{
public:
    icy::Model3D *model;

    int last_file_index = 0;
    std::string path;
    void SaveFullSnapshot(std::string fileName);
    void ReadFullSnapshot(std::string fileName);

//    void ScanDirectory(std::string directoryPath);
};

#endif // SNAPSHOTWRITER_H
