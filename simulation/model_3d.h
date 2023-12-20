#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <algorithm>
#include <chrono>
#include <unordered_set>
#include <utility>
#include <cmath>
#include <random>
#include <mutex>
#include <iostream>
#include <string>

#include "parameters_sim_3d.h"
#include "point_3d.h"
#include "poisson_disk_sampling.h"
#include "gpu_implementation4.h"

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/LU>



namespace icy { class Model3D; }

class icy::Model3D
{
public:
    Model3D();
    void Reset();
    void Prepare();        // invoked once, at simulation start
    bool Step();           // either invoked by Worker or via GUI
    void RequestAbort() {abortRequested = true;}   // asynchronous stop

    void UnlockCycleMutex();

    icy::SimParams3D prms;
    GPU_Implementation4 gpu;
    float compute_time_per_cycle;

    std::mutex processing_current_cycle_data; // locked until the current cycle results' are copied to host and processed
    std::string outputDirectory;

private:
    void ResetGrid();
    void P2G();
    void UpdateNodes();
    void G2P();

    bool abortRequested;
};

#endif
