#include <iostream>
#include <functional>
#include <string>
#include <filesystem>
#include <atomic>
#include <thread>

#include <spdlog/spdlog.h>

#include "model_3d.h"
#include "parameters_sim_3d.h"
#include "snapshotmanager.h"


void run_simulation(icy::Model3D &model, icy::SnapshotManager &snapshot);


void start_simulation_from_json(std::string jsonFile, bool export_vtp)
{

    spdlog::info("starting simulation from JSON configuration file {}", jsonFile);
    icy::Model3D model;
    icy::SnapshotManager snapshot;
    snapshot.export_vtp = export_vtp;
    snapshot.model = &model;
    model.gpu.initialize();
    std::string rawPointsFile = model.prms.ParseFile(jsonFile);
    snapshot.ReadRawPoints(rawPointsFile);
    run_simulation(model, snapshot);
}


void resume_simulation_from_snapshot(std::string snapshotFile, bool export_vtp)
{
    spdlog::info("resuming simulation from full snapshot file {}", snapshotFile);
    icy::Model3D model;
    icy::SnapshotManager snapshot;
    snapshot.export_vtp = export_vtp;
    snapshot.model = &model;
    model.gpu.initialize();
    snapshot.ReadFullSnapshot(snapshotFile);
    run_simulation(model, snapshot);
}

void run_simulation(icy::Model3D &model, icy::SnapshotManager &snapshot)
{
    constexpr int save_full_snapshot_every = 100;
    std::string dir = "full_snapshots";
    std::filesystem::path od(dir);
    if(!std::filesystem::is_directory(od) || !std::filesystem::exists(od)) std::filesystem::create_directory(od);

    std::thread snapshot_thread;

    snapshot.AllocateMemoryForFrames();
    snapshot.export_force = true;
    snapshot.export_h5 = true;

    // what to do once the data is available
    model.gpu.transfer_completion_callback = [&](){
        if(snapshot_thread.joinable()) snapshot_thread.join();
        snapshot_thread = std::thread([&](){
            int frame = model.prms.AnimationFrameNumber();
            spdlog::info("data transfer callback {}", frame);
            snapshot.SaveFrame();

            if(frame%save_full_snapshot_every == 0)
            {
                char fileName[20];
                snprintf(fileName, sizeof(fileName), "s%05d.h5", frame);
                std::string savePath = dir + "/" + fileName;
                snapshot.SaveFullSnapshot(savePath);
            }

            model.UnlockCycleMutex();
            spdlog::info("callback {} done", frame);
        });
    };

    std::thread t([&](){
        bool result;
        do
        {
            result = model.Step();
        } while(result);
    });

    t.join();
    model.gpu.synchronize();
    snapshot_thread.join();

    std::cout << "cm done\n";
}


