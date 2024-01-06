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
    constexpr int save_full_snapshot_every = 50;
    std::string dir = "full_snapshots";
    std::filesystem::path od(dir);
    if(!std::filesystem::is_directory(od) || !std::filesystem::exists(od)) std::filesystem::create_directory(od);

    std::thread snapshot_thread;
    std::atomic<bool> request_full_snapshot = false;
    std::atomic<bool> request_terminate = false;

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

            if(frame%save_full_snapshot_every == 0 || request_full_snapshot)
            {
                request_full_snapshot = false;
                char fileName[20];
                snprintf(fileName, sizeof(fileName), "s%05d.h5", frame);
                std::string savePath = dir + "/" + fileName;
                snapshot.SaveFullSnapshot(savePath);
            }

            model.UnlockCycleMutex();
            spdlog::info("callback {} done", frame);
        });
    };

    std::thread simulation_thread([&](){
        bool result;
        do
        {
            result = model.Step();
            std::cout << "(s)save, (q)save and quit\n";
        } while(result && !request_terminate);
        spdlog::critical("simulation ended");
        request_terminate = true;
    });

    do
    {
        std::string user_input;
        std::cin >> user_input;

        if(user_input[0]=='s')
        {
            request_full_snapshot = true;
            spdlog::critical("requested to save a full snapshot");
        }
        else if(user_input[0]=='q'){
            request_terminate = true;
            request_full_snapshot = true;
            spdlog::critical("requested to save the snapshot and terminate");
        }
    } while(!request_terminate);

    simulation_thread.join();
    model.gpu.synchronize();
    snapshot_thread.join();

    std::cout << "cm done\n";
}


