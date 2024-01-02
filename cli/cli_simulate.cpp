#include <spdlog/spdlog.h>


void start_simulation_from_json(std::string jsonFile)
{
    spdlog::info("starting simulation from JSON configuration file {}", jsonFile);
}


void resume_simulation_from_snapshot(std::string snapshotFile)
{
    spdlog::info("resuming simulation from full snapshot file {}", snapshotFile);

}



//icy::Model3D model;
//icy::SnapshotManager snapshot;
//std::atomic<bool> stop = false;
//std::thread snapshot_thread;


/*
    // initialize the model
    model.Reset();
    snapshot.model = &model;

    // what to do once the data is available
    model.gpu.transfer_completion_callback = [&](){
        if(snapshot_thread.joinable()) snapshot_thread.join();
        snapshot_thread = std::thread([&](){
            int snapshot_number = model.prms.SimulationStep / model.prms.UpdateEveryNthStep;
            if(stop) { std::cout << "screenshot aborted\n"; return; }
            spdlog::info("completion callback {}", snapshot_number);
            model.FinalizeDataTransfer();
            std::string outputPath = snapshot_directory + "/" + std::to_string(snapshot_number) + ".h5";
            snapshot.SaveSnapshot(outputPath, snapshot_number % 100 == 0);
            model.UnlockCycleMutex();
            spdlog::info("callback {} done", snapshot_number);
        });
    };

    // ensure that the folder exists
    std::filesystem::path outputFolder(snapshot_directory);
    std::filesystem::create_directory(outputFolder);

    std::thread t([&](){
        bool result;
        do
        {
            result = model.Step();
        } while(!stop && result);
    });

    t.join();
    model.gpu.synchronize();
    snapshot_thread.join();

    std::cout << "cm done\n";
*/
