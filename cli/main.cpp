#include <iostream>
#include <functional>
#include <string>
#include <filesystem>
#include <atomic>
#include <thread>
#include <chrono>

#include <cxxopts.hpp>
#include <spdlog/spdlog.h>

#include "model.h"
#include "snapshotwriter.h"

std::atomic<bool> stop = false;
std::string snapshot_directory = "cm_snapshots";
std::thread snapshot_thread;

icy::Model model;
icy::SnapshotWriter snapshot;


int main(int argc, char** argv)
{
    // parse options
    cxxopts::Options options("Ice MPM", "CLI version of MPM simulation");

    options.add_options()
        ("file", "Configuration file", cxxopts::value<std::string>())
        ;
    options.parse_positional({"file"});

    auto option_parse_result = options.parse(argc, argv);

    if(option_parse_result.count("file"))
    {
        std::string params_file = option_parse_result["file"].as<std::string>();
        snapshot_directory = model.prms.ParseFile(params_file);
    }

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

    return 0;
}
