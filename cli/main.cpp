#include <iostream>
#include <functional>
#include <string>
#include <filesystem>
#include <chrono>

#include <cxxopts.hpp>
#include <spdlog/spdlog.h>

#include "model_3d.h"
#include "parameters_sim_3d.h"
#include "snapshotmanager.h"

#include <omp.h>

void start_simulation_from_json(std::string jsonFile, bool export_vtp, bool export_h5_raw);
void resume_simulation_from_snapshot(std::string snapshotFile, bool export_vtp, bool export_h5_raw);
void convert_to_bgeo_vtp(std::string directory, bool vtp, bool bgeo, bool export_indenter);


int main(int argc, char** argv)
{
#pragma omp parallel
    { std::cout << omp_get_thread_num(); }
    std::cout << std::endl;

    // parse options
    cxxopts::Options options("Ice MPM 3D", "CLI version of MPM 3D simulation");

    options.add_options()
        // point generation
        ("g,generate", "Make a set of N points for the simulation starting input", cxxopts::value<int>()->default_value("10000000"))
        ("o,output", "Output file name", cxxopts::value<std::string>()->default_value("raw_10m.h5"))
        ("x,bx", "Length of the block", cxxopts::value<float>()->default_value("2.5"))
        ("y,by", "Height of the block", cxxopts::value<float>()->default_value("1.0"))
        ("z,bz", "Width of the block", cxxopts::value<float>()->default_value("1.5"))

        ("cone", "Generate cone")
        ("diameter", "Diameter of the cone", cxxopts::value<float>()->default_value("0.2688"))
        ("top", "Diameter at the top of the cone", cxxopts::value<float>()->default_value("0.0254"))
        ("angle", "Taper angle of the cone", cxxopts::value<float>()->default_value("21"))
        ("height", "Total height of the sample", cxxopts::value<float>()->default_value("0.1"))


        // simulation output (.H5) conversion to BGEO and/or VTP
        ("c,convert", "Directory where iterative h5 fies are saved", cxxopts::value<std::string>())
        ("p,convert-parallel", "Directory where raw h5 fies are saved", cxxopts::value<std::string>())
        ("b,bgeo", "Export as BGEO")
        ("v,vtp", "Export points as VTP, cylinder as VTU")
        ("i,indenter", "Export indenter data")

        // execution of the simulation
        ("s,simulate", "Input JSON configuration file", cxxopts::value<std::string>())
        ("r,resume", "Input full snapshot (.h5) file", cxxopts::value<std::string>())
        ("w,export-raw", "Export H5 frames without compression")
        ;
    options.parse_positional({"file"});

    auto option_parse_result = options.parse(argc, argv);

    if(option_parse_result.count("convert"))
    {
        // convert to BGEO/VTP
        std::string input_directory = option_parse_result["convert"].as<std::string>();
        bool export_bgeo = option_parse_result.count("bgeo");
        bool export_vtp = option_parse_result.count("vtp");
        bool export_indenter = option_parse_result.count("indenter");
        convert_to_bgeo_vtp(input_directory, export_vtp, export_bgeo, export_indenter);
    }
    else if(option_parse_result.count("convert-parallel"))
    {
        // convert to BGEO/VTP
        std::string input_directory = option_parse_result["convert-parallel"].as<std::string>();
        spdlog::info("converting in parallel; {}", input_directory);
        icy::SnapshotManager::H5Raw_to_Paraview(input_directory);
    }
    else if(option_parse_result.count("simulate"))
    {
        // start simulation
        std::string input_json = option_parse_result["simulate"].as<std::string>();
        bool export_vtp = option_parse_result.count("vtp");
        bool export_h5_raw = option_parse_result.count("export-raw");
        start_simulation_from_json(input_json, export_vtp, export_h5_raw);
    }
    else if(option_parse_result.count("resume"))
    {
        // resume simulation from full snapshot
        std::string input_snapshot = option_parse_result["resume"].as<std::string>();
        bool export_vtp = option_parse_result.count("vtp");
        bool export_h5_raw = option_parse_result.count("export-raw");
        resume_simulation_from_snapshot(input_snapshot, export_vtp, export_h5_raw);
    }

    return 0;
}


