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


void generate_points(float bx, float by, float bz, int n, std::string fileName);
void start_simulation_from_json(std::string jsonFile, bool export_vtp);
void resume_simulation_from_snapshot(std::string snapshotFile, bool export_vtp);
void convert_to_bgeo_vtp(std::string directory, bool vtp, bool bgeo, bool export_indenter);


int main(int argc, char** argv)
{
    // parse options
    cxxopts::Options options("Ice MPM 3D", "CLI version of MPM 3D simulation");

    options.add_options()
        // point generation
        ("g,generate", "Make a set of N points for the simulation starting input", cxxopts::value<int>()->default_value("10000000"))
        ("o,output", "Output file name", cxxopts::value<std::string>()->default_value("raw_10m.h5"))
        ("x,bx", "Length of the block", cxxopts::value<float>()->default_value("2.5"))
        ("y,by", "Height of the block", cxxopts::value<float>()->default_value("1.0"))
        ("z,bz", "Width of the block", cxxopts::value<float>()->default_value("1.5"))

        // simulation output (.H5) conversion to BGEO and/or VTP
        ("c,convert", "Directory where iterative h5 fies are saved", cxxopts::value<std::string>())
        ("b,bgeo", "Export as BGEO")
        ("v,vtp", "Export points as VTP, cylinder as VTU")
        ("i,indenter", "Export indenter data")

        // execution of the simulation
        ("s,simulate", "Input JSON configuration file", cxxopts::value<std::string>())
        ("r,resume", "Input full snapshot (.h5) file", cxxopts::value<std::string>())
        ;
    options.parse_positional({"file"});

    auto option_parse_result = options.parse(argc, argv);

    if(option_parse_result.count("generate"))
    {
        // generate points input file
        std::string output_file = option_parse_result["output"].as<std::string>();
        int n = option_parse_result["generate"].as<int>();
        float bx = option_parse_result["bx"].as<float>();
        float by = option_parse_result["by"].as<float>();
        float bz = option_parse_result["bz"].as<float>();
        generate_points(bx, by, bz, n, output_file);
    }
    else if(option_parse_result.count("convert"))
    {
        // convert to BGEO/VTP
        std::string input_directory = option_parse_result["convert"].as<std::string>();
        bool export_bgeo = option_parse_result.count("bgeo");
        bool export_vtp = option_parse_result.count("vtp");
        bool export_indenter = option_parse_result.count("indenter");
        convert_to_bgeo_vtp(input_directory, export_vtp, export_bgeo, export_indenter);
    }
    else if(option_parse_result.count("simulate"))
    {
        // start simulation
        std::string input_json = option_parse_result["simulate"].as<std::string>();
        bool export_vtp = option_parse_result.count("vtp");
        start_simulation_from_json(input_json, export_vtp);
    }
    else if(option_parse_result.count("resume"))
    {
        // resume simulation from full snapshot
        std::string input_snapshot = option_parse_result["resume"].as<std::string>();
        bool export_vtp = option_parse_result.count("vtp");
        resume_simulation_from_snapshot(input_snapshot, export_vtp);
    }

    return 0;
}


