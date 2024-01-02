#include <iostream>
#include <functional>
#include <string>
#include <filesystem>
#include <atomic>
#include <thread>
#include <chrono>

#include <cxxopts.hpp>
#include <spdlog/spdlog.h>
#include <H5Cpp.h>
#include <Partio.h>

#include "model_3d.h"
#include "parameters_sim_3d.h"
#include "snapshotmanager.h"

#include <vtkCellArray.h>
#include <vtkNew.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkFloatArray.h>
#include <vtkPointData.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkCylinderSource.h>
#include <vtkPolyDataMapper.h>
#include <vtkTransform.h>
#include <vtkTransformFilter.h>
#include <vtkAppendFilter.h>

void generate_points(float bx, float by, float bz, int n, std::string fileName);
void start_simulation_from_json(std::string jsonFile);
void resume_simulation_from_snapshot(std::string snapshotFile);

void convert_to_bgeo_vtp(std::string directory, bool vtp, bool bgeo);
void export_bgeo_f(int frame, std::vector<VisualPoint> &current_frame);
void export_vtu_f(int frame, std::vector<VisualPoint> &current_frame, icy::SimParams3D &prms);
void export_indenter_f(int frame, std::vector<VisualPoint> &current_frame, icy::SimParams3D &prms);


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

        convert_to_bgeo_vtp(input_directory, export_vtp, export_bgeo);
    }
    else if(option_parse_result.count("simulate"))
    {
        // start simulation
        std::string input_json = option_parse_result["simulate"].as<std::string>();
        start_simulation_from_json(input_json);
    }
    else if(option_parse_result.count("resume"))
    {
        // resume simulation from full snapshot
        std::string input_snapshot = option_parse_result["resume"].as<std::string>();
        resume_simulation_from_snapshot(input_snapshot);
    }

    return 0;
}



void generate_points(float bx, float by, float bz, int n, std::string fileName)
{
    spdlog::info("generate_points: {}, {}", n, fileName);
    std::vector<std::array<float, 3>> pts = icy::SnapshotManager::GenerateBlock(bx, by, bz, n);
    H5::H5File file(fileName, H5F_ACC_TRUNC);

    hsize_t dims_points[2] = {pts.size(), 3};
    H5::DataSpace dataspace_points(2, dims_points);

    hsize_t chunk_dims[2] = {1024*1024,3};
    if(chunk_dims[0] > pts.size()) chunk_dims[0] = pts.size()/10;
    if(chunk_dims[0] == 0) chunk_dims[0] = 1;
    spdlog::info("chunk {}; dims_pts {}", chunk_dims[0], dims_points[0]);
    H5::DSetCreatPropList proplist;
    proplist.setChunk(2, chunk_dims);
    proplist.setDeflate(7);
    H5::DataSet dataset_points = file.createDataSet("Points_Raw", H5::PredType::NATIVE_FLOAT, dataspace_points, proplist);
    dataset_points.write(pts.data(), H5::PredType::NATIVE_FLOAT);

    file.close();
    spdlog::info("generating and saving done");
}


// =================================== convert frames ==================================


void convert_to_bgeo_vtp(std::string directory, bool vtp, bool bgeo)
{
    spdlog::info("convert to vtp {}, to bgeo {}, directory {}", vtp, bgeo, directory);

    std::vector<VisualPoint> current_frame, saved_frame;
    std::vector<float> indenter_force_buffer;
    std::vector<Vector3r> indenter_force;
    icy::SimParams3D prms;

    // load first frame
    char fileName[20];
    int current_frame_number = 0;
    snprintf(fileName, sizeof(fileName), "v%05d.h5", current_frame_number);
    std::string filePath = path + "/" + fileName;
    spdlog::info("reading visual frame {} to file {}", current_frame_number, filePath);

    if(!std::filesystem::exists(filePath))
    {
        spdlog::critical("file {} does not exist",filePath);
        return;
    }

    H5::H5File file(filePath, H5F_ACC_RDONLY);

    spdlog::info("reading Params");
    // read params
    H5::DataSet dataset_params = file.openDataSet("Params");
    hsize_t dims_params = 0;
    dataset_params.getSpace().getSimpleExtentDims(&dims_params, NULL);
    if(dims_params != sizeof(icy::SimParams3D)) throw std::runtime_error("SimParams3D size mismatch");
    dataset_params.read(&prms, H5::PredType::NATIVE_B8);

    // allocate arrays for snapshot
    indenter_force_buffer.resize(icy::SimParams3D::indenter_array_size);

    // read indenter data
    spdlog::info("reading indenter data");
    H5::DataSet dataset_indenter = file.openDataSet("Indenter_Force");
    dataset_indenter.read(indenter_force_buffer.data(), H5::PredType::NATIVE_FLOAT);
    Eigen::Vector3f indenter_force_elem;
    indenter_force_elem.setZero();
    for(int i=0;i<icy::SimParams3D::indenter_array_size;i++) indenter_force_elem[i%3] += indenter_force_buffer[i];
    indenter_force.push_back(indenter_force_elem);

    // read points
    spdlog::info("reading Points");
    H5::DataSet dataset_points = file.openDataSet("Points");
    hsize_t dims_points = 0;
    dataset_points.getSpace().getSimpleExtentDims(&dims_points, NULL);
    int nPoints = dims_points/sizeof(VisualPoint);
    current_frame.resize(nPoints);
    dataset_points.read(current_frame.data(), H5::PredType::NATIVE_B8);
    file.close();


    // CONVERT FORMATS AS NEEDED
    if(export_bgeo) export_bgeo_f(current_frame_number, current_frame);
    if(export_vtu) export_vtu_f(current_frame_number, current_frame, prms);
    if(export_indenter) export_indenter_f(current_frame_number, current_frame, prms);

    spdlog::info("icy::SnapshotManager::ReadFirstFrame done");

    // load subsequent frames
    int next_frame = 0;
    while(true)
    {
        next_frame++;
        char fileName[20];
        snprintf(fileName, sizeof(fileName), "v%05d.h5", next_frame);
        std::string filePath = path + "/" + fileName;
        spdlog::info("reading file {}", filePath);

        if(!std::filesystem::exists(filePath)) break;

        H5::H5File file(filePath, H5F_ACC_RDONLY);

        H5::DataSet dataset_params = file.openDataSet("Params");
        hsize_t dims_params = 0;
        dataset_params.getSpace().getSimpleExtentDims(&dims_params, NULL);
        if(dims_params != sizeof(icy::SimParams3D)) throw std::runtime_error("SimParams3D size mismatch");
        dataset_params.read(&prms, H5::PredType::NATIVE_B8);

        // read indenter data
        H5::DataSet dataset_indenter = file.openDataSet("Indenter_Force");
        dataset_indenter.read(indenter_force_buffer.data(), H5::PredType::NATIVE_FLOAT);
        indenter_force_elem.setZero();
        for(int i=0;i<icy::SimParams3D::indenter_array_size;i++) indenter_force_elem[i%3] += indenter_force_buffer[i];
        indenter_force.push_back(indenter_force_elem);

        H5::DataSet dataset_points = file.openDataSet("Points");
        hsize_t dims_points = 0;
        dataset_points.getSpace().getSimpleExtentDims(&dims_points, NULL);
        int nPoints = dims_points/sizeof(VisualPoint);
        saved_frame.resize(nPoints);
        dataset_points.read(saved_frame.data(), H5::PredType::NATIVE_B8);
        file.close();

        // advance "current_frame" one step forward
        for(int i=0; i<prms.nPts;i++)
        {
            VisualPoint &vp = current_frame[i];
            Eigen::Vector3f updated_pos = vp.pos() + vp.vel()*prms.InitialTimeStep;
            for(int j=0;j<3;j++) vp.p[j] = updated_pos[j];
        }

        // update select points
        for(int i=0; i<saved_frame.size(); i++)
        {
            VisualPoint &vp = saved_frame[i];
            current_frame[vp.id] = vp;
        }

        if(export_bgeo) export_bgeo_f(next_frame, current_frame);
        if(export_vtu) export_vtu_f(next_frame, current_frame, prms);
        if(export_indenter) export_indenter_f(next_frame, current_frame, prms);

        spdlog::info("read next frame {} done", next_frame);
    }

    if(export_force)
    {
        spdlog::info("saving indenter_force");
        std::ofstream ofs("indenter_force.csv", std::ofstream::out | std::ofstream::trunc);
        ofs << "fx,fy,fz,F_total\n";
        for(int i=0;i<indenter_force.size();i++)
        {
            Eigen::Vector3f &v = indenter_force[i];
            ofs << v[0] << ',' << v[1] << ',' << v[2] << ',' << v.norm() << '\n';
        }
        ofs.close();
    }

}


void export_indenter_f(int frame, std::vector<VisualPoint> &current_frame, icy::SimParams3D &prms)
{
    spdlog::info("export indenter {}",frame);
    std::string dir;
    char fileName[20];

    // INDENTER !!!
    dir = "output_vtu_indenter";
    snprintf(fileName, sizeof(fileName), "i_%05d.vtu", frame);
    std::string savePath = dir + "/" + fileName;
    spdlog::info("writing vtk file {}", savePath);

    vtkNew<vtkCylinderSource> cylinder;
    //    vtkNew<vtkPolyDataMapper> indenterMapper;
    vtkNew<vtkTransform> transform;
    vtkNew<vtkTransformFilter> transformFilter;
    vtkNew<vtkAppendFilter> appendFilter;
    vtkNew<vtkUnstructuredGrid> unstructuredGrid;
    vtkNew<vtkXMLUnstructuredGridWriter> writer2;


    cylinder->SetResolution(33);
    cylinder->SetRadius(prms.IndDiameter/2.f);
    cylinder->SetHeight(prms.GridZ * prms.cellsize);


    double indenter_x = prms.indenter_x;
    double indenter_y = prms.indenter_y;
    double indenter_z = prms.GridZ * prms.cellsize/2;
    cylinder->SetCenter(indenter_x, indenter_z, -indenter_y);
    cylinder->Update();


    transform->RotateX(90);
    transformFilter->SetTransform(transform);
    transformFilter->SetInputConnection(cylinder->GetOutputPort());
    transformFilter->Update();

    //    indenterMapper->SetInputConnection(transformFilter->GetOutputPort());



    // Combine the two data sets.
    //    appendFilter->AddInputData(indenterMapper->GetOutput());
    appendFilter->SetInputConnection(transformFilter->GetOutputPort());
    appendFilter->Update();

    unstructuredGrid->ShallowCopy(appendFilter->GetOutput());

    // Write the unstructured grid.
    writer2->SetFileName(savePath.c_str());
    writer2->SetInputData(unstructuredGrid);
    writer2->Write();
}


void export_vtu_f(int frame, std::vector<VisualPoint> &current_frame, icy::SimParams3D &prms)
{
    spdlog::info("export_vtu {}", frame);
    int n = current_frame.size();

    vtkNew<vtkPoints> points;
    vtkNew<vtkFloatArray> values;
    points->SetNumberOfPoints(n);
    values->SetNumberOfValues(n);
    values->SetName("Jp_inv");

    for(int i=0;i<n;i++)
    {
        VisualPoint &vp = current_frame[i];
        points->SetPoint(i, vp.p[0], vp.p[1], vp.p[2]);
        values->SetValue(i, vp.Jp_inv);
    }
    values->Modified();

    vtkNew<vtkPolyData> polydata;
    polydata->SetPoints(points);
    polydata->GetPointData()->AddArray(values);
    polydata->GetPointData()->SetActiveScalars("Jp_inv");

    std::string dir = "output_vtk";
    char fileName[20];
    snprintf(fileName, sizeof(fileName), "p_%05d.vtp", frame);
    std::string savePath = dir + "/" + fileName;
    spdlog::info("writing vtp file {}", savePath);

    // Write the file
    vtkNew<vtkXMLPolyDataWriter> writer;
    writer->SetFileName(savePath.c_str());
    writer->SetInputData(polydata);
    writer->Write();

    spdlog::info("export_vtk frame done {}", frame);
}


void export_bgeo_f(int frame, std::vector<VisualPoint> &current_frame)
{
    spdlog::info("export_bgeo frame {}", frame);

    Partio::ParticlesDataMutable* parts = Partio::create();
    Partio::ParticleAttribute attr_Jp = parts->addAttribute("Jp_inv", Partio::FLOAT, 1);
    Partio::ParticleAttribute attr_pos = parts->addAttribute("position", Partio::VECTOR, 3);

    parts->addParticles(current_frame.size());
    for(int i=0;i<current_frame.size();i++)
    {
        float* val = parts->dataWrite<float>(attr_pos, i);
        for(int j=0;j<3;j++) val[j] = current_frame[i].p[j];
        float *val_Jp = parts->dataWrite<float>(attr_Jp, i);
        val_Jp[0] = current_frame[i].Jp_inv;
    }

    // filename
    std::string bgeo_dir = "output_bgeo";

    char fileName[20];
    snprintf(fileName, sizeof(fileName), "%05d.bgeo", frame);
    std::string savePath = bgeo_dir + "/" + fileName;
    spdlog::info("writing bgeo file {}", savePath);
    Partio::write(savePath.c_str(), *parts);
    parts->release();
    spdlog::info("export_bgeo frame done {}", frame);
}










// ======================= run simulation ======================================

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
