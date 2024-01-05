#include <spdlog/spdlog.h>
#include <H5Cpp.h>
#include <Partio.h>

#include <vtkCellArray.h>
#include <vtkNew.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkFloatArray.h>
#include <vtkIntArray.h>
#include <vtkPointData.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkCylinderSource.h>
#include <vtkPolyDataMapper.h>
#include <vtkTransform.h>
#include <vtkTransformFilter.h>
#include <vtkAppendFilter.h>
#include <vtkStructuredGrid.h>
#include <vtkXMLStructuredGridWriter.h>
#include <vtkCellData.h>

#include "parameters_sim_3d.h"
#include "snapshotmanager.h"


void export_bgeo_f(int frame, std::vector<icy::SnapshotManager::VisualPoint> &current_frame);
void export_indenter_f(int frame, std::vector<icy::SnapshotManager::VisualPoint> &current_frame,
                       icy::SimParams3D &prms, std::vector<double> &indenter_force_buffer);
void export_vtu_f(int frame, std::vector<icy::SnapshotManager::VisualPoint> &current_frame, icy::SimParams3D &prms,
                  std::vector<icy::SnapshotManager::VisualPoint> &initial_frame);


void convert_to_bgeo_vtp(std::string directory, bool export_vtp, bool export_bgeo, bool export_indenter)
{
    spdlog::info("convert to vtp {}, to bgeo {}, directory {}", export_vtp, export_bgeo, directory);

    std::vector<icy::SnapshotManager::VisualPoint> current_frame, saved_frame, initial_frame;
    std::vector<double> indenter_force_buffer;
    std::vector<Vector3r> indenter_force_history;
    icy::SimParams3D prms;


    int frame = 1;
    std::string filePath;
    char fileName[20];

    snprintf(fileName, sizeof(fileName), "v%05d.h5", frame);
    filePath = directory + "/" + fileName;

    do
    {
        spdlog::info("reading visual frame {}", frame);

        H5::H5File file(filePath, H5F_ACC_RDONLY);
        H5::DataSet dataset_params = file.openDataSet("Params");
        hsize_t dims_params = 0;
        dataset_params.getSpace().getSimpleExtentDims(&dims_params, NULL);
        if(dims_params != sizeof(icy::SimParams3D)) throw std::runtime_error("SimParams3D size mismatch");
        dataset_params.read(&prms, H5::PredType::NATIVE_B8);

        // read indenter data
        indenter_force_buffer.resize(prms.indenter_array_size);
        spdlog::info("reading indenter data");
        H5::DataSet dataset_indenter = file.openDataSet("Indenter_Force");
        dataset_indenter.read(indenter_force_buffer.data(), H5::PredType::NATIVE_DOUBLE);

        Vector3r indenter_force_elem;
        indenter_force_elem.setZero();
        for(int i=0;i<prms.indenter_array_size;i++) indenter_force_elem[i%3] += indenter_force_buffer[i];
        indenter_force_history.push_back(indenter_force_elem);

        if(export_bgeo || export_vtp)
        {
            H5::DataSet dataset_points = file.openDataSet("Points");
            hsize_t dims_points = 0;
            dataset_points.getSpace().getSimpleExtentDims(&dims_points, NULL);
            int nPoints = dims_points/sizeof(icy::SnapshotManager::VisualPoint);
            saved_frame.resize(nPoints);
            dataset_points.read(saved_frame.data(), H5::PredType::NATIVE_B8);

            if(current_frame.size()==0)
            {
                // this is the first frame
                initial_frame = current_frame = saved_frame;
            }
            else
            {
                // advance "current_frame" one step forward
                for(int i=0; i<prms.nPts;i++)
                {
                    icy::SnapshotManager::VisualPoint &vp = current_frame[i];
                    Eigen::Map<Eigen::Vector3f> updated_pos(vp.p);
                    updated_pos = vp.pos() + vp.vel()*prms.InitialTimeStep;
                    //for(int j=0;j<3;j++) vp.p[j] = updated_pos[j];
                }

                // update select points
                for(int i=0; i<saved_frame.size(); i++)
                {
                    icy::SnapshotManager::VisualPoint &vp = saved_frame[i];
                    current_frame[vp.id] = vp;
                }
            }
        }

        file.close();

        if(export_bgeo) export_bgeo_f(frame, current_frame);
        if(export_vtp) export_vtu_f(frame, current_frame, prms, initial_frame);
        if(export_indenter) export_indenter_f(frame, current_frame, prms, indenter_force_buffer);

        frame++;
        snprintf(fileName, sizeof(fileName), "v%05d.h5", frame);
        filePath = directory + "/" + fileName;
        std::cout << '\n';
    } while(std::filesystem::exists(filePath));

    // write CSV
    spdlog::info("saving indenter_force");
    std::ofstream ofs("indenter_force.csv", std::ofstream::out | std::ofstream::trunc);
    ofs << "t,F_total,fx,fy\n";
    for(int i=0;i<indenter_force_history.size();i++)
    {
        Vector3r &v = indenter_force_history[i];
        double t = i*prms.InitialTimeStep*prms.UpdateEveryNthStep;
        ofs << t << ',' << v.norm() << ',' << v[0] << ',' << v[1] << '\n';
    }
    ofs.close();

}


void export_indenter_f(int frame, std::vector<icy::SnapshotManager::VisualPoint> &current_frame, icy::SimParams3D &prms,
                       std::vector<double> &indenter_force_buffer)
{
    std::string dir = "output_vtu_indenter";
    // ensure that directory exists
    std::filesystem::path od(dir);
    if(!std::filesystem::is_directory(od) || !std::filesystem::exists(od)) std::filesystem::create_directory(od);
    char fileName[20];

    // INDENTER / cylinder shape
    snprintf(fileName, sizeof(fileName), "i_%05d.vtu", frame);
    std::string savePath = dir + "/" + fileName;
    spdlog::info("writing vtk file {}", savePath);

    vtkNew<vtkCylinderSource> cylinder;
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

    // Combine the two data sets.
    //    appendFilter->AddInputData(indenterMapper->GetOutput());
    appendFilter->SetInputConnection(transformFilter->GetOutputPort());
    appendFilter->Update();

    unstructuredGrid->ShallowCopy(appendFilter->GetOutput());

    // Write the unstructured grid.
    writer2->SetFileName(savePath.c_str());
    writer2->SetInputData(unstructuredGrid);
    writer2->Write();


    // TEKSCAN-like grid
    vtkNew<vtkPoints> grid_points;
    vtkNew<vtkStructuredGrid> structuredGrid;

    double h = prms.cellsize;
    double hsq = h*h;
    int indenter_blocks = (int)((prms.IceBlockDimZ/h)*0.8+prms.GridZ*0.2);
    int fromZ = (prms.GridZ-indenter_blocks)/2;

    int nx = indenter_blocks+1;
    int ny = prms.n_indenter_subdivisions_angular*0.3+1;
    structuredGrid->SetDimensions(nx, ny, 1);
    grid_points->SetNumberOfPoints(nx*ny);
    for(int idx_y=0; idx_y<ny; idx_y++)
        for(int idx_x=0; idx_x<nx; idx_x++)
        {
            grid_points->SetPoint(idx_x+idx_y*nx, idx_x*h, idx_y*h, 0);
        }
    structuredGrid->SetPoints(grid_points);

    vtkNew<vtkFloatArray> values;
    values->SetName("Pressure");
    values->SetNumberOfValues((nx-1)*(ny-1));
    for(int idx_y=0; idx_y<(ny-1); idx_y++)
        for(int idx_x=0; idx_x<(nx-1); idx_x++)
        {
            int idx = (idx_x+fromZ) + prms.GridZ*(prms.n_indenter_subdivisions_angular-idx_y-1);
            Eigen::Map<Vector3r> f(&indenter_force_buffer[3*idx]);
            values->SetValue((idx_x+idx_y*(nx-1)), f.norm()/hsq);
        }

    structuredGrid->GetCellData()->SetScalars(values);

    // Write the unstructured grid.
    dir = "output_tekscan";
    // ensure that directory exists
    std::filesystem::path od2(dir);
    if(!std::filesystem::is_directory(od2) || !std::filesystem::exists(od2)) std::filesystem::create_directory(od2);
    snprintf(fileName, sizeof(fileName), "t_%05d.vts", frame);
    savePath = dir + "/" + fileName;
    spdlog::info("writing vts file for tekscan-like grid {}", savePath);

    vtkNew<vtkXMLStructuredGridWriter> writer3;
    writer3->SetFileName(savePath.c_str());
    writer3->SetInputData(structuredGrid);
    writer3->Write();
}


void export_vtu_f(int frame, std::vector<icy::SnapshotManager::VisualPoint> &current_frame, icy::SimParams3D &prms,
        std::vector<icy::SnapshotManager::VisualPoint> &initial_frame)
{
    spdlog::info("export_vtu {}", frame);
    int n = current_frame.size();

    vtkNew<vtkPoints> points;
    vtkNew<vtkFloatArray> values, displacement_total;
    vtkNew<vtkIntArray> values_random_colors;
    points->SetNumberOfPoints(n);
    values->SetNumberOfValues(n);
    displacement_total->SetNumberOfValues(n);
    values_random_colors->SetNumberOfValues(n);
    values->SetName("Jp_inv");
    values_random_colors->SetName("random_colors");
    displacement_total->SetName("displacement");

    for(int i=0;i<n;i++)
    {
        icy::SnapshotManager::VisualPoint &vp = current_frame[i];
        icy::SnapshotManager::VisualPoint &vpi = initial_frame[i];
        points->SetPoint(i, vp.p[0], vp.p[1], vp.p[2]);
        values->SetValue(i, vp.Jp_inv);
        int color_value = (i%4)+(vp.Jp_inv < 1 ? 0 : 10);
        values_random_colors->SetValue(i, color_value);
        displacement_total->SetValue(i, (vp.pos()-vpi.pos()).norm());
    }
    values->Modified();
    values_random_colors->Modified();

    vtkNew<vtkPolyData> polydata;
    polydata->SetPoints(points);
    polydata->GetPointData()->AddArray(values);
    polydata->GetPointData()->AddArray(values_random_colors);
    polydata->GetPointData()->AddArray(displacement_total);
//    polydata->GetPointData()->SetActiveScalars("Jp_inv");

    std::string dir = "output_vtk";
    std::filesystem::path od(dir);
    if(!std::filesystem::is_directory(od) || !std::filesystem::exists(od)) std::filesystem::create_directory(od);

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


void export_bgeo_f(int frame, std::vector<icy::SnapshotManager::VisualPoint> &current_frame)
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
    std::filesystem::path od(bgeo_dir);
    if(!std::filesystem::is_directory(od) || !std::filesystem::exists(od)) std::filesystem::create_directory(od);

    char fileName[20];
    snprintf(fileName, sizeof(fileName), "%05d.bgeo", frame);
    std::string savePath = bgeo_dir + "/" + fileName;
    spdlog::info("writing bgeo file {}", savePath);
    Partio::write(savePath.c_str(), *parts);
    parts->release();
    spdlog::info("export_bgeo frame done {}", frame);
}


