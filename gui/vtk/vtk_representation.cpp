#include "vtk_representation.h"
#include "model_3d.h"
#include "parameters_sim_3d.h"
//#include <omp.h>
#include <algorithm>
#include <iostream>

icy::VisualRepresentation::VisualRepresentation()
{
    int nLut = sizeof lutArrayTemperatureAdj / sizeof lutArrayTemperatureAdj[0];
//    hueLut->SetNumberOfTableValues(nLut);
//    for ( int i=0; i<nLut; i++)
//        hueLut->SetTableValue(i, lutArrayTemperatureAdj[i][0],
//                              lutArrayTemperatureAdj[i][1],
//                              lutArrayTemperatureAdj[i][2], 1.0);

    nLut = sizeof lutArrayPastel / sizeof lutArrayPastel[0];
    hueLut_pastel->SetNumberOfTableValues(nLut);
    for ( int i=0; i<nLut; i++)
        hueLut_pastel->SetTableValue(i, lutArrayPastel[i][0],
                              lutArrayPastel[i][1],
                              lutArrayPastel[i][2], 1.0);
    hueLut_pastel->SetTableRange(0,39);

    nLut = sizeof lutArrayMPMColors / sizeof lutArrayMPMColors[0];
    lutMPM->SetNumberOfTableValues(nLut);
    for ( int i=0; i<nLut; i++)
        lutMPM->SetTableValue(i, lutArrayMPMColors[i][0],
                              lutArrayMPMColors[i][1],
                              lutArrayMPMColors[i][2], 1.0);

    hueLut_four->SetNumberOfColors(5);
    hueLut_four->SetTableValue(0, 0.3, 0.3, 0.3);
    hueLut_four->SetTableValue(1, 1.0, 0, 0);
    hueLut_four->SetTableValue(2, 0, 1.0, 0);
    hueLut_four->SetTableValue(3, 0, 0, 1.0);
    hueLut_four->SetTableValue(4, 0, 0.5, 0.5);
    hueLut_four->SetTableRange(0,4);


    indenterMapper->SetInputConnection(indenterSource->GetOutputPort());
    actor_indenter->SetMapper(indenterMapper);

    indenterMapper->SetInputConnection(indenterSource->GetOutputPort());
    actor_indenter->SetMapper(indenterMapper);
//    actor_indenter->GetProperty()->LightingOff();
    actor_indenter->GetProperty()->EdgeVisibilityOn();
    actor_indenter->GetProperty()->VertexVisibilityOff();
    actor_indenter->GetProperty()->SetColor(0.3,0.1,0.1);
    actor_indenter->GetProperty()->SetEdgeColor(90.0/255.0, 90.0/255.0, 97.0/255.0);
//    actor_indenter->GetProperty()->ShadingOff();
//    actor_indenter->GetProperty()->SetInterpolationToFlat();
    actor_indenter->PickableOff();
    actor_indenter->GetProperty()->SetLineWidth(3);
//    actor_indenter->RotateX(90);
    actor_indenter->RotateZ(90);
    actor_indenter->RotateX(90);


    points_polydata->SetPoints(points);
    points_polydata->GetPointData()->AddArray(visualized_values);

    points_filter->SetInputData(points_polydata);
    points_filter->Update();

    points_mapper->SetInputData(points_filter->GetOutput());
    points_mapper->UseLookupTableScalarRangeOn();
    points_mapper->SetLookupTable(lutMPM);

    visualized_values->SetName("visualized_values");

    actor_points->SetMapper(points_mapper);
    actor_points->GetProperty()->SetPointSize(2);
    actor_points->GetProperty()->SetVertexColor(1.,1.,0);
    actor_points->GetProperty()->SetColor(0.1,0.1,0.4);
//    actor_points->GetProperty()->LightingOff();
//    actor_points->GetProperty()->ShadingOff();
//    actor_points->GetProperty()->SetInterpolationToFlat();
    actor_points->PickableOff();
//    actor_points->GetProperty()->RenderPointsAsSpheresOn();


    grid_mapper->SetInputData(structuredGrid);
//    grid_mapper->SetLookupTable(hueLut);

    actor_grid->SetMapper(grid_mapper);
    actor_grid->GetProperty()->SetEdgeVisibility(true);
//    actor_grid->GetProperty()->SetEdgeColor(0.8,0.8,0.8);
    actor_grid->GetProperty()->LightingOff();
    actor_grid->GetProperty()->ShadingOff();
    actor_grid->GetProperty()->SetInterpolationToFlat();
    actor_grid->PickableOff();
    actor_grid->GetProperty()->SetColor(0.05,0.05,0.05);
    actor_grid->GetProperty()->SetRepresentationToWireframe();

    scalarBar->SetLookupTable(lutMPM);
    scalarBar->SetMaximumWidthInPixels(130);
    scalarBar->SetBarRatio(0.07);
    scalarBar->SetMaximumHeightInPixels(200);
    scalarBar->GetPositionCoordinate()->SetCoordinateSystemToNormalizedDisplay();
    scalarBar->GetPositionCoordinate()->SetValue(0.01,0.015, 0.0);
    scalarBar->SetLabelFormat("%.1e");
    scalarBar->GetLabelTextProperty()->BoldOff();
    scalarBar->GetLabelTextProperty()->ItalicOff();
    scalarBar->GetLabelTextProperty()->ShadowOff();
    scalarBar->GetLabelTextProperty()->SetColor(0.1,0.1,0.1);

    // text
    vtkTextProperty* txtprop = actorText->GetTextProperty();
    txtprop->SetFontFamilyToArial();
    txtprop->BoldOff();
    txtprop->SetFontSize(14);
    txtprop->ShadowOff();
    txtprop->SetColor(0,0,0);
    actorText->SetDisplayPosition(500, 30);
}



void icy::VisualRepresentation::SynchronizeTopology()
{
    points->SetNumberOfPoints(model->points.size());
    visualized_values->SetNumberOfValues(model->points.size());
    points_polydata->GetPointData()->SetActiveScalars("visualized_values");
    points_mapper->SetColorModeToMapScalars();

    SynchronizeValues();

    // structured grid
    real &h = model->prms.cellsize;
    real gx = model->prms.GridX*h;
    real gy = model->prms.GridY*h;
    real gz = model->prms.GridZ*h;

    structuredGrid->SetDimensions(2, 2, 2);

    grid_points->SetNumberOfPoints(8);
    grid_points->SetPoint(0, 0.,0.,0.);
    grid_points->SetPoint(1, gx, 0, 0);
    grid_points->SetPoint(2, 0, gy, 0);
    grid_points->SetPoint(3, gx, gy, 0);
    grid_points->SetPoint(4, 0.,0.,gz);
    grid_points->SetPoint(5, gx, 0, gz);
    grid_points->SetPoint(6, 0, gy, gz);
    grid_points->SetPoint(7, gx, gy, gz);

    structuredGrid->SetPoints(grid_points);

    indenterSource->SetRadius(model->prms.IndDiameter/2.f);
    indenterSource->SetHeight(model->prms.IceBlockDimZ);
    indenterSource->SetResolution(33);
}


void icy::VisualRepresentation::SynchronizeValues()
{
    actor_points->GetProperty()->SetPointSize(model->prms.ParticleViewSize);

    model->hostside_data_update_mutex.lock();
//#pragma omp parallel
    for(int i=0;i<model->points.size();i++)
    {
        const icy::Point3D &p = model->points[i];
        points->SetPoint((vtkIdType)i, p.pos[0], p.pos[1], p.pos[2]);
      }

    double centerVal = 0;
    double range = std::pow(10,ranges[VisualizingVariable]);
    points_mapper->SetLookupTable(lutMPM);
    scalarBar->SetLookupTable(lutMPM);


    if(VisualizingVariable == VisOpt::none)
    {
        points_mapper->ScalarVisibilityOff();

        //points_mapper->UseLookupTableScalarRangeOff();
    }
    else if(VisualizingVariable == VisOpt::NACC_case)
    {
        points_mapper->ScalarVisibilityOn();

        for(int i=0;i<model->points.size();i++) visualized_values->SetValue((vtkIdType)i, model->points[i].q);
        points_mapper->SetLookupTable(hueLut_four);
        scalarBar->SetLookupTable(hueLut_four);
    }
    else if(VisualizingVariable == VisOpt::Jp)
    {
        points_mapper->ScalarVisibilityOn();

        for(int i=0;i<model->points.size();i++) visualized_values->SetValue((vtkIdType)i, model->points[i].Jp_inv-1);
    }

    model->hostside_data_update_mutex.unlock();


//    float minmax[2];
//    visualized_values->GetValueRange(minmax);
    lutMPM->SetTableRange(centerVal-range, centerVal+range);
    hueLut->SetTableRange(centerVal-range, centerVal+range);

    points->Modified();
    visualized_values->Modified();
    points_filter->Update();

    double indenter_x = model->prms.indenter_x;
    double indenter_y = model->prms.indenter_y;
    double indenter_z = model->prms.GridZ*model->prms.cellsize/2;
    indenterSource->SetCenter(indenter_y, indenter_z, indenter_x);  // y z x
}


void icy::VisualRepresentation::ChangeVisualizationOption(int option)
{
    VisualizingVariable = (VisOpt)option;
    SynchronizeTopology();
}


