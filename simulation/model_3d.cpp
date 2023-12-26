#include "model_3d.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>

icy::Model3D::Model3D()
{
    prms.Reset();
    gpu.model = this;
    outputDirectory = "default_output";
    compute_time_per_cycle = 0;
};


bool icy::Model3D::Step()
{
    real simulation_time = prms.SimulationTime;
    std::cout << '\n';
    spdlog::info("step {} ({}) started; sim_time {:.3}", prms.SimulationStep, prms.SimulationStep/prms.UpdateEveryNthStep, simulation_time);

    int count_unupdated_steps = 0;
    gpu.cuda_reset_indenter_force_accumulator();
    if(prms.SimulationStep % (prms.UpdateEveryNthStep*2) == 0) cudaEventRecord(gpu.eventCycleStart);
    do
    {
        prms.indenter_x = prms.indenter_x_initial + simulation_time*prms.IndVelocity;
        gpu.cuda_reset_grid();
        gpu.cuda_p2g();
        gpu.cuda_update_nodes(prms.indenter_x, prms.indenter_y);
        gpu.cuda_g2p();
        count_unupdated_steps++;
        simulation_time += prms.InitialTimeStep;
    } while((prms.SimulationStep+count_unupdated_steps) % prms.UpdateEveryNthStep != 0);
    if(prms.SimulationStep % (prms.UpdateEveryNthStep*2) == 0) cudaEventRecord(gpu.eventCycleStop);
    spdlog::info("cycle loop completed for step {}",prms.SimulationStep);

    processing_current_cycle_data.lock();   // if locked, previous results are not yet processed by the host

    gpu.cuda_transfer_from_device();
    spdlog::info("went past cuda_transfer_from_device()");

    if(prms.SimulationStep % (prms.UpdateEveryNthStep*2) != 0)
    {
        cudaEventSynchronize(gpu.eventCycleStop);
        cudaEventElapsedTime(&compute_time_per_cycle, gpu.eventCycleStart, gpu.eventCycleStop);
        compute_time_per_cycle /= prms.UpdateEveryNthStep;
        spdlog::info("cycle time {:.3} ms", compute_time_per_cycle);
    }

    prms.SimulationTime = simulation_time;
    prms.SimulationStep += count_unupdated_steps;
    return (prms.SimulationTime < prms.SimulationEndTime && !gpu.error_code);
}


void icy::Model3D::UnlockCycleMutex()
{
    // current data was handled by host - allow next cycle to proceed
    processing_current_cycle_data.unlock();
}


void icy::Model3D::Reset()
{
    const real &by = prms.IceBlockDimY;
    const real &h = prms.cellsize;

    // this should be called after prms are set as desired (either via GUI or CLI)
    spdlog::info("icy::Model::Reset()");

    prms.SimulationStep = 0;
    prms.SimulationTime = 0;
    compute_time_per_cycle = 0;

    prms.indenter_y = by + 2*h + prms.IndDiameter/2 - prms.IndDepth;
    prms.indenter_x = prms.indenter_x_initial = 4*h - prms.IndDiameter/2;

    spdlog::info("icy::Model::Reset() done");
}




void icy::Model3D::Prepare()
{
    spdlog::info("icy::Model::Prepare()");
    abortRequested = false;
    gpu.cuda_update_constants();
    spdlog::info("icy::Model::Prepare() done");
}

