#include "model_3d.h"
#include <spdlog/spdlog.h>

icy::Model3D::Model3D()
{
    prms.Reset();
    gpu.prms = &prms;
    gpu.model = this;
};


bool icy::Model3D::Step()
{
    /*
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

    processing_current_cycle_data.lock();   // if locked, previous results are not yet processed by the host

    gpu.cuda_transfer_from_device();

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
*/
    return false;
}


void icy::Model3D::UnlockCycleMutex()
{
    // current data was handled by host - allow next cycle to proceed
    processing_current_cycle_data.unlock();
}


void icy::Model3D::Reset()
{
    // this should be called after prms are set as desired (either via GUI or CLI)
    spdlog::info("icy::Model::Reset()");

    prms.SimulationStep = 0;
    prms.SimulationTime = 0;
    compute_time_per_cycle = 0;
    indenter_force_history.clear();

    const real &bx = prms.IceBlockDimX;
    const real &by = prms.IceBlockDimY;
    const real &bz = prms.IceBlockDimZ;
    const real &bvol = bx*by*bz;
    const real &h = prms.cellsize;
    const real &magic_constant = 0.25;

    const real z_center = prms.GridZ*h/2;
    const real block_z_min = max(z_center - bz/2, 0);
    const real bloc_z_max = min(z_center + bz/2, prms.GridZ*h);
    const real kRadius =cbrt(magic_constant*bvol/prms.PointsWanted);
    const std::array<real, 3>kXMin{5.0*h, 2.0*h, block_z_min};
    const std::array<real, 3>kXMax{5.0*h+block_length, 2.0*h+block_height, block_z_max};

    spdlog::info("starting thinks::PoissonDiskSampling");
    std::vector<std::array<real, 3>> prresult = thinks::PoissonDiskSampling(kRadius, kXMin, kXMax);
    prms.nPts = prresult.size();
    spdlog::info("finished thinks::PoissonDiskSampling; {} ", prms.nPts);
    points.resize(prms.nPts);

    prms.ParticleVolume = bvol/prms.nPts;
    prms.ParticleMass = prms.ParticleVolume*prms.Density;
    for(int k = 0; k<prms.nPts; k++)
    {
        Point3D &p = points[k];
        p.Reset();
        for(int i=0;i<3;i++) p.pos[i] = prresult[k][i];
        p.pos_initial = p.pos;
    }
    prms.indenter_y = block_height + 2*h + prms.IndDiameter/2 - prms.IndDepth;
    prms.indenter_x = prms.indenter_x_initial = 4*h - prms.IndDiameter/2;

    gpu.cuda_allocate_arrays(prms.GridTotal, prms.nPts);
    gpu.transfer_ponts_to_device(points);
    Prepare();
    spdlog::info("icy::Model::Reset() done");
}

void icy::Model3D::ResetToStep0()
{
    spdlog::info("ResetToStep0()");

    prms.SimulationStep = 0;
    prms.SimulationTime = 0;
    compute_time_per_cycle = 0;
    indenter_force_history.clear();

    const real &h = prms.cellsize;

    for(int k = 0; k<points.size(); k++) points[k].Reset();
    prms.indenter_y = prms.BlockHeight + 2*h + prms.IndDiameter/2 - prms.IndDepth;
    prms.indenter_x = prms.indenter_x_initial = 5*h - prms.IndDiameter/2 - h;
    gpu.transfer_ponts_to_device(points);
    Prepare();
    spdlog::info("ResetToStep0() done");
}


void icy::Model3D::Prepare()
{
    spdlog::info("icy::Model::Prepare()");
    abortRequested = false;
    gpu.cuda_update_constants();
}

