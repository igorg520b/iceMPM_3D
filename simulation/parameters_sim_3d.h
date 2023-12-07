#ifndef P_SIM_H
#define P_SIM_H

#include <iostream>
#include <string>
#include <filesystem>
#include <fstream>

#include <Eigen/Core>
#include <Eigen/LU>

#include "rapidjson/reader.h"
#include "rapidjson/document.h"
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

typedef double real;
//typedef float real;
typedef Eigen::Vector3<real> Vector3r;
typedef Eigen::Matrix3<real> Matrix3r;


// variables related to the formulation of the model

namespace icy { struct SimParams3D; }

struct icy::SimParams3D
{
public:
    constexpr static double pi = 3.14159265358979323846;
    constexpr static int dim = 3;
    constexpr static int nGridArrays = 4;   // vx, vy, vz, m

    // index of the corresponding array in SoA
    constexpr static size_t idx_intact = 0;
    constexpr static size_t idx_Jp_inv = 1;
    constexpr static size_t posx = 2;
    constexpr static size_t velx = posx + 1;
    constexpr static size_t Fe00 = velx+3;
    constexpr static size_t Bp00 = Fe00+9;
    constexpr static size_t nPtsArrays = Bp00 + 9;

    real *grid_array;      // device-side grid data
    real *pts_array;
    size_t nPtsPitch, nGridPitch; // in number of elements(!), for coalesced access on the device
    constexpr static int n_indenter_subdivisions_angular = 360;
    constexpr static int n_indenter_subdivisions_transverse = 512;
    constexpr static int indenter_array_size = 3*n_indenter_subdivisions_angular*n_indenter_subdivisions_transverse;
    real *indenter_force_accumulator; // size is indenter_array_size
    int tpb_P2G, tpb_Upd, tpb_G2P;  // threads per block for each operation

    int PointsWanted, nPts;
    int GridX, GridY, GridZ;
    real GridXDimension;

    real InitialTimeStep, SimulationEndTime;
    real Gravity, Density, PoissonsRatio, YoungsModulus;
    real mu, kappa; // Lame
    real IceFrictionCoefficient;

    real IceCompressiveStrength, IceTensileStrength, IceShearStrength;
    real NACC_beta, NACC_M, NACC_Msq;     // these are all computed
    real DP_tan_phi;

    real cellsize, cellsize_inv, Dp_inv;

    int UpdateEveryNthStep; // run N steps without update

    real IndDiameter, IndRSq, IndVelocity, IndDepth;
    real IceBlockDimX, IceBlockDimY, IceBlockDimZ;
    int HoldBlockOnTheRight;

    real ParticleVolume, ParticleMass, ParticleViewSize;

    int SimulationStep;
    real SimulationTime;

    real indenter_x, indenter_x_initial, indenter_y;

    void Reset();
    std::string ParseFile(std::string fileName);

    void ComputeLame();
    void ComputeCamClayParams2();
    void ComputeHelperVariables();
};

#endif
