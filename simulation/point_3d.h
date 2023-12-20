#ifndef PARTICLE_H
#define PARTICLE_H

#include <cmath>
#include <Eigen/Core>

#include "parameters_sim_3d.h"
#include <cuda.h>
#include <cuda_runtime.h>

namespace icy { struct Point3D; }

struct icy::Point3D
{
    Vector3r pos, velocity;
    Matrix3r Bp, Fe; // refer to "The Material Point Method for Simulating Continuum Materials"

    real Jp_inv; // track the change in det(Fp)

    Vector3r pos_initial; // for resetting
    char q;

    void Reset();
    void TransferToBuffer(real *buffer, const int pitch, const int point_index) const;  // distribute to SOA
    void PullFromBuffer(const real *buffer, const int pitch, const int point_index);

    static Vector3r getPos(const real *buffer, const int pitch, const int point_index);
    static Vector3r getVelocity(const real *buffer, const int pitch, const int point_index);
    static char getQ(const real *buffer, const int pitch, const int point_index);
    static double getJp_inv(const real *buffer, const int pitch, const int point_index);
};


#endif // PARTICLE_H
