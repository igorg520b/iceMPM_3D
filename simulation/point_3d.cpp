#include "point_3d.h"

#include <Eigen/SVD>
#include <Eigen/LU>
#include <Eigen/Core>


void icy::Point3D::Reset()
{
    pos = pos_initial;
    Fe.setIdentity();
    velocity.setZero();
    Bp.setZero();
    q = 0;
    Jp_inv = 1;
}

void icy::Point3D::TransferToBuffer(real *buffer, const int pitch, const int point_index) const
{
    char* ptr_intact = (char*)(&buffer[pitch*icy::SimParams3D::idx_intact]);
    ptr_intact[point_index] = q;

    buffer[point_index + pitch*icy::SimParams3D::idx_Jp_inv] = Jp_inv;

    for(int i=0; i<3; i++)
    {
        buffer[point_index + pitch*(icy::SimParams3D::posx+i)] = pos[i];
        buffer[point_index + pitch*(icy::SimParams3D::velx+i)] = velocity[i];
        for(int j=0; i<3; j++)
        {
            buffer[point_index + pitch*(icy::SimParams3D::Fe00 + i*3 + j)] = Fe(i,j);
            buffer[point_index + pitch*(icy::SimParams3D::Bp00 + i*3 + j)] = Bp(i,j);
        }
    }
}

void icy::Point3D::PullFromBuffer(const real *buffer, const int pitch, const int point_index)
{
    char* ptr_intact = (char*)(&buffer[pitch*icy::SimParams3D::idx_intact]);
    q = ptr_intact[point_index];
    Jp_inv = buffer[point_index + pitch*icy::SimParams3D::idx_Jp_inv];

    for(int i=0; i<3; i++)
    {
        pos[i] = buffer[point_index + pitch*(icy::SimParams3D::posx+i)];
        velocity[i] = buffer[point_index + pitch*(icy::SimParams3D::velx+i)];
        for(int j=0; i<3; j++)
        {
            Fe(i,j) = buffer[point_index + pitch*(icy::SimParams3D::Fe00 + i*3 + j)];
            Bp(i,j) = buffer[point_index + pitch*(icy::SimParams3D::Bp00 + i*3 + j)];
        }
    }
}
