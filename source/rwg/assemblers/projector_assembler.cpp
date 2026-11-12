// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Classes for assembling RWG-based BEM projector matrices.
*/

#include "rwg/assemblers/projector_assembler.hpp"

#include <stdexcept>

#include "types.hpp"
#include "geometry/primitives/triangle.hpp"
#include "geometry/point_cloud.hpp"
#include "matrix/base.hpp"


namespace bem::rwg
{

template <uint8_t obs_dim>
void ProjectorAssembler<obs_dim>::assemble(
    MatrixBase<Complex>& mat,
    ProjectorBase& op,
    const Complex k
    )
{

    if (op.src_dof() == 3)
    {

        mat.resize(obs_cloud_.num_points() * obs_dim, mesh_.num_edges());
        mat.preallocate(
            obs_cloud_.num_points() * obs_dim * elems_.size() * EDGE_ELEM_RATIO
            );

        for (Index ii = 0; ii < elems_.size(); ++ii)
        {
            Triangle<3> src_tri = mesh_.elem_primitive(elems_[ii]);

            EigMatXN<Complex, 3> values = op.compute(k, obs_cloud_.points(), src_tri);

            for (uint8_t src_edge = 0; src_edge < 3; ++src_edge)
            {
                Index col = mesh_.elem_edges()(src_edge, elems_[ii]);

                for (Index obs_point = 0; obs_point < obs_cloud_.num_points(); ++obs_point)
                {
                    for (uint8_t obs_dim_idx = 0; obs_dim_idx < obs_dim; ++obs_dim_idx)
                    {
                        Index row = obs_dim_idx + obs_point * obs_dim;
                        mat.add_value(row, col, values(row, src_edge));
                    }
                }
            }
        }

    }

    else if (op.src_dof() == 1)
    {

        mat.resize(obs_cloud_.num_points() * obs_dim, mesh_.num_elems());
        mat.preallocate(obs_cloud_.num_points() * obs_dim * elems_.size());

        for (Index ii = 0; ii < elems_.size(); ++ii)
        {
            Triangle<3> src_tri = mesh_.elem_primitive(elems_[ii]);

            EigMatXN<Complex, 1> values = op.compute(k, obs_cloud_.points(), src_tri);

            for (Index obs_point = 0; obs_point < obs_cloud_.num_points(); ++obs_point)
            {
                for (uint8_t obs_dim_idx = 0; obs_dim_idx < obs_dim; ++obs_dim_idx)
                {
                    Index row = obs_dim_idx + obs_point * obs_dim;
                    mat.add_value(row, elems_[ii], values(row, 0));
                }
            }
        }

    }

    mat.assemble();

    return;

};

template <uint8_t obs_dim>
void ProjectorAssembler<obs_dim>::assemble_shape_derivative(
    MatrixBase<Complex>& mat,
    ProjectorBase& op,
    const Complex k,
    ConstEigRef<EigMatNX<Float, 3>> velocity
    )
{

    if (op.src_dof() != OperatorDof::EDGE)
        throw std::invalid_argument(
            "ProjectorAssembler::assemble_shape_derivative(): only edge degrees of freedom are supported."
            );

    if (velocity.cols() != mesh_.num_verts())
        throw std::invalid_argument(
            "ProjectorAssembler::assemble_shape_derivative(): velocity size does not match the mesh."
            );

    mat.resize(obs_cloud_.num_points() * obs_dim, mesh_.num_edges());
    mat.preallocate(obs_cloud_.num_points() * obs_dim * elems_.size() * EDGE_ELEM_RATIO);

    for (Index ii = 0; ii < elems_.size(); ++ii)
    {
        Triangle<3> src_tri = mesh_.elem_primitive(elems_[ii]);

        const EigMat<Complex> derivs = op.compute_shape(k, obs_cloud_.points(), src_tri);

        EigColVecN<Float, 9> vel;
        for (uint8_t vert = 0; vert < 3; ++vert)
            vel.segment<3>(3 * vert) = velocity.col(mesh_.elems(vert, elems_[ii]));

        for (uint8_t src_edge = 0; src_edge < 3; ++src_edge)
        {
            Index col = mesh_.elem_edges()(src_edge, elems_[ii]);
            const EigColVec<Complex> values = derivs.middleCols(9 * src_edge, 9) * vel.cast<Complex>();

            for (Index row = 0; row < values.size(); ++row)
                mat.add_value(row, col, values[row]);
        }
    }

    mat.assemble();

    return;

};


template <uint8_t obs_dim>
EigMatNX<Complex, 3> ProjectorAssembler<obs_dim>::shape_sensitivity(
    ProjectorBase& op,
    const Complex k,
    ConstEigRef<EigColVec<Complex>> x,
    ConstEigRef<EigColVec<Complex>> y
    )
{

    if (op.src_dof() != OperatorDof::EDGE)
        throw std::invalid_argument(
            "ProjectorAssembler::shape_sensitivity(): only edge degrees of freedom are supported."
            );

    EigMatNX<Complex, 3> sens = EigMatNX<Complex, 3>::Zero(3, mesh_.num_verts());

    for (Index ii = 0; ii < elems_.size(); ++ii)
    {
        Triangle<3> src_tri = mesh_.elem_primitive(elems_[ii]);

        const EigMat<Complex> derivs = op.compute_shape(k, obs_cloud_.points(), src_tri);

        EigRowVecN<Complex, 9> contrib = EigRowVecN<Complex, 9>::Zero();
        for (uint8_t src_edge = 0; src_edge < 3; ++src_edge)
        {
            const Complex x_edge = x[mesh_.elem_edges()(src_edge, elems_[ii])];
            contrib += x_edge * (y.transpose() * derivs.middleCols(9 * src_edge, 9));
        }

        for (uint8_t vert = 0; vert < 3; ++vert)
            sens.col(mesh_.elems(vert, elems_[ii])) += contrib.segment<3>(3 * vert).transpose();
    }

    return sens;

};


template class ProjectorAssembler<1>;
template class ProjectorAssembler<3>;

}

