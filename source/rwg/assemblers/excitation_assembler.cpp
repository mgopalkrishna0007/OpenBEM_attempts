// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Classes for assembling excitation vectors and matrices for RWG-based BEM systems.
*/

#include "rwg/assemblers/excitation_assembler.hpp"

#include <stdexcept>

#include "types.hpp"
#include "geometry/mesh/triangle_mesh.hpp"
#include "geometry/primitives/triangle.hpp"
#include "matrix/base.hpp"


namespace bem::rwg
{

void ExcitationAssembler::assemble(
    MatrixBase<Complex>& mat,
    ExcitationBase& exc,
    const Complex k
    )
{

    if (exc.obs_dof() == 3)
    {

        Index obs_num_edges = mesh_.num_edges();

        mat.resize(obs_num_edges, exc.num_excitations());
        mat.preallocate(obs_num_edges * exc.num_excitations());

        for (Index ii = 0; ii < elems_.size(); ++ii)
        {
            Index face = elems_[ii];
            Triangle<3> obs_tri = mesh_.elem_primitive(face);

            EigMatNX<Complex, 3> values = exc.compute(k, obs_tri);

            for (Index col = 0; col < values.cols(); col++)
            {
                for (uint8_t edge = 0; edge < 3; edge++)
                {
                    Index row = mesh_.elem_edges()(edge, face);
                    mat.add_value(row, col, values(edge, col));
                }
            }
        }

    }

    else if (exc.obs_dof() == 1)
    {

        Index obs_num_faces = mesh_.num_elems();

        mat.resize(obs_num_faces, exc.num_excitations());
        mat.preallocate(obs_num_faces * exc.num_excitations());

        for (Index ii = 0; ii < elems_.size(); ++ii)
        {
            Index face = elems_[ii];
            Triangle<3> obs_tri = mesh_.elem_primitive(face);

            EigMatNX<Complex, 1> values = exc.compute(k, obs_tri);

            for (Index rhs = 0; rhs < values.cols(); rhs++)
                mat.set_value(face, rhs, values(0, rhs));
        }

    }

    mat.assemble();

    return;

};


void ExcitationAssembler::assemble_shape_derivative(
    MatrixBase<Complex>& mat,
    ExcitationBase& exc,
    const Complex k,
    ConstEigRef<EigMatNX<Float, 3>> velocity
    )
{

    if (exc.obs_dof() != OperatorDof::EDGE)
        throw std::invalid_argument(
            "ExcitationAssembler::assemble_shape_derivative(): only edge degrees of freedom are supported."
            );

    if (velocity.cols() != mesh_.num_verts())
        throw std::invalid_argument(
            "ExcitationAssembler::assemble_shape_derivative(): velocity size does not match the mesh."
            );

    mat.resize(mesh_.num_edges(), exc.num_excitations());
    mat.preallocate(mesh_.num_edges() * exc.num_excitations());

    for (Index ii = 0; ii < elems_.size(); ++ii)
    {
        Index face = elems_[ii];
        Triangle<3> obs_tri = mesh_.elem_primitive(face);

        const EigMat<Complex> derivs = exc.compute_shape(k, obs_tri);

        EigColVecN<Float, 9> vel;
        for (uint8_t vert = 0; vert < 3; ++vert)
            vel.segment<3>(3 * vert) = velocity.col(mesh_.elems(vert, face));

        for (Index rhs = 0; rhs < exc.num_excitations(); ++rhs)
        {
            const EigColVecN<Complex, 3> values = derivs.middleCols(9 * rhs, 9) * vel.cast<Complex>();
            for (uint8_t edge = 0; edge < 3; ++edge)
                mat.add_value(mesh_.elem_edges()(edge, face), rhs, values[edge]);
        }
    }

    mat.assemble();

    return;

};


EigMatNX<Complex, 3> ExcitationAssembler::shape_sensitivity(
    ExcitationBase& exc,
    const Complex k,
    ConstEigRef<EigMat<Complex>> y
    )
{

    if (exc.obs_dof() != OperatorDof::EDGE)
        throw std::invalid_argument(
            "ExcitationAssembler::shape_sensitivity(): only edge degrees of freedom are supported."
            );

    EigMatNX<Complex, 3> sens = EigMatNX<Complex, 3>::Zero(3, mesh_.num_verts());

    for (Index ii = 0; ii < elems_.size(); ++ii)
    {
        Index face = elems_[ii];
        Triangle<3> obs_tri = mesh_.elem_primitive(face);

        const EigMat<Complex> derivs = exc.compute_shape(k, obs_tri);

        for (Index rhs = 0; rhs < exc.num_excitations(); ++rhs)
        {
            EigRowVecN<Complex, 3> weights;
            for (uint8_t edge = 0; edge < 3; ++edge)
                weights[edge] = y(mesh_.elem_edges()(edge, face), rhs);

            const EigRowVecN<Complex, 9> contrib = weights * derivs.middleCols(9 * rhs, 9);
            for (uint8_t vert = 0; vert < 3; ++vert)
                sens.col(mesh_.elems(vert, face)) += contrib.segment<3>(3 * vert).transpose();
        }
    }

    return sens;

};

}
