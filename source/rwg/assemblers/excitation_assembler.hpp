// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Classes for assembling excitation vectors for RWG-based BEM systems.
*/

#ifndef BEM_RWG_EXC_ASSEMBLER_H
#define BEM_RWG_EXC_ASSEMBLER_H

#include "types.hpp"
#include "geometry/mesh/triangle_mesh.hpp"
#include "rwg/assemblers/base.hpp"


namespace bem
{
// Forward declarations
template <typename T> class MatrixBase;
}


namespace bem::rwg
{

// Forward declarations
class ExcitationBase;

/**
* \addtogroup assm
* @{
*/

/**
* @brief Class for generating excitation matrices for RWG-based systems.
*/
class ExcitationAssembler: public ExcitationAssemblerBase
{

    using base = ExcitationAssemblerBase;

public:

    /**
    * @brief Constructs an `ExcitationAssembler` for a given mesh on given test elements.
    * @param[in] mesh - Triangle mesh for which the excitation matrix is to be assembled.
    * @param[in] elems - Triangle index pairs on which to test the incident field.
    */
    ExcitationAssembler(
        const TriangleMesh<3>& mesh,
        EigRowVec<Index> elems = EigRowVec<Index>::Zero(1, 0)
        ):
        mesh_(mesh),
        elems_(elems)
    {
        if (elems_.cols() == 0)
            elems_ = EigRowVec<Index>::LinSpaced(mesh_.num_elems(), 0, mesh_.num_elems() - 1);
        return;
    };


    /**
    * @brief Assembles the excitation matrix for a given excitation object and observation triangle mesh.
    * @param[out] mat - Matrix to store the assembled excitation coefficients, with columns corresponding
    * to each right-hand side, and rows corresponding to observation mesh edges.
    * @param[in] exc - Excitation object that computes the coefficients to assemble into `mat`.
    * @param[in] k - Complex wavenumber.
    */
    void assemble(
        MatrixBase<Complex>& mat,
        ExcitationBase& exc,
        const Complex k
        ) override;


    /**
    * @brief Assembles the shape derivative of the excitation matrix for a given vertex velocity field.
    * @param[out] mat - Matrix to store \f$ \partial_s \mathbf{b} \f$, with the same layout as in `assemble()`.
    * @param[in] exc - Excitation object; must implement `ExcitationBase::compute_shape()`.
    * @param[in] k - Complex wavenumber.
    * @param[in] velocity - Velocity of each mesh vertex (3 x number of vertices).
    */
    void assemble_shape_derivative(
        MatrixBase<Complex>& mat,
        ExcitationBase& exc,
        const Complex k,
        ConstEigRef<EigMatNX<Float, 3>> velocity
        );


    /**
    * @brief Computes the shape sensitivity \f$ \sum_{n,j} y_{nj}\,\partial b_{nj}/\partial\vec{r}_v \f$
    * with respect to every mesh vertex \f$ \vec{r}_v \f$.
    * @param[in] exc - Excitation object; must implement `ExcitationBase::compute_shape()`.
    * @param[in] k - Complex wavenumber.
    * @param[in] y - Weights, one row per mesh edge and one column per excitation.
    * @return Sensitivity with respect to the coordinates of each vertex (3 x number of vertices).
    */
    EigMatNX<Complex, 3> shape_sensitivity(
        ExcitationBase& exc,
        const Complex k,
        ConstEigRef<EigMat<Complex>> y
        );


protected:

    const TriangleMesh<3>& mesh_;
    EigRowVec<Index> elems_;

};

/**
* @}
*/

}

#ifndef BEM_LINKED
#include "rwg/assemblers/excitation_assembler.cpp"
#endif

#endif
