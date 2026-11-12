// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Class for assembling discrete operators for the TEFIE.
*/

#ifndef BEM_RWG_INTEQ_TEFIE_H
#define BEM_RWG_INTEQ_TEFIE_H

#include <vector>
#include <algorithm>

#include "types.hpp"
#include "constants.hpp"
#include "materials.hpp"

#include "geometry/point_cloud.hpp"

#include "matrix/base.hpp"
#include "matrix/eigen_matrix.hpp"

#include "rwg/function_space.hpp"

#include "rwg/integral_equations/base.hpp"

#include "rwg/operators/single_layer.hpp"
#include "rwg/operators/double_layer.hpp"
#include "rwg/operators/gram.hpp"

#include "rwg/projectors/single_layer.hpp"
#include "rwg/projectors/double_layer.hpp"

#include "rwg/excitations/base.hpp"

#include "rwg/assemblers/operator_assembler.hpp"
#include "rwg/assemblers/excitation_assembler.hpp"
#include "rwg/assemblers/projector_assembler.hpp"


namespace bem::rwg
{

/**
* \addtogroup rwgie
* @{
*/

/**
* @brief Class defining the RWG-based TEFIE.
* @tparam MatrixType - Matrix type used for all operators, must derive from `MatrixBase<Complex>`.
*/
template <typename MatrixType = EigenMatrix<Complex>>
class Tefie: public IntegralEquationBase<MatrixType>
{

    using base = IntegralEquationBase<MatrixType>;
    using base::base;

public:

    /**
    * @brief Sets custom operators for the `Tefie` object.
    * @param[in] op_T - Object for the vector hypersingular operator.
    * @param[in] op_K - Object for the vector double-layer potential operator.
    */
    void set_operators(
        const VectorHypersingularOp& op_T,
        const VectorDoubleLayerPvOp& op_K
        )
    {
        op_T_ = op_T;
        op_K_ = op_K;
        return;
    };


    /**
    * @brief Returns the operator matrix associated with the electric surface current density.
    * @param[in] f - Frequency in Hz.
    * @param[in] material - Material for which the equation is defined.
    * @return Operator matrix.
    */
    MatrixType j_matrix(
        const Float f,
        const Material& material
        )
    {
        Complex k = material.k(f);
        Complex mu = material.mu();

        MatrixType T;
        assembler_.assemble(T, op_T_, k);

        T.scale(-J * two_pi * f * mu);

        return T;
    };


    /**
    * @brief Returns the operator matrix associated with the magnetic surface current density.
    * @param[in] f - Frequency in Hz.
    * @param[in] material - Material for which the equation is defined.
    * @return Operator matrix.
    */
    MatrixType m_matrix(
        const Float f,
        const Material& material
        )
    {
        Complex k = material.k(f);

        MatrixType K;
        assembler_.assemble(K, op_K_, k);

        K.scale(-one);
        K.add_ax(id_matrix(), -half);

        return K;
    };


    /**
    * @brief Returns the identity operator matrix associated with the magnetic surface current density.
    * @return Operator matrix.
    */
    MatrixType id_matrix()
    {
        RotVectorIdentityOp op_Ir;
        MatrixType Ir;
        assembler_.assemble(Ir, op_Ir, 0);
        if (base::flip_normals_)
            Ir.scale(-one);
        return Ir;
    };


    /**
    * @brief Returns the excitation matrix for a given excitation operator.
    * @param[in] f - Frequency in Hz.
    * @param[in] material - Material for which the equation is defined.
    * @param[in] exc - Excitation operator.
    * @return Excitation matrix.
    */
    MatrixType exc_matrix(
        const Float f,
        const Material& material,
        ExcitationBase& exc
        )
    {
        std::vector<Index> obs_elems_vec (base::elem_pairs_.cols());
        EigRowVec<Index>::Map(&obs_elems_vec[0], base::elem_pairs_.row(0).size()) = base::elem_pairs_.row(0);

        std::sort(obs_elems_vec.begin(), obs_elems_vec.end());
        obs_elems_vec.erase(std::unique(obs_elems_vec.begin(), obs_elems_vec.end()), obs_elems_vec.end());

        EigRowVec<Index> obs_elems = EigRowVec<Index>::Map(&obs_elems_vec[0], obs_elems_vec.size());
        ExcitationAssembler exc_assembler (base::obs_mesh_, obs_elems);

        Complex k = material.k(f);
        MatrixType inc (base::obs_mesh_.num_edges(), exc.num_excitations());
        exc_assembler.assemble(inc, exc, k);

        return inc;
    };


    /**
    * @brief Returns the projector matrix associated with the electric surface current density.
    * @param[in] f - Frequency in Hz.
    * @param[in] material - Material for which the equation is defined.
    * @param[in] points - Points at which to project fields.
    * @return Projector matrix.
    */
    MatrixType j_projector(
        const Float f,
        const Material& material,
        const PointCloud<3>& points
        )
    {
        Complex k = material.k(f);
        Complex mu = material.mu();

        ProjectorAssembler<3> proj_assembler (points, base::src_mesh_);

        MatrixType T;
        proj_assembler.assemble(T, proj_T_, k);

        T.scale(-J * two_pi * f * mu);

        return T;
    };


    /**
    * @brief Returns the projector matrix associated with the magnetic surface current density.
    * @param[in] f - Frequency in Hz.
    * @param[in] material - Material for which the equation is defined.
    * @param[in] points - Points at which to project fields.
    * @return Projector matrix.
    */
    MatrixType m_projector(
        const Float f,
        const Material& material,
        const PointCloud<3>& points
        )
    {
        Complex k = material.k(f);

        ProjectorAssembler<3> proj_assembler (points, base::src_mesh_);

        MatrixType K;
        proj_assembler.assemble(K, proj_K_, k);

        K.scale(-one);

        return K;
    };


    /** @name Shape derivatives. */
    /**@{*/

    /**
    * @brief Returns the shape derivative of the operator matrix associated with the electric
    * surface current density, for a given vertex velocity field.
    * @param[in] f - Frequency in Hz.
    * @param[in] material - Material for which the equation is defined.
    * @param[in] velocity - Velocity of each mesh vertex (3 x number of vertices). The mesh is
    * deformed as \f$ F_s(\vec{r}) = \vec{r} + s\vec{V}(\vec{r}) \f$, with \f$ \vec{V} \f$ interpolated
    * linearly from the vertex velocities.
    * @return Shape derivative \f$ \partial_s \mathbf{Z} \f$ of `j_matrix()`.
    * @details Uses the analytical formulas of Kataja and Toivanen, Eng. Anal. Bound. Elem. 37 (2013),
    * implemented in `VectorHypersingularShapeOp`. The observation and source meshes must be the same.
    */
    MatrixType j_matrix_shape_derivative(
        const Float f,
        const Material& material,
        ConstEigRef<EigMatNX<Float, 3>> velocity
        )
    {
        Complex k = material.k(f);
        Complex mu = material.mu();

        MatrixType dT;
        assembler_.assemble_shape_derivative(dT, op_T_shape_, k, velocity);

        dT.scale(-J * two_pi * f * mu);

        return dT;
    };


    /**
    * @brief Returns the shape sensitivity \f$ \mathbf{y}^T(\partial\mathbf{Z}/\partial\vec{r}_v)\,\mathbf{x} \f$
    * of the operator matrix associated with the electric surface current density, with respect to
    * the coordinates of every mesh vertex \f$ \vec{r}_v \f$.
    * @param[in] f - Frequency in Hz.
    * @param[in] material - Material for which the equation is defined.
    * @param[in] x - Column vector multiplying the matrix from the right (e.g. the current coefficients).
    * @param[in] y - Column vector multiplying the matrix from the left (e.g. the adjoint solution).
    * @return Sensitivities (3 x number of vertices).
    * @details This is the matrix term of the adjoint variable method; no shape-derivative matrix is formed.
    */
    EigMatNX<Complex, 3> j_matrix_shape_sensitivity(
        const Float f,
        const Material& material,
        const MatrixType& x,
        const MatrixType& y
        )
    {
        Complex k = material.k(f);
        Complex mu = material.mu();

        EigMatNX<Complex, 3> sens = assembler_.shape_sensitivity(
            op_T_shape_, k, column_vector(x), column_vector(y)
            );

        return sens * (-J * two_pi * f * mu);
    };


    /**
    * @brief Returns the shape derivative of the excitation matrix for a given vertex velocity field.
    * @param[in] f - Frequency in Hz.
    * @param[in] material - Material for which the equation is defined.
    * @param[in] exc - Excitation operator; must implement `ExcitationBase::compute_shape()`.
    * @param[in] velocity - Velocity of each mesh vertex (3 x number of vertices).
    * @return Shape derivative \f$ \partial_s \mathbf{b} \f$ of `exc_matrix()`.
    */
    MatrixType exc_matrix_shape_derivative(
        const Float f,
        const Material& material,
        ExcitationBase& exc,
        ConstEigRef<EigMatNX<Float, 3>> velocity
        )
    {
        ExcitationAssembler exc_assembler (base::obs_mesh_, observation_elems());

        Complex k = material.k(f);
        MatrixType dinc;
        exc_assembler.assemble_shape_derivative(dinc, exc, k, velocity);

        return dinc;
    };


    /**
    * @brief Returns the shape sensitivity \f$ \mathbf{y}^T(\partial\mathbf{b}/\partial\vec{r}_v) \f$ of the
    * excitation matrix with respect to the coordinates of every mesh vertex \f$ \vec{r}_v \f$.
    * @param[in] f - Frequency in Hz.
    * @param[in] material - Material for which the equation is defined.
    * @param[in] exc - Excitation operator; must implement `ExcitationBase::compute_shape()`.
    * @param[in] y - Weights, with the same size as the excitation matrix.
    * @return Sensitivities (3 x number of vertices).
    */
    EigMatNX<Complex, 3> exc_matrix_shape_sensitivity(
        const Float f,
        const Material& material,
        ExcitationBase& exc,
        const MatrixType& y
        )
    {
        ExcitationAssembler exc_assembler (base::obs_mesh_, observation_elems());

        EigMat<Complex> weights (y.num_rows(), y.num_cols());
        for (Index row = 0; row < y.num_rows(); ++row)
            for (Index col = 0; col < y.num_cols(); ++col)
                weights(row, col) = y.value(row, col);

        Complex k = material.k(f);
        return exc_assembler.shape_sensitivity(exc, k, weights);
    };


    /**
    * @brief Returns the shape derivative of the projector matrix associated with the electric
    * surface current density, for a given vertex velocity field.
    * @param[in] f - Frequency in Hz.
    * @param[in] material - Material for which the equation is defined.
    * @param[in] points - Points at which to project fields; must not lie close to the mesh.
    * @param[in] velocity - Velocity of each mesh vertex (3 x number of vertices).
    * @return Shape derivative of `j_projector()`.
    */
    MatrixType j_projector_shape_derivative(
        const Float f,
        const Material& material,
        const PointCloud<3>& points,
        ConstEigRef<EigMatNX<Float, 3>> velocity
        )
    {
        Complex k = material.k(f);
        Complex mu = material.mu();

        ProjectorAssembler<3> proj_assembler (points, base::src_mesh_);

        MatrixType dT;
        proj_assembler.assemble_shape_derivative(dT, proj_T_, k, velocity);

        dT.scale(-J * two_pi * f * mu);

        return dT;
    };


    /**
    * @brief Returns the shape sensitivity \f$ \mathbf{y}^T(\partial\mathbf{P}/\partial\vec{r}_v)\,\mathbf{x} \f$
    * of the projector matrix associated with the electric surface current density, with respect to
    * the coordinates of every mesh vertex \f$ \vec{r}_v \f$.
    * @param[in] f - Frequency in Hz.
    * @param[in] material - Material for which the equation is defined.
    * @param[in] points - Points at which to project fields; must not lie close to the mesh.
    * @param[in] x - Column vector of current coefficients.
    * @param[in] y - Column vector of weights for the projected field components.
    * @return Sensitivities (3 x number of vertices).
    */
    EigMatNX<Complex, 3> j_projector_shape_sensitivity(
        const Float f,
        const Material& material,
        const PointCloud<3>& points,
        const MatrixType& x,
        const MatrixType& y
        )
    {
        Complex k = material.k(f);
        Complex mu = material.mu();

        ProjectorAssembler<3> proj_assembler (points, base::src_mesh_);

        EigMatNX<Complex, 3> sens = proj_assembler.shape_sensitivity(
            proj_T_, k, column_vector(x), column_vector(y)
            );

        return sens * (-J * two_pi * f * mu);
    };

    /**@}*/


protected:

    /**
    * @brief Returns the observation elements associated with the stored element pairs.
    * @return Sorted, unique observation element indices.
    */
    EigRowVec<Index> observation_elems() const
    {
        std::vector<Index> obs_elems_vec (base::elem_pairs_.cols());
        EigRowVec<Index>::Map(&obs_elems_vec[0], base::elem_pairs_.row(0).size()) = base::elem_pairs_.row(0);

        std::sort(obs_elems_vec.begin(), obs_elems_vec.end());
        obs_elems_vec.erase(std::unique(obs_elems_vec.begin(), obs_elems_vec.end()), obs_elems_vec.end());

        return EigRowVec<Index>::Map(&obs_elems_vec[0], obs_elems_vec.size());
    };


    /**
    * @brief Copies the first column of a matrix into an Eigen column vector.
    * @param[in] mat - Matrix to copy.
    * @return Column vector.
    */
    static EigColVec<Complex> column_vector(const MatrixType& mat)
    {
        EigColVec<Complex> vec (mat.num_rows());
        for (Index row = 0; row < mat.num_rows(); ++row)
            vec[row] = mat.value(row, 0);
        return vec;
    };


    VectorHypersingularOp op_T_;
    VectorDoubleLayerPvOp op_K_;

    VectorHypersingularShapeOp op_T_shape_;

    VectorHypersingularProj proj_T_;
    VectorDoubleLayerProj proj_K_;

    OperatorAssembler assembler_ = OperatorAssembler (base::obs_mesh_, base::src_mesh_, base::elem_pairs_);

};

/**
* @}
*/

}

#endif
