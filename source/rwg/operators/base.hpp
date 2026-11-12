// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Base classes for RWG-based BEM operators.
*/

#ifndef BEM_RWG_OPS_BASE_H
#define BEM_RWG_OPS_BASE_H

#include <memory>
#include <stdexcept>

#include "types.hpp"
#include "geometry/operations.hpp"
#include "geometry/primitives/triangle.hpp"
#include "rwg/function_space.hpp"
#include "rwg/integrators/obs/base.hpp"


namespace bem::rwg
{

/**
* \ingroup rwgops
* @{
*/

/**
* @brief Base class for RWG-based BEM operators.
*/
class OperatorBase
{
public:

    /**
    * @brief Returns the degrees of freedom for the testing function space.
    * @return Observation degrees of freedom.
    */
    virtual OperatorDof obs_dof() const = 0;


    /**
    * @brief Returns the degrees of freedom for the expansion function space.
    * @return Source degrees of freedom.
    */
    virtual OperatorDof src_dof() const = 0;


    /**
    * @brief Computes operator values for the given observation and source triangles.
    * @param[in] k - Complex wavenumber.
    * @param[in] obs_tri - Observation triangle.
    * @param[in] src_tri - Source triangle.
    * @return Operator values for each pair of observation and source degrees of freedom.
    * @details
    * Rows of the output matrix correspond to observation degrees of freedom, and columns
    * correspond to source degrees of freedom.
    */
    virtual EigMat<Complex> compute(
        const Complex k,
        const Triangle<3>& obs_tri,
        const Triangle<3>& src_tri
        ) const = 0;


    /**
    * @brief Assembles the computed integrals into the final operator values.
    * @param[in] k - Complex wavenumber.
    * @param[in] obs_tri - Observation triangle.
    * @param[in] src_tri - Source triangle.
    * @param[in] obs_result - Integration result.
    * @return Operator values for each combination of degrees of freedom.
    */
    virtual EigMat<Complex> assemble(
        const Complex k,
        const Triangle<3>& obs_tri,
        const Triangle<3>& src_tri,
        const ObsResult& obs_result
        ) const = 0;


    /**
    * @brief Transforms the coordinates of the observation and source triangles into a local
    * coordinate system defined by the source triangle.
    * @param[out] obs_tri_local - Observation triangle in the source's local coordinate system.
    * @param[out] src_tri_local - Source triangle in its local coordinate system.
    * @param[in] obs_tri - Original observation triangle.
    * @param[in] src_tri - Original source triangle.
    */
    static void transform_coordinates(
        Triangle<3>& obs_tri_local,
        Triangle<2>& src_tri_local,
        const Triangle<3>& obs_tri,
        const Triangle<3>& src_tri
        )
    {
        EigMatMN<Float, 3, 3> local_uvw = src_tri.local_coordinate_basis();
        EigColVecN<Float, 3> local_origin = src_tri.local_origin();

        obs_tri_local.set_v(
            GeometryOps<3>::transform_coordinate_system(
                obs_tri.v(), local_origin, local_uvw
                ),
            obs_tri.edge_polarities()
        );
        src_tri_local.set_v(
            GeometryOps<3>::transform_coordinate_system(
                src_tri.v(), local_origin, local_uvw
                ).topRows(2),
            src_tri.edge_polarities()
        );

        return;
    };


    /**
    * @brief Returns a unique pointer to a newly constructed object of the derived type.
    * @return Unique pointer to the new object.
    */
    virtual std::unique_ptr<OperatorBase> clone() const
    { throw std::runtime_error("OperatorBase::clone(): Not implemented."); };


    /**
    * @brief Virtual destructor.
    */
    virtual ~OperatorBase() = default;

};

/**
* @brief Base class for shape derivatives of RWG-based BEM operators.
* @details
* A shape operator returns the derivatives of the element-pair matrix of an operator with
* respect to the coordinates of the six vertices of the observation and source triangles.
* For a pair with \f$ n_o \f$ observation and \f$ n_s \f$ source degrees of freedom, the output
* has \f$ n_o n_s \f$ rows, ordered as `obs_dof_index * n_s + src_dof_index`, and 18 columns,
* ordered as `3 * vertex + component`, where vertices 0-2 are the vertices of the observation
* triangle and 3-5 those of the source triangle. If both triangles share a mesh vertex, the
* derivative with respect to that vertex is the sum of the two corresponding columns.
*
* The derivatives are those of the deformation \f$ F_s(\vec{r}) = \vec{r} + s\,\vec{\tau}\,
* \lambda_m(\vec{r}) \f$ (Kataja and Toivanen, Eng. Anal. Bound. Elem. 37, 2013), in which the RWG
* functions follow the mesh through the Piola transform.
*/
class ShapeOperatorBase
{
public:

    /**
    * @brief Returns the degrees of freedom for the testing function space.
    * @return Observation degrees of freedom.
    */
    virtual OperatorDof obs_dof() const = 0;


    /**
    * @brief Returns the degrees of freedom for the expansion function space.
    * @return Source degrees of freedom.
    */
    virtual OperatorDof src_dof() const = 0;


    /**
    * @brief Computes the shape derivatives for the given observation and source triangles.
    * @param[in] k - Complex wavenumber.
    * @param[in] obs_tri - Observation triangle.
    * @param[in] src_tri - Source triangle.
    * @return Shape derivatives in the global coordinate system, laid out as described above.
    */
    virtual EigMat<Complex> compute(
        const Complex k,
        const Triangle<3>& obs_tri,
        const Triangle<3>& src_tri
        ) const = 0;


    /**
    * @brief Assembles the computed integrals into the shape derivatives.
    * @param[in] k - Complex wavenumber.
    * @param[in] obs_tri - Observation triangle in the source's local coordinate system.
    * @param[in] src_tri - Source triangle in its local coordinate system.
    * @param[in] obs_result - Integration result, including `ObsResult::grad_g_shape`.
    * @return Shape derivatives in the source's local coordinate system.
    */
    virtual EigMat<Complex> assemble(
        const Complex k,
        const Triangle<3>& obs_tri,
        const Triangle<3>& src_tri,
        const ObsResult& obs_result
        ) const = 0;


    /**
    * @brief Returns a unique pointer to a newly constructed object of the derived type.
    * @return Unique pointer to the new object.
    */
    virtual std::unique_ptr<ShapeOperatorBase> clone() const
    { throw std::runtime_error("ShapeOperatorBase::clone(): Not implemented."); };


    /**
    * @brief Virtual destructor.
    */
    virtual ~ShapeOperatorBase() = default;

};

/**
* @}
*/

}

#endif
