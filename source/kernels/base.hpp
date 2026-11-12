// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Base classes for BEM operator kernels.
*/

#ifndef BEM_KERNEL_BASE_H
#define BEM_KERNEL_BASE_H

#include <cassert>
#include <stdexcept>

#include "types.hpp"
#include "constants.hpp"


namespace bem
{

const Float KERNEL_DEFAULT_TOL = 1.0e-6;

/**
* \addtogroup kern
* @{
*/

/**
* @brief Base class for scalar kernels.
* @tparam dim - Dimension of the kernel (2 or 3).
*/
template <uint8_t dim>
class ScalarKernelBase
{

    static_assert((dim == 2 || dim == 3), "`dim` must be 2 or 3.");

public:

    /**
    * @brief Computes the scalar kernel for given observation and source points.
    * @param[in] r_obs - Observer position vector.
    * @param[in] r_src - Set of source position vectors.
    * @param[in] k - Complex wavenumber.
    * @return Scalar kernel value.
    */
    virtual EigRowVec<Complex> compute(
        ConstEigRef<EigColVecN<Float, dim>> r_obs,
        ConstEigRef<EigMatNX<Float, dim>> r_src,
        const Complex k
        ) const = 0;


    /**
    * @brief Computes the gradient of the scalar kernel for given observation and source points.
    * @param[in] r_obs - Observer position vector.
    * @param[in] r_src - Set of source position vectors.
    * @param[in] k - Complex wavenumber.
    * @return Components of the gradient of the scalar kernel.
    */
    virtual EigMatNX<Complex, dim> compute_grad(
        ConstEigRef<EigColVecN<Float, dim>> r_obs,
        ConstEigRef<EigMatNX<Float, dim>> r_src,
        const Complex k
        ) const = 0;


    /**
    * @brief Computes the Hessian of the scalar kernel for given observation and source points.
    * @param[in] r_obs - Observer position vector.
    * @param[in] r_src - Set of source position vectors.
    * @param[in] k - Complex wavenumber.
    * @return Unique components of the Hessian of the scalar kernel, ordered as
    * \f$ (xx, xy, xz, yy, yz, zz) \f$ in 3D or \f$ (xx, xy, yy) \f$ in 2D.
    * @note Used for shape derivatives of field projections. Kernels that do not need it may leave
    * the default implementation, which throws.
    */
    virtual EigMatNX<Complex, dim * (dim + 1) / 2> compute_hess(
        ConstEigRef<EigColVecN<Float, dim>> r_obs,
        ConstEigRef<EigMatNX<Float, dim>> r_src,
        const Complex k
        ) const
    { throw std::runtime_error("ScalarKernelBase::compute_hess(): Not implemented."); };


    /**
    * @brief Virtual destructor.
    */
    virtual ~ScalarKernelBase() = default;

};

/**
* @}
*/

}

#endif
