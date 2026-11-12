// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* 2D quadrature over the source triangle with singularity treatment for RWG-based BEM operators.
*/

#ifndef BEM_RWG_OPINT_SRC_SINGULARITY_H
#define BEM_RWG_OPINT_SRC_SINGULARITY_H

#include "types.hpp"
#include "geometry/primitives/triangle.hpp"
#include "quadrature/triangle/base.hpp"
#include "quadrature/triangle/gauss.hpp"

#include "kernels/base.hpp"
#include "kernels/hgf.hpp"
#include "rwg/integrators/src/base.hpp"
#include "rwg/integrators/src/quadrature.hpp"


namespace bem::rwg
{

/**
* \addtogroup rwgopintsrc
* @{
*/

/**
* @brief Class for 2D quadrature over the source triangle with singularity treatment for
* RWG-based BEM operators.
* Reference:
* - [1] O. Ergul, L. Gurel, "The Multilevel Fast Multipole Algorithm (MLFMA) for Solving Large-Scale
* Computational Electromagnetics Problems," book, Wiley-IEEE Press, 2014.
*/
class SrcSingularity: public SrcIntegratorBase
{

    using base = SrcIntegratorBase;

public:

    /**
    * @brief Constructs a `SrcSingularity` with a specified triangle quadrature object.
    * @tparam TriangleQuadratureType - Type of the triangle quadrature object, which must
    * derive from `TriangleQuadratureBase<2>`.
    * @tparam ScalarKernelType - Object for computing the scalar kernel with its singularity
    * subtracted, which must derive from `ScalarKernelBase<3>`.
    * @param[in] tri_quad - Triangle quadrature object to use for integration (optional).
    * @param[in] kernel - Object for computing the scalar kernel with its singularity subtracted (optional).
    */
    template <
        typename TriangleQuadratureType = GaussTriangleQuadrature<2>,
        typename ScalarKernelType = SingularitySubtractedTaylorHGF
        >
    SrcSingularity(
        const TriangleQuadratureType tri_quad = GaussTriangleQuadrature<2>(),
        const ScalarKernelType kernel = SingularitySubtractedTaylorHGF()
        ) { set(tri_quad, kernel); return; };


    /**
    * @brief Sets the triangle quadrature object and kernel object.
    * @tparam TriangleQuadratureType - Type of the triangle quadrature object, which must derive
    * from `TriangleQuadratureBase<2>`.
    * @tparam ScalarKernelType - Object for computing the scalar kernel with its singularity
    * subtracted, which must derive from `ScalarKernelBase<3>`.
    * @param[in] tri_quad - Triangle quadrature object to use for integration.
    * @param[in] kernel - Object for computing the scalar kernel with its singularity subtracted.
    */
    template <typename TriangleQuadratureType, typename ScalarKernelType>
    void set(const TriangleQuadratureType& tri_quad, const ScalarKernelType& kernel)
    { src_quad_.set(tri_quad, kernel); return; }


    /**
    * @brief Computes the integral over the source triangle.
    * @param[in] k - Complex wavenumber.
    * @param[in] src_tri - Source triangle in 2D space.
    * @param[in] r_obs - Observation points in the local coordinate system of `src_tri`.
    * @param[in] g_terms - Whether to compute kernel terms (optional).
    * @param[in] grad_g_terms - Whether to compute kernel gradient terms (optional).
    * @param[in] shape_grad_terms - Whether to compute moments of the kernel gradient (optional).
    * @param[in] shape_hess_terms - Whether to compute moments of the kernel Hessian (optional).
    * @return Integration result.
    * @details
    * If `g_terms` is true, the function computes
    * \f[
    * \int_{\mathrm{src\_tri}} d\mathcal{S}'\,G(k, \vec{r}, \vec{r}\,')
    * \f]
    * and
    * \f[
    * \int_{\mathrm{src\_tri}} d\mathcal{S}'\,\vec{r}\,'\,G(k, \vec{r}, \vec{r}\,')
    * \f]
    * for the scalar kernel \f$ G(k, \vec{r}, \vec{r}\,') \f$.
    * If `grad_g_terms` is true, the function computes
    * \f[
    * \int_{\mathrm{src\_tri}} d\mathcal{S}'\,\nabla G(k, \vec{r}, \vec{r}\,'),
    * \f]
    * \f[
    * \int_{\mathrm{src\_tri}} d\mathcal{S}'\,x'\,\nabla G(k, \vec{r}, \vec{r}\,'),
    * \f]
    * and
    * \f[
    * \int_{\mathrm{src\_tri}} d\mathcal{S}'\,y'\,\nabla G(k, \vec{r}, \vec{r}\,'),
    * \f]
    * for the scalar kernel \f$ G(k, \vec{r}, \vec{r}\,') \f$, and for local source
    * triangle coordinates \f$\vec{r}\,' = [x', y']^T\f$.
    */
    SrcResult integrate(
        const Complex k,
        const Triangle<2>& src_tri,
        ConstEigRef<EigMatNX<Float, 3>> r_obs,
        const bool g_terms = true,
        const bool grad_g_terms = true,
        const bool shape_grad_terms = false,
        const bool shape_hess_terms = false
        ) override;


protected:

    /**
    * @brief Computes the singular integrals.
    * @param[in] k - Complex wavenumber.
    * @param[in] src_tri - Source triangle in 2D space.
    * @param[in] r_obs - Observation points in the local coordinate system of `src_tri`.
    * @param[in] g_terms - Whether to compute kernel terms (optional).
    * @param[in] grad_g_terms - Whether to compute kernel gradient terms (optional).
    * @return Integration result.
    */
    SrcResult integrate_singular(
        const Complex k,
        const Triangle<2>& src_tri,
        ConstEigRef<EigMatNX<Float, 3>> r_obs,
        const bool g_terms = true,
        const bool grad_g_terms = true
        );


    /**
    * @brief Computes the singular parts of the moments of the kernel gradient analytically.
    * @param[in] k - Complex wavenumber.
    * @param[in] src_tri - Source triangle in 2D space.
    * @param[in] r_obs - Observation points in the local coordinate system of `src_tri`.
    * @param[out] rs_grad_g - Singular part of the first moments (see `SrcResult::rs_grad_g`).
    * @param[out] rsrs_grad_g - Singular part of the second moments (see `SrcResult::rsrs_grad_g`).
    * @details
    * The subtracted kernel is \f$ (1/R - k^2 R/2) / (4\pi) \f$, consistent with
    * `integrate_singular()` and with the `SingularitySubtractedHGF` and
    * `SingularitySubtractedTaylorHGF` kernels used for the remainder. With
    * \f$ \vec{r} = \vec{\rho} + z\hat{z} \f$ and \f$ \vec{p} = \vec{r}\,' - \vec{\rho} \f$, the
    * weighted integrals
    * \f[
    * \int \frac{w\,\vec{p}}{R^3},\quad z\int \frac{w}{R^3},\quad
    * \int \frac{w\,\vec{p}}{R},\quad z\int \frac{w}{R}
    * \f]
    * for polynomial weights \f$ w \f$ of degree up to two are reduced to line integrals over the
    * triangle edges with the surface divergence theorem, using \f$ \nabla' R = \vec{p}/R \f$ and
    * \f$ \nabla'(1/R) = -\vec{p}/R^3 \f$.
    */
    void integrate_singular_shape(
        const Complex k,
        const Triangle<2>& src_tri,
        ConstEigRef<EigMatNX<Float, 3>> r_obs,
        EigMatNX<Complex, 6>& rs_grad_g,
        EigMatNX<Complex, 9>& rsrs_grad_g
        );


    SrcQuadrature src_quad_;

};

/**
* @}
*/

}

#ifndef BEM_LINKED
#include "rwg/integrators/src/singularity.cpp"
#endif

#endif
