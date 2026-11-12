// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Base class for integration over the source triangle for RWG-based BEM operators.
*/

#ifndef BEM_RWG_OPINT_SRC_BASE_H
#define BEM_RWG_OPINT_SRC_BASE_H

#include "types.hpp"
#include "geometry/primitives/triangle.hpp"


namespace bem::rwg
{

/**
* \addtogroup rwgopintsrc
* @{
*/

/**
* @brief Data structure to hold the results of integration over the source triangle.
*/
struct SrcResult
{
    EigRowVec<Complex> g;
    EigMatNX<Complex, 2> rs_g;
    EigMatNX<Complex, 3> grad_g;

    /** @name Shape-derivative terms (only filled when requested). */
    /**@{*/

    /// \f$ \int x'\,\nabla G \f$ (rows 0-2) and \f$ \int y'\,\nabla G \f$ (rows 3-5).
    EigMatNX<Complex, 6> rs_grad_g;

    /// \f$ \int x'x'\,\nabla G \f$ (rows 0-2), \f$ \int x'y'\,\nabla G \f$ (rows 3-5) and
    /// \f$ \int y'y'\,\nabla G \f$ (rows 6-8).
    EigMatNX<Complex, 9> rsrs_grad_g;

    /// \f$ \int \nabla\nabla G \f$, components ordered as \f$ (xx, xy, xz, yy, yz, zz) \f$.
    EigMatNX<Complex, 6> hess_g;

    /// \f$ \int x'\,\nabla\nabla G \f$ (rows 0-5) and \f$ \int y'\,\nabla\nabla G \f$ (rows 6-11).
    EigMatNX<Complex, 12> rs_hess_g;

    /**@}*/

    void resize(Index n) {
        g.resize(1, n); rs_g.resize(2, n); grad_g.resize(3, n);
        rs_grad_g.resize(6, n); rsrs_grad_g.resize(9, n);
        hess_g.resize(6, n); rs_hess_g.resize(12, n);
        g.setZero(); rs_g.setZero(); grad_g.setZero();
        rs_grad_g.setZero(); rsrs_grad_g.setZero();
        hess_g.setZero(); rs_hess_g.setZero();
        return;
    };
};


/**
* @brief Base class for integration over the source triangle for RWG-based BEM operators.
*/
class SrcIntegratorBase
{
public:

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
    * If `shape_grad_terms` is true, the function computes the first and second moments of the
    * kernel gradient,
    * \f[
    * \int_{\mathrm{src\_tri}} d\mathcal{S}'\,p(x', y')\,\nabla G(k, \vec{r}, \vec{r}\,'),
    * \qquad p \in \{x', y', x'^2, x'y', y'^2\},
    * \f]
    * stored in `SrcResult::rs_grad_g` and `SrcResult::rsrs_grad_g`.
    * If `shape_hess_terms` is true, the function computes
    * \f[
    * \int_{\mathrm{src\_tri}} d\mathcal{S}'\,p(x', y')\,\nabla\nabla G(k, \vec{r}, \vec{r}\,'),
    * \qquad p \in \{1, x', y'\},
    * \f]
    * stored in `SrcResult::hess_g` and `SrcResult::rs_hess_g`.
    * These terms are required for shape derivatives of operators and projectors.
    */
    virtual SrcResult integrate(
        const Complex k,
        const Triangle<2>& src_tri,
        ConstEigRef<EigMatNX<Float, 3>> r_obs,
        const bool g_terms = true,
        const bool grad_g_terms = true,
        const bool shape_grad_terms = false,
        const bool shape_hess_terms = false
        ) = 0;


    /**
    * @brief Returns a result of zeros of the appropriate size.
    * @param[in] k - Complex wavenumber.
    * @param[in] src_tri - Source triangle in 2D space.
    * @param[in] r_obs - Observation points in the local coordinate system of `src_tri`.
    * @return Integration result.
    */
    virtual SrcResult zeros(
        const Complex k,
        const Triangle<2>& src_tri,
        ConstEigRef<EigMatNX<Float, 3>> r_obs
        ) { SrcResult result; result.resize(r_obs.cols()); return result; };


    /**
    * @brief Virtual destructor.
    */
    virtual ~SrcIntegratorBase() = default;

};

/**
* @}
*/

}

#endif
