// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Quadrature over the source triangle for RWG-based BEM operators.
*/

#include "rwg/integrators/src/quadrature.hpp"

#include "types.hpp"
#include "geometry/primitives/triangle.hpp"


namespace bem::rwg
{

SrcResult SrcQuadrature::integrate(
    const Complex k,
    const Triangle<2>& src_tri,
    ConstEigRef<EigMatNX<Float, 3>> r_obs,
    const bool g_terms,
    const bool grad_g_terms,
    const bool shape_grad_terms,
    const bool shape_hess_terms
    )
{

    // Lambda used for iterative or adaptive numerical integration
    auto eval = [&](ConstEigRef<EigMatNX<Float, 2>> r_src) -> EigRowVec<Complex>
    {
        EigMatNX<Float, 3> r_src_3d = EigMatNX<Float, 3>::Zero(3, r_src.cols());
        r_src_3d.topRows(2) = r_src;
        return kernel_->compute(r_obs.rowwise().mean(), r_src_3d, k);
    };

    // Get the quadrature points and weights
    QuadratureData<2> qd = tri_quad_->compute(src_tri, eval);

    EigMatNX<Float, 3> points_3d = EigMatNX<Float, 3>::Zero(3, qd.points.cols());
    points_3d.topRows(2) = qd.points;

    // Assemble the integration results
    SrcResult result;

    if (g_terms)
    {
        result.g.setZero(1, r_obs.cols());
        result.rs_g.setZero(2, r_obs.cols());

        for (std::size_t ro = 0; ro < r_obs.cols(); ++ro)
        {
            EigRowVec<Complex> gw = kernel_->compute(
                r_obs.col(ro), points_3d, k
                ).array() * qd.weights.array();
            result.g[ro] = gw.array().sum();
            result.rs_g.col(ro) = qd.points * gw.transpose();
        }
    }

    if (grad_g_terms)
    {
        result.grad_g.setZero(3, r_obs.cols());

        for (std::size_t ro = 0; ro < r_obs.cols(); ++ro)
        {
            result.grad_g.col(ro) = kernel_->compute_grad(
                r_obs.col(ro), points_3d, k
                ) * qd.weights.transpose();
        }
    }

    if (shape_grad_terms || shape_hess_terms)
    {
        // Quadrature weights multiplied by the source-coordinate monomials x', y', x'^2, x'y', y'^2
        const EigRowVec<Float> wx = qd.weights.array() * qd.points.row(0).array();
        const EigRowVec<Float> wy = qd.weights.array() * qd.points.row(1).array();
        const EigRowVec<Float> wxx = wx.array() * qd.points.row(0).array();
        const EigRowVec<Float> wxy = wx.array() * qd.points.row(1).array();
        const EigRowVec<Float> wyy = wy.array() * qd.points.row(1).array();

        if (shape_grad_terms)
        {
            result.rs_grad_g.setZero(6, r_obs.cols());
            result.rsrs_grad_g.setZero(9, r_obs.cols());

            for (std::size_t ro = 0; ro < r_obs.cols(); ++ro)
            {
                const EigMatNX<Complex, 3> grad = kernel_->compute_grad(r_obs.col(ro), points_3d, k);
                result.rs_grad_g.col(ro).segment<3>(0) = grad * wx.transpose();
                result.rs_grad_g.col(ro).segment<3>(3) = grad * wy.transpose();
                result.rsrs_grad_g.col(ro).segment<3>(0) = grad * wxx.transpose();
                result.rsrs_grad_g.col(ro).segment<3>(3) = grad * wxy.transpose();
                result.rsrs_grad_g.col(ro).segment<3>(6) = grad * wyy.transpose();
            }
        }

        if (shape_hess_terms)
        {
            result.hess_g.setZero(6, r_obs.cols());
            result.rs_hess_g.setZero(12, r_obs.cols());

            for (std::size_t ro = 0; ro < r_obs.cols(); ++ro)
            {
                const EigMatNX<Complex, 6> hess = kernel_->compute_hess(r_obs.col(ro), points_3d, k);
                result.hess_g.col(ro) = hess * qd.weights.transpose();
                result.rs_hess_g.col(ro).segment<6>(0) = hess * wx.transpose();
                result.rs_hess_g.col(ro).segment<6>(6) = hess * wy.transpose();
            }
        }
    }

    return result;

};

}

