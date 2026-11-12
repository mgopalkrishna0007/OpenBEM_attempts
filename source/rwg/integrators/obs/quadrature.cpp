// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Quadrature over the observation triangle for RWG-based BEM operators.
*/

#include "rwg/integrators/obs/quadrature.hpp"

#include "types.hpp"
#include "geometry/primitives/triangle.hpp"
#include "geometry/operations.hpp"
#include "rwg/integrators/obs/base.hpp"
#include "rwg/integrators/src/base.hpp"


namespace bem::rwg
{

ObsResult ObsQuadrature::integrate(
    const Complex k,
    const Triangle<3>& obs_tri,
    const Triangle<2>& src_tri,
    const bool g_term,
    const bool rs_g_terms,
    const bool grad_g_terms,
    const bool rot_grad_g_terms,
    const bool shape_terms
    )
{

    // Evaluation function for iterative or adaptive numerical integration
    auto eval = [&](ConstEigRef<EigMatNX<Float, 3>> r_obs) -> EigRowVec<Complex>
    {
        return src_integrator_->integrate(k, src_tri, r_obs, true, false).g;
    };

    QuadratureData<3> qd = tri_quad_->compute(obs_tri, eval);

    bool src_g_terms = g_term || rs_g_terms;
    bool src_grad_g_terms = grad_g_terms || rot_grad_g_terms || shape_terms;

    SrcResult src_result = src_integrator_->integrate(
        k, src_tri, qd.points, src_g_terms, src_grad_g_terms, shape_terms, false
        );

    ObsResult obs_result;

    if (g_term)
        obs_result.g = compute_g_term(src_result, qd);

    if (rs_g_terms)
        obs_result.rs_g = compute_rs_g_terms(src_result, qd);

    if (grad_g_terms)
        obs_result.grad_g = compute_grad_g_terms(src_result, qd);

    if (rot_grad_g_terms)
        obs_result.rot_grad_g = compute_rot_grad_g_terms(src_result, qd);

    if (shape_terms)
        obs_result.grad_g_shape = compute_grad_g_shape_terms(src_result, qd);

    return obs_result;

};


Complex ObsQuadrature::compute_g_term(
    const SrcResult& src_result, const QuadratureData<3>& qd
    )
{
    return qd.weights.dot(src_result.g);
};


EigRowVecN<Complex, 12> ObsQuadrature::compute_rs_g_terms(
    const SrcResult& src_result, const QuadratureData<3>& qd
    )
{

    EigRowVec<Complex> wgw = src_result.g * qd.weights.asDiagonal();
    EigMatNX<Complex, 2> rs_wgw = src_result.rs_g * qd.weights.asDiagonal();

    EigColVecN<Complex, 2> sum_rs_wgw = rs_wgw.rowwise().sum();
    EigRowVecN<Complex, 3> p_wgw = wgw * qd.points.transpose();
    EigMatMN<Complex, 2, 3> p_rs_wgw = rs_wgw * qd.points.transpose();

    EigRowVecN<Complex, 12> I = EigRowVecN<Complex, 12>::Zero(1, 12);

    I[0] = wgw.sum();
    I[1] = sum_rs_wgw[0];
    I[2] = sum_rs_wgw[1];

    I[3] = p_wgw[0];
    I[4] = p_wgw[1];
    I[5] = p_wgw[2];

    I[6] = p_rs_wgw(0, 0);
    I[7] = p_rs_wgw(0, 1);
    I[8] = p_rs_wgw(0, 2);

    I[9] = p_rs_wgw(1, 0);
    I[10] = p_rs_wgw(1, 1);
    I[11] = p_rs_wgw(1, 2);

    return I;

};


EigRowVecN<Complex, 9> ObsQuadrature::compute_grad_g_terms(
    const SrcResult& src_result, const QuadratureData<3>& qd
    )
{

    EigMatNX<Complex, 3> grad_wgw = src_result.grad_g * qd.weights.asDiagonal();
    EigColVecN<Complex, 3> sum_grad_wgw = grad_wgw.rowwise().sum();
    EigMatMN<Complex, 3, 3> p_grad_wgw = grad_wgw * qd.points.transpose();

    EigRowVecN<Complex, 9> I = EigRowVecN<Complex, 9>::Zero(1, 9);

    I[0] = sum_grad_wgw[0];
    I[1] = p_grad_wgw(0, 1);
    I[2] = p_grad_wgw(0, 2);

    I[3] = sum_grad_wgw[1];
    I[4] = p_grad_wgw(1, 0);
    I[5] = p_grad_wgw(1, 2);

    I[6] = sum_grad_wgw[2];
    I[7] = p_grad_wgw(2, 0);
    I[8] = p_grad_wgw(2, 1);

    return I;

};


EigRowVecN<Complex, 15> ObsQuadrature::compute_rot_grad_g_terms(
    const SrcResult& src_result, const QuadratureData<3>& qd
    )
{

    EigMatNX<Complex, 3> grad_wgw = src_result.grad_g * qd.weights.asDiagonal();
    EigMatMN<Complex, 3, 3> p_grad_wgw = grad_wgw * qd.points.transpose();

    EigMatMN<Complex, 3, 1> xy_grad_wgw = grad_wgw * (
        qd.points.row(0).array() * qd.points.row(1).array()
        ).matrix().transpose();
    EigMatMN<Complex, 3, 1> xz_grad_wgw = grad_wgw * (
        qd.points.row(0).array() * qd.points.row(2).array()
        ).matrix().transpose();
    EigMatMN<Complex, 3, 1> yz_grad_wgw = grad_wgw * (
        qd.points.row(1).array() * qd.points.row(2).array()
        ).matrix().transpose();

    EigMatMN<Complex, 3, 1> xx_grad_wgw = grad_wgw * (
        qd.points.row(0).array() * qd.points.row(0).array()
        ).matrix().transpose();
    EigMatMN<Complex, 3, 1> yy_grad_wgw = grad_wgw * (
        qd.points.row(1).array() * qd.points.row(1).array()
        ).matrix().transpose();
    EigMatMN<Complex, 3, 1> zz_grad_wgw = grad_wgw * (
        qd.points.row(2).array() * qd.points.row(2).array()
        ).matrix().transpose();

    EigRowVecN<Complex, 15> I = EigRowVecN<Complex, 15>::Zero(1, 15);

    I[0] = p_grad_wgw(0, 0);
    I[5] = p_grad_wgw(1, 1);
    I[12] = p_grad_wgw(2, 2);

    I[1] = xy_grad_wgw[0];
    I[2] = xz_grad_wgw[0];
    I[3] = yy_grad_wgw[0];
    I[4] = zz_grad_wgw[0];

    I[6] = xy_grad_wgw[1];
    I[7] = yz_grad_wgw[1];
    I[8] = xx_grad_wgw[1];
    I[9] = zz_grad_wgw[1];

    I[10] = xz_grad_wgw[2];
    I[11] = yz_grad_wgw[2];
    I[13] = xx_grad_wgw[2];
    I[14] = yy_grad_wgw[2];

    return I;

};


EigMat<Complex> ObsQuadrature::compute_grad_g_shape_terms(
    const SrcResult& src_result, const QuadratureData<3>& qd
    )
{

    const Index num_points = qd.points.cols();

    // Weighted observation monomials (1, x, y, z, x^2, xy, xz, y^2, yz, z^2)
    EigMat<Float> mono (num_points, 10);
    for (Index q = 0; q < num_points; ++q)
    {
        const Float x = qd.points(0, q), y = qd.points(1, q), z = qd.points(2, q);
        const Float w = qd.weights[q];
        mono.row(q) << w, w * x, w * y, w * z, w * x * x, w * x * y, w * x * z,
            w * y * y, w * y * z, w * z * z;
    }

    // Source moments (1, x', y', x'^2, x'y', y'^2) of each gradient component
    const EigMatNX<Complex, 3>* src[6];
    EigMatNX<Complex, 3> rs_x = src_result.rs_grad_g.topRows(3);
    EigMatNX<Complex, 3> rs_y = src_result.rs_grad_g.bottomRows(3);
    EigMatNX<Complex, 3> rsrs_xx = src_result.rsrs_grad_g.topRows(3);
    EigMatNX<Complex, 3> rsrs_xy = src_result.rsrs_grad_g.middleRows(3, 3);
    EigMatNX<Complex, 3> rsrs_yy = src_result.rsrs_grad_g.bottomRows(3);
    src[0] = &src_result.grad_g;
    src[1] = &rs_x;
    src[2] = &rs_y;
    src[3] = &rsrs_xx;
    src[4] = &rsrs_xy;
    src[5] = &rsrs_yy;

    EigMat<Complex> I = EigMat<Complex>::Zero(3, 60);
    for (uint8_t b = 0; b < 6; ++b)
    {
        const EigMatMN<Complex, 3, 10> block = (*src[b]) * mono;
        for (uint8_t a = 0; a < 10; ++a)
            I.col(6 * a + b) = block.col(a);
    }

    return I;

};

}
