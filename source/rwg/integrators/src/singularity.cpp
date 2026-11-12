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

#include "rwg/integrators/src/singularity.hpp"

#include <cmath>
#include <stdexcept>

#include "types.hpp"
#include "constants.hpp"
#include "geometry/primitives/edge.hpp"
#include "geometry/primitives/triangle.hpp"


namespace bem::rwg
{

SrcResult SrcSingularity::integrate(
    const Complex k,
    const Triangle<2>& src_tri,
    ConstEigRef<EigMatNX<Float, 3>> r_obs,
    const bool g_terms,
    const bool grad_g_terms,
    const bool shape_grad_terms,
    const bool shape_hess_terms
    )
{
    if (shape_hess_terms)
        throw std::runtime_error(
            "SrcSingularity::integrate(): Hessian shape terms are only supported for observation "
            "points away from the source triangle."
            );

    SrcResult result_singular = integrate_singular(k, src_tri, r_obs, g_terms, grad_g_terms);
    SrcResult result_nonsingular = src_quad_.integrate(
        k, src_tri, r_obs, g_terms, grad_g_terms, shape_grad_terms, false
        );

    SrcResult result;
    result.g = result_singular.g + result_nonsingular.g;
    result.rs_g = result_singular.rs_g + result_nonsingular.rs_g;
    result.grad_g = result_singular.grad_g + result_nonsingular.grad_g;

    if (shape_grad_terms)
    {
        integrate_singular_shape(k, src_tri, r_obs, result.rs_grad_g, result.rsrs_grad_g);
        result.rs_grad_g += result_nonsingular.rs_grad_g;
        result.rsrs_grad_g += result_nonsingular.rsrs_grad_g;
    }

    return result;
};


SrcResult SrcSingularity::integrate_singular(
    const Complex k,
    const Triangle<2>& src_tri,
    ConstEigRef<EigMatNX<Float, 3>> r_obs,
    const bool g_terms,
    const bool grad_g_terms
    )
{

    // projection of the observation point on the source triangle's local plane, assuming the
    // observation points are in the source's local coordinate system
    EigMatNX<Float, 2> rho_obs = r_obs.topRows(2);
    EigRowVec<Float> z = r_obs.row(2);
    EigRowVec<Float> z_abs = z.cwiseAbs();
    EigRowVec<Float> z_sq = z.array() * z.array();

    EigRowVec<Float> beta = EigRowVec<Float>::Zero(1, r_obs.cols());
    EigRowVec<Float> t0_f2 = EigRowVec<Float>::Zero(1, r_obs.cols());
    EigMatNX<Float, 2> u_f2 = EigMatNX<Float, 2>::Zero(2, r_obs.cols());
    EigMatNX<Float, 2> u_f3 = EigMatNX<Float, 2>::Zero(2, r_obs.cols());

    EigMatNX<Float, 2> plus_to_rho (2, r_obs.cols());
    EigMatNX<Float, 2> minus_to_rho (2, r_obs.cols());

    EigRowVec<Float> s_plus (1, r_obs.cols());
    EigRowVec<Float> s_minus (1, r_obs.cols());

    EigRowVec<Float> t0 (1, r_obs.cols());
    EigRowVec<Float> R0 (1, r_obs.cols());

    EigRowVec<Float> R_plus (1, r_obs.cols());
    EigRowVec<Float> R_minus (1, r_obs.cols());

    EigRowVec<Float> R0_sq (1, r_obs.cols());
    EigRowVec<Float> atan_plus (1, r_obs.cols());
    EigRowVec<Float> atan_minus (1, r_obs.cols());

    EigRowVec<Float> f2 (1, r_obs.cols());
    EigRowVec<Float> f3 (1, r_obs.cols());

    for (uint8_t idx = 0; idx < 3; idx++)
    {

        // Indices of the edge's vertices
        uint8_t idx_plus = idx;
        uint8_t idx_minus = (idx + 1) % 3;

        Edge<2> edge (src_tri.v(idx_plus), src_tri.v(idx_minus));
        EigColVecN<Float, 2> u_hat = { edge.unit_vec()[1], -edge.unit_vec()[0] };

        // Projection of the projected observation point to the line defined by the edge
        plus_to_rho = rho_obs.colwise() - edge.v(0);
        minus_to_rho = rho_obs.colwise() - edge.v(1);
        s_plus = edge.unit_vec().transpose() * plus_to_rho;
        s_minus = s_plus.array() - edge.length();
        t0 = -(u_hat.transpose() * plus_to_rho);
        R0 = Eigen::sqrt((t0.array() * t0.array()) + z_sq.array());

        // Distance from the observation points to each end point of the edge
        R_plus = Eigen::sqrt(plus_to_rho.colwise().squaredNorm().array() + z_sq.array());
        R_minus = Eigen::sqrt(minus_to_rho.colwise().squaredNorm().array() + z_sq.array());

        // Compute the terms which compose the final integrals
        R0_sq = R0.array() * R0.array();

        atan_plus = (t0.array() * s_plus.array()).atan2(
            R0_sq.array() + z_abs.array() * R_plus.array()
            );

        atan_minus = (t0.array() * s_minus.array()).atan2(
            R0_sq.array() + z_abs.array() * R_minus.array()
            );

        f2 = Eigen::log(((R_plus + s_plus).array() / (R_minus + s_minus).array()));
        f2 = (!f2.array().isFinite()).select(EigRowVec<Float>::Zero(1, f2.cols()), f2);

        f3 = s_plus.array() * R_plus.array() -
            s_minus.array() * R_minus.array() +
            R0_sq.array() * f2.array();

        beta += atan_plus - atan_minus;
        t0_f2 += (t0.array() * f2.array()).matrix();
        u_f2 += u_hat * f2;
        u_f3 += u_hat * f3;

        assert(atan_plus.array().isFinite().all() &&
            "SrcSingularity::integrate_singular(): atan_plus has nan or inf.");
        assert(atan_minus.array().isFinite().all() &&
            "SrcSingularity::integrate_singulars(): atan_minus has nan or inf.");
        assert(f2.array().isFinite().all() &&
            "SrcSingularity::integrate_singular(): f2 has nan or inf.");
        assert(f3.array().isFinite().all() &&
            "SrcSingularity::integrate_singular(): f3 has nan or inf.");

    }

    SrcResult result;

    if (g_terms)
    {
        result.g = (t0_f2 - z_abs.cwiseProduct(beta)) / four_pi;

        result.rs_g = 0.5 * u_f3 / four_pi;
        result.rs_g += r_obs.topRows(2) * result.g.asDiagonal();
    }

    if (grad_g_terms)
    {
        EigRowVec<Float> sign = z.array() / z_abs.array();
        sign = (
            z_abs.array() < TRIANGLE_DEFAULT_TOL * src_tri.shortest_edge_length()
        ).select(1, sign);

        result.grad_g = EigMatNX<Complex, 3>::Zero(3, r_obs.cols());
        result.grad_g.topRows(2) = -u_f2;
        result.grad_g.row(2) = -sign.array() * beta.array();

        EigMatNX<Float, 3> grad_r = EigMatNX<Float, 3>::Zero(3, r_obs.cols());
        grad_r.topRows(2) = 0.5 * u_f3;
        grad_r.row(2) = z.array() * z_abs.array() * beta.array() - z.array() * t0_f2.array();

        result.grad_g += (half * k * k) * grad_r;
        result.grad_g /= four_pi;
    }

    return result;

};


void SrcSingularity::integrate_singular_shape(
    const Complex k,
    const Triangle<2>& src_tri,
    ConstEigRef<EigMatNX<Float, 3>> r_obs,
    EigMatNX<Complex, 6>& rs_grad_g,
    EigMatNX<Complex, 9>& rsrs_grad_g
    )
{

    using Vec2 = EigColVecN<Float, 2>;
    using Mat2 = EigMatMN<Float, 2, 2>;

    rs_grad_g.setZero(6, r_obs.cols());
    rsrs_grad_g.setZero(9, r_obs.cols());

    const Mat2 id2 = Mat2::Identity();
    const Complex half_k_sq = half * k * k;
    const Float tol_z = TRIANGLE_DEFAULT_TOL * src_tri.shortest_edge_length();

    for (Index ro = 0; ro < r_obs.cols(); ++ro)
    {

        const Vec2 rho = r_obs.col(ro).topRows(2);
        const Float z = r_obs(2, ro);
        const Float z_abs = std::abs(z);
        const Float sign = (z_abs < tol_z) ? one : z / z_abs;

        Float sum_beta = 0, sum_t0_f0 = 0, sum_t0_l0 = 0;
        Vec2 u_f0 = Vec2::Zero(), u_l0 = Vec2::Zero(), u_l3 = Vec2::Zero();
        Mat2 a_ipp = Mat2::Zero(), a_m2 = Mat2::Zero();

        // Edge sums for the third moments: sum_i (int_edge p p^T R dl) (x) u_i and with 1/R
        Mat2 a_pppR[2] = {Mat2::Zero(), Mat2::Zero()};
        Mat2 a_ppp3[2] = {Mat2::Zero(), Mat2::Zero()};

        for (uint8_t idx = 0; idx < 3; ++idx)
        {
            const Vec2 v_start = src_tri.v(idx);
            const Vec2 v_end = src_tri.v((idx + 1) % 3);
            const Float len = (v_end - v_start).norm();
            const Vec2 l_hat = (v_end - v_start) / len;
            const Vec2 u_hat = { l_hat[1], -l_hat[0] }; // outward edge normal

            // Points on the edge are rho + t0 u_hat + l l_hat with l in [lm, lp]
            const Float lm = (v_start - rho).dot(l_hat);
            const Float lp = lm + len;
            const Float t0 = (v_start - rho).dot(u_hat);
            const Float r0_sq = t0 * t0 + z * z;
            const Float rm = std::sqrt(lm * lm + r0_sq);
            const Float rp = std::sqrt(lp * lp + r0_sq);

            Float f0 = (lm + lp >= 0) ? std::log((rp + lp) / (rm + lm)) : std::log((rm - lm) / (rp - lp));
            if (!std::isfinite(f0))
                f0 = 0;

            Float beta = 0;
            if (r0_sq > float_eps * len * len)
                beta = std::atan2(t0 * lp, r0_sq + z_abs * rp) - std::atan2(t0 * lm, r0_sq + z_abs * rm);

            // Line integrals along the edge
            const Float f1 = rp - rm;                                               // int l/R
            const Float f2 = half * (lp * rp - lm * rm) - half * r0_sq * f0;        // int l^2/R
            const Float l0 = half * (lp * rp - lm * rm) + half * r0_sq * f0;        // int R
            const Float l1 = (rp * rp * rp - rm * rm * rm) / 3;                     // int l R
            const Float l2 = (lp * (2 * lp * lp + r0_sq) * rp
                - lm * (2 * lm * lm + r0_sq) * rm) / 8 - r0_sq * r0_sq * f0 / 8;    // int l^2 R
            const Float l3 = (lp * (2 * lp * lp + 5 * r0_sq) * rp
                - lm * (2 * lm * lm + 5 * r0_sq) * rm) / 8
                + 3 * r0_sq * r0_sq * f0 / 8;                                       // int R^3

            sum_beta += beta;
            sum_t0_f0 += t0 * f0;
            sum_t0_l0 += t0 * l0;
            u_f0 += u_hat * f0;
            u_l0 += u_hat * l0;
            u_l3 += u_hat * l3;

            a_ipp += (t0 * l0 * u_hat + l1 * l_hat) * u_hat.transpose();
            a_m2 += (t0 * f0 * u_hat + f1 * l_hat) * u_hat.transpose();

            const Mat2 uu = u_hat * u_hat.transpose();
            const Mat2 ul = u_hat * l_hat.transpose() + l_hat * u_hat.transpose();
            const Mat2 ll = l_hat * l_hat.transpose();
            const Mat2 ppR = t0 * t0 * l0 * uu + t0 * l1 * ul + l2 * ll;
            const Mat2 pp3 = t0 * t0 * f0 * uu + t0 * f1 * ul + f2 * ll;
            for (uint8_t c = 0; c < 2; ++c)
            {
                a_pppR[c] += ppR * u_hat[c];
                a_ppp3[c] += pp3 * u_hat[c];
            }
        }

        // Area integrals over the source triangle
        const Float i0 = sum_t0_f0 - z_abs * sum_beta;        // int 1/R
        const Float iR = (sum_t0_l0 + z * z * i0) / 3;        // int R
        const Vec2 ip = u_l0;                                  // int p/R
        const Vec2 ipR = u_l3 / 3;                             // int p R
        const Mat2 ipp = a_ipp - iR * id2;                     // int p p^T/R
        const Float z_m0 = sign * sum_beta;                    // z int 1/R^3
        const Vec2 m1 = -u_f0;                                 // int p/R^3
        const Mat2 m2 = -a_m2 + i0 * id2;                      // int p p^T/R^3

        // Third moments: ippp[c](a, b) = int p_a p_b p_c / R, mppp likewise with 1/R^3
        Mat2 ippp[2], mppp[2];
        for (uint8_t c = 0; c < 2; ++c)
        {
            ippp[c] = a_pppR[c];
            mppp[c] = -a_ppp3[c];
            for (uint8_t a = 0; a < 2; ++a)
                for (uint8_t b = 0; b < 2; ++b)
                {
                    ippp[c](a, b) -= (a == c ? ipR[b] : 0) + (b == c ? ipR[a] : 0);
                    mppp[c](a, b) += (a == c ? ip[b] : 0) + (b == c ? ip[a] : 0);
                }
        }

        // Weights w = w0 + w1 . p + p^T W2 p for w in {x', y', x'x', x'y', y'y'}
        auto weighted = [&] (const Float w0, const Vec2& w1, const Mat2& w2) -> EigColVecN<Complex, 3>
        {
            Vec2 t_m = Vec2::Zero(), t_i = Vec2::Zero();
            for (uint8_t c = 0; c < 2; ++c)
            {
                t_m[c] = (w2.array() * mppp[c].array()).sum();
                t_i[c] = (w2.array() * ippp[c].array()).sum();
            }
            const Float w2_m2 = (w2.array() * m2.array()).sum();
            const Float w2_ipp = (w2.array() * ipp.array()).sum();

            // int w grad(1/R): in-plane int w p/R^3, normal -z int w/R^3
            const Vec2 s_inv_par = w0 * m1 + m2 * w1 + t_m;
            const Float s_inv_z = -(w0 * z_m0 + z * (w1.dot(m1) + w2_m2));

            // int w grad(R): in-plane -int w p/R, normal z int w/R
            const Vec2 s_r_par = -(w0 * ip + ipp * w1 + t_i);
            const Float s_r_z = z * (w0 * i0 + w1.dot(ip) + w2_ipp);

            EigColVecN<Complex, 3> out;
            out[0] = (s_inv_par[0] - half_k_sq * s_r_par[0]) / four_pi;
            out[1] = (s_inv_par[1] - half_k_sq * s_r_par[1]) / four_pi;
            out[2] = (s_inv_z - half_k_sq * s_r_z) / four_pi;
            return out;
        };

        const Vec2 ex = { 1, 0 }, ey = { 0, 1 };
        const Mat2 zero2 = Mat2::Zero();
        Mat2 w_xx = Mat2::Zero(), w_xy = Mat2::Zero(), w_yy = Mat2::Zero();
        w_xx(0, 0) = 1;
        w_xy(0, 1) = half; w_xy(1, 0) = half;
        w_yy(1, 1) = 1;

        // x' = rho_x + p_x, etc.
        rs_grad_g.col(ro).segment<3>(0) = weighted(rho[0], ex, zero2);
        rs_grad_g.col(ro).segment<3>(3) = weighted(rho[1], ey, zero2);
        rsrs_grad_g.col(ro).segment<3>(0) = weighted(rho[0] * rho[0], 2 * rho[0] * ex, w_xx);
        rsrs_grad_g.col(ro).segment<3>(3) = weighted(rho[0] * rho[1], rho[1] * ex + rho[0] * ey, w_xy);
        rsrs_grad_g.col(ro).segment<3>(6) = weighted(rho[1] * rho[1], 2 * rho[1] * ey, w_yy);

    }

    return;

};

}
