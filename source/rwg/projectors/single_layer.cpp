// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* RWG-based single-layer potential BEM projectors.
*/

#include "rwg/projectors/single_layer.hpp"

#include "types.hpp"
#include "constants.hpp"
#include "geometry/primitives/triangle.hpp"
#include "rwg/function_space.hpp"


namespace bem::rwg
{

EigMat<Complex> VectorSingleLayerProj::compute(
    const Complex k,
    ConstEigRef<EigMatNX<Float, 3>> obs_points,
    const Triangle<3>& src_tri
    )
{

    // Convert to the source triangle's local coordinate system
    EigMatNX<Float, 3> obs_points_local;
    Triangle<2> src_tri_local;
    transform_coordinates(obs_points_local, src_tri_local, obs_points, src_tri);

    const SrcResult src_result = src_integrator_->integrate(
        k, src_tri_local, obs_points_local, true, false
        );

    EigMatMN<Float, 3, 3> local_uvw = src_tri.local_coordinate_basis();
    EigRowVecN<Float, 3> norms = Rwg::normalization(src_tri);
    EigMatXN<Complex, 3> result = EigMatXN<Complex, 3>::Zero(3 * obs_points.cols(), 3);

    // source triangle edges
    for (uint8_t jj = 0; jj < 3; ++jj)
    {
        EigMatNX<Complex, 2> rwg_g = src_result.rs_g - src_tri_local.v((jj + 2) % 3) *
            src_result.g;
        result.col(jj) = (
            local_uvw.leftCols(2) * rwg_g
            ).reshaped(3 * obs_points.cols(), 1) * norms(jj);
    }

    return result;

};


EigMat<Complex> ScalarSingleLayerProj::compute(
    const Complex k,
    ConstEigRef<EigMatNX<Float, 3>> obs_points,
    const Triangle<3>& src_tri
    )
{

    // Convert to the source triangle's local coordinate system
    EigMatNX<Float, 3> obs_points_local;
    Triangle<2> src_tri_local;
    transform_coordinates(obs_points_local, src_tri_local, obs_points, src_tri);

    const SrcResult src_result = src_integrator_->integrate(
        k, src_tri_local, obs_points_local, true, false
        );

    return src_result.g.transpose() * Pulse::normalization(src_tri);

};


EigMat<Complex> GradScalarSingleLayerProj::compute(
    const Complex k,
    ConstEigRef<EigMatNX<Float, 3>> obs_points,
    const Triangle<3>& src_tri
    )
{

    // Convert to the source triangle's local coordinate system
    EigMatNX<Float, 3> obs_points_local;
    Triangle<2> src_tri_local;
    transform_coordinates(obs_points_local, src_tri_local, obs_points, src_tri);

    const SrcResult src_result = src_integrator_->integrate(
        k, src_tri_local, obs_points_local, false, true
        );

    EigMatMN<Float, 3, 3> local_uvw = src_tri.local_coordinate_basis();
    EigMatXN<Complex, 1> result = (
        local_uvw * src_result.grad_g
        ).reshaped(3 * obs_points.cols(), 1) * Pulse::normalization(src_tri);

    return result;

};


EigMat<Complex> VectorHypersingularProj::compute(
    const Complex k,
    ConstEigRef<EigMatNX<Float, 3>> obs_points,
    const Triangle<3>& src_tri
    )
{
    EigMatXN<Complex, 3> result = proj_g_.compute(k, obs_points, src_tri);
    EigMatXN<Complex, 1> hessg_term = proj_gradg_.compute(k, obs_points, src_tri) / k / k;
    result += hessg_term * src_tri.edge_polarities();
    return result;
};


EigMat<Complex> VectorHypersingularProj::compute_shape(
    const Complex k,
    ConstEigRef<EigMatNX<Float, 3>> obs_points,
    const Triangle<3>& src_tri
    )
{

    // Convert to the source triangle's local coordinate system
    EigMatNX<Float, 3> obs_points_local;
    Triangle<2> src_tri_local;
    transform_coordinates(obs_points_local, src_tri_local, obs_points, src_tri);

    const SrcResult src_result = src_integrator_->integrate(
        k, src_tri_local, obs_points_local, true, true, true, true
        );

    const EigMatMN<Float, 3, 3> local_uvw = src_tri.local_coordinate_basis();
    const EigRowVecN<Float, 3> norms = Rwg::normalization(src_tri);
    const Complex inv_k_sq = one / (k * k);
    const Index num_points = obs_points.cols();

    // Hessian component index for (a, b)
    static const uint8_t hidx[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};

    // Barycentric coordinates lambda = offset + grad . r' in the local (2D) system
    EigMatMN<Float, 2, 3> grad_lambda;
    EigRowVecN<Float, 3> off_lambda;
    {
        const Float twice_area = 2 * src_tri_local.area();
        for (uint8_t vert = 0; vert < 3; ++vert)
        {
            const EigColVecN<Float, 2> e = src_tri_local.v((vert + 2) % 3) - src_tri_local.v((vert + 1) % 3);
            grad_lambda.col(vert) = EigColVecN<Float, 2> (-e[1], e[0]) / twice_area;
            off_lambda[vert] = one - grad_lambda.col(vert).dot(src_tri_local.v(vert));
        }
    }

    EigMat<Complex> result = EigMat<Complex>::Zero(3 * num_points, 27);

    for (Index pt = 0; pt < num_points; ++pt)
    {
        // Source moments at this observation point
        const Complex g0 = src_result.g[pt];
        const Complex gx = src_result.rs_g(0, pt), gy = src_result.rs_g(1, pt);

        // int w d_c G for w in (1, x', y', x'^2, x'y', y'^2), indexed [w][c]
        Complex dg[6][3];
        for (uint8_t c = 0; c < 3; ++c)
        {
            dg[0][c] = src_result.grad_g(c, pt);
            dg[1][c] = src_result.rs_grad_g(c, pt);
            dg[2][c] = src_result.rs_grad_g(3 + c, pt);
            dg[3][c] = src_result.rsrs_grad_g(c, pt);
            dg[4][c] = src_result.rsrs_grad_g(3 + c, pt);
            dg[5][c] = src_result.rsrs_grad_g(6 + c, pt);
        }

        // int w d_a d_b G for w in (1, x', y')
        auto hess = [&] (const uint8_t w, const uint8_t a, const uint8_t b) -> Complex
        {
            const uint8_t h = hidx[a][b];
            if (w == 0) return src_result.hess_g(h, pt);
            return src_result.rs_hess_g(6 * (w - 1) + h, pt);
        };

        for (uint8_t jj = 0; jj < 3; ++jj)
        {
            const EigColVecN<Float, 2> pn = src_tri_local.v((jj + 2) % 3);
            const Float t = norms[jj];

            for (uint8_t vert = 0; vert < 3; ++vert)
            {
                const Float ax = grad_lambda(0, vert), ay = grad_lambda(1, vert), a0 = off_lambda[vert];

                // int (grad lambda . f_n) G, with f_n = t (r' - p_n)
                const Complex gl_f_g = t * (ax * gx + ay * gy - grad_lambda.col(vert).dot(pn) * g0);

                // Local derivative tensor D(m, c): field component m, vertex component c
                EigMatMN<Complex, 3, 3> D = EigMatMN<Complex, 3, 3>::Zero();
                for (uint8_t c = 0; c < 3; ++c)
                {
                    D(c, c) += gl_f_g;

                    // - int f_n,m lambda d_c G, for in-plane m: f_n,m lambda = t (r'_m - p_m)(a0 + ax x' + ay y')
                    for (uint8_t m = 0; m < 2; ++m)
                    {
                        const Float pm = pn[m];
                        // coefficients in (1, x', y', x'^2, x'y', y'^2)
                        Float w[6] = {-pm * a0, -pm * ax, -pm * ay, 0, 0, 0};
                        if (m == 0) { w[0] += 0; w[1] += a0; w[3] += ax; w[4] += ay; }
                        else        { w[2] += a0; w[4] += ax; w[5] += ay; }
                        Complex val = 0;
                        for (uint8_t q = 0; q < 6; ++q)
                            if (w[q] != 0) val += w[q] * dg[q][c];
                        D(m, c) -= t * val;
                    }

                    // - (div f_n / k^2) int lambda d_m d_c G, with div f_n = 2 t
                    for (uint8_t m = 0; m < 3; ++m)
                        D(m, c) -= 2 * t * inv_k_sq * (a0 * hess(0, m, c) + ax * hess(1, m, c) + ay * hess(2, m, c));
                }

                // Rotate both the field component and the vertex component to the global system
                const EigMatMN<Complex, 3, 3> Dg = local_uvw * D * local_uvw.transpose();
                for (uint8_t c = 0; c < 3; ++c)
                    result.block(3 * pt, 9 * jj + 3 * vert + c, 3, 1) = Dg.col(c);
            }
        }
    }

    return result;

};

}
