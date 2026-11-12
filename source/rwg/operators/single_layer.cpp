// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* RWG-based single-layer potential BEM operators.
*/

#include "rwg/operators/single_layer.hpp"

#include <external/Eigen/Dense>

#include "types.hpp"
#include "constants.hpp"
#include "geometry/primitives/triangle.hpp"
#include "rwg/function_space.hpp"
#include "rwg/integrators/obs/base.hpp"


namespace bem::rwg
{

namespace
{

// Polynomials in the observation coordinates (x, y, z) and in the source coordinates (x', y'),
// in the local coordinate system of the source triangle. Coefficients are stored in the
// monomial order used by `ObsResult::rs_g` and `ObsResult::grad_g_shape`.
using PolyObs = EigColVecN<Float, 10>; // 1, x, y, z, x^2, xy, xz, y^2, yz, z^2
using PolySrc = EigColVecN<Float, 6>;  // 1, x', y', x'^2, x'y', y'^2

PolyObs lin_obs(const Float c0, ConstEigRef<EigColVecN<Float, 3>> g)
{
    PolyObs p = PolyObs::Zero();
    p[0] = c0; p[1] = g[0]; p[2] = g[1]; p[3] = g[2];
    return p;
}

PolySrc lin_src(const Float c0, const Float gx, const Float gy)
{
    PolySrc p = PolySrc::Zero();
    p[0] = c0; p[1] = gx; p[2] = gy;
    return p;
}

// Product of two linear observation polynomials
PolyObs mul_obs(const PolyObs& a, const PolyObs& b)
{
    static const uint8_t quad[3][3] = {{4, 5, 6}, {5, 7, 8}, {6, 8, 9}};
    PolyObs p = PolyObs::Zero();
    p[0] = a[0] * b[0];
    for (uint8_t i = 0; i < 3; ++i)
    {
        p[1 + i] = a[0] * b[1 + i] + b[0] * a[1 + i];
        for (uint8_t j = 0; j < 3; ++j)
            p[quad[i][j]] += a[1 + i] * b[1 + j];
    }
    return p;
}

// Product of two linear source polynomials
PolySrc mul_src(const PolySrc& a, const PolySrc& b)
{
    static const uint8_t quad[2][2] = {{3, 4}, {4, 5}};
    PolySrc p = PolySrc::Zero();
    p[0] = a[0] * b[0];
    for (uint8_t i = 0; i < 2; ++i)
    {
        p[1 + i] = a[0] * b[1 + i] + b[0] * a[1 + i];
        for (uint8_t j = 0; j < 2; ++j)
            p[quad[i][j]] += a[1 + i] * b[1 + j];
    }
    return p;
}

// int int a(r) b(r') G for linear a and b, from the moments in ObsResult::rs_g arranged as a
// 4 x 3 matrix (observation monomials 1, x, y, z by source monomials 1, x', y')
Complex int_g(const EigMatMN<Complex, 4, 3>& gm, const PolyObs& a, const PolySrc& b)
{
    return (a.head<4>().transpose().cast<Complex>() * gm * b.head<3>().cast<Complex>())(0, 0);
}

// Tangential gradient and offset of the barycentric coordinate of vertex `a`: lambda(r) = offset + grad . r
void barycentric(const Triangle<3>& tri, const uint8_t a, EigColVecN<Float, 3>& grad, Float& offset)
{
    const EigColVecN<Float, 3> c = (tri.v(1) - tri.v(0)).cross(tri.v(2) - tri.v(0));
    const Float twice_area = c.norm();
    grad = (c / twice_area).cross(tri.v((a + 2) % 3) - tri.v((a + 1) % 3)) / twice_area;
    offset = one - grad.dot(tri.v(a));
    return;
}

}


EigMat<Complex> VectorSingleLayerOp::compute(
    const Complex k,
    const Triangle<3>& obs_tri,
    const Triangle<3>& src_tri
    ) const
{
    Triangle<3> obs_tri_local;
    Triangle<2> src_tri_local;
    transform_coordinates(obs_tri_local, src_tri_local, obs_tri, src_tri);

    const ObsResult obs_result = obs_integrator_->integrate(
        k, obs_tri_local, src_tri_local, false, true, false, false
        );

    return assemble(k, obs_tri_local, src_tri_local.to_3d(), obs_result);
};


EigMat<Complex> VectorSingleLayerOp::assemble(
    const Complex k,
    const Triangle<3>& obs_tri,
    const Triangle<3>& src_tri,
    const ObsResult& obs_result
    ) const
{
    EigMatMN<Complex, 3, 3> result = EigMatMN<Complex, 3, 3>::Zero(3, 3);

    const EigRowVecN<Complex, 12>& I = obs_result.rs_g;

    // source triangle edges
    for (uint8_t jj = 0; jj < 3; ++jj)
    {
        Float xn = src_tri.v((jj + 2) % 3)[0];
        Float yn = src_tri.v((jj + 2) % 3)[1];

        // observer triangle edges
        for (uint8_t ii = 0; ii < 3; ++ii)
        {
            Float xm = obs_tri.v((ii + 2) % 3)[0];
            Float ym = obs_tri.v((ii + 2) % 3)[1];

            result(ii, jj) = (xm * xn + ym * yn) * I[0]
                - xm * I[1] - ym * I[2]
                - xn * I[3] - yn * I[4]
                + I[6] + I[10];
        }
    }

    result.array() *= (
        Rwg::normalization(obs_tri).transpose() * Rwg::normalization(src_tri)
        ).array();

    return result;
};


EigMat<Complex> RotVectorSingleLayerOp::compute(
    const Complex k,
    const Triangle<3>& obs_tri,
    const Triangle<3>& src_tri
    ) const
{
    Triangle<3> obs_tri_local;
    Triangle<2> src_tri_local;
    transform_coordinates(obs_tri_local, src_tri_local, obs_tri, src_tri);

    const ObsResult obs_result = obs_integrator_->integrate(
        k, obs_tri_local, src_tri_local, false, true, false, false
        );

    return assemble(k, obs_tri_local, src_tri_local.to_3d(), obs_result);
};


EigMat<Complex> RotVectorSingleLayerOp::assemble(
    const Complex k,
    const Triangle<3>& obs_tri,
    const Triangle<3>& src_tri,
    const ObsResult& obs_result
    ) const
{
    EigMatMN<Complex, 3, 3> result = EigMatMN<Complex, 3, 3>::Zero(3, 3);

    const EigRowVecN<Complex, 12>& I = obs_result.rs_g;

    Float nx = obs_tri.normal()[0];
    Float ny = obs_tri.normal()[1];
    Float nz = obs_tri.normal()[2];

    // source triangle edges
    for (uint8_t jj = 0; jj < 3; ++jj)
    {
        Float xn = src_tri.v((jj + 2) % 3)[0];
        Float yn = src_tri.v((jj + 2) % 3)[1];

        // observer triangle edges
        for (uint8_t ii = 0; ii < 3; ++ii)
        {
            Float xm = obs_tri.v((ii + 2) % 3)[0];
            Float ym = obs_tri.v((ii + 2) % 3)[1];
            Float zm = obs_tri.v((ii + 2) % 3)[2];

            result(ii, jj) = -(
                (ny * zm - nz * ym) * I[1]
                + (nz * xm - nx * zm) * I[2]
                + (nx * zm * yn - ny* zm * xn + nz * ym * xn - nz * xm * yn) * I[0]
                + nz * yn * I[3] - nz * xn * I[4] + (ny * xn - nx * yn) * I[5]
                + nz * I[7] - nz * I[9] - ny * I[8] + nx * I[11]
            );
        }
    }

    result.array() *= (
        NxRwg::normalization(obs_tri).transpose() * Rwg::normalization(src_tri)
        ).array();

    return result;
};


EigMat<Complex> ScalarSingleLayerOp::compute(
    const Complex k,
    const Triangle<3>& obs_tri,
    const Triangle<3>& src_tri
    ) const
{
    Triangle<3> obs_tri_local;
    Triangle<2> src_tri_local;
    transform_coordinates(obs_tri_local, src_tri_local, obs_tri, src_tri);

    const ObsResult obs_result = obs_integrator_->integrate(
        k, obs_tri_local, src_tri_local, true, false, false, false
        );

    return assemble(k, obs_tri_local, src_tri_local.to_3d(), obs_result);
};


EigMat<Complex> ScalarSingleLayerOp::assemble(
    const Complex k,
    const Triangle<3>& obs_tri,
    const Triangle<3>& src_tri,
    const ObsResult& obs_result
    ) const
{
    EigMatMN<Complex, 1, 1> result = EigMatMN<Complex, 1, 1>::Zero(1, 1);

    result[0] = obs_result.g;
    result[0] *= Pulse::normalization(obs_tri)[0] * Pulse::normalization(src_tri)[0];

    return result;
};


EigMat<Complex> RotGradScalarSingleLayerOp::compute(
    const Complex k,
    const Triangle<3>& obs_tri,
    const Triangle<3>& src_tri
    ) const
{
    Triangle<3> obs_tri_local;
    Triangle<2> src_tri_local;
    transform_coordinates(obs_tri_local, src_tri_local, obs_tri, src_tri);

    const ObsResult obs_result = obs_integrator_->integrate(
        k, obs_tri_local, src_tri_local, false, false, true, false
        );

    return assemble(k, obs_tri_local, src_tri_local.to_3d(), obs_result);
};


EigMat<Complex> RotGradScalarSingleLayerOp::assemble(
    const Complex k,
    const Triangle<3>& obs_tri,
    const Triangle<3>& src_tri,
    const ObsResult& obs_result
    ) const
{
    EigMatMN<Complex, 3, 1> result = EigMatMN<Complex, 3, 1>::Zero(3, 1);

    if (GeometryOps<3>::common_vertices(obs_tri, src_tri) == 3)
        return result;

    const EigRowVecN<Complex, 9>& I = obs_result.grad_g;

    Float nx = obs_tri.normal()[0];
    Float ny = obs_tri.normal()[1];
    Float nz = obs_tri.normal()[2];

    // observer triangle edges
    for (uint8_t ii = 0; ii < 3; ++ii)
    {
        Float xm = obs_tri.v((ii + 2) % 3)[0];
        Float ym = obs_tri.v((ii + 2) % 3)[1];
        Float zm = obs_tri.v((ii + 2) % 3)[2];

        result[ii] = -(
            (zm * ny - ym * nz) * I[0] + (xm * nz - zm * nx) * I[3]
            + nz * I[1] - ny * I[2] - nz * I[4] + nx * I[5]
            + (ym * nx - ny * xm) * I[6]
            + ny * I[7] - nx * I[8]
            );

        // result[ii] = -(-nx * ((I[3] * zm - I[6] * ym) - (I[5] - I[8]))
        //     - ny * ((I[6] * xm - I[0] * zm) - (I[7] - I[2]))
        //     - nz * ((I[0] * ym - I[3] * xm) - (I[1] - I[4])));
    }

    result.array() *= NxRwg::normalization(obs_tri).array() * Pulse::normalization(src_tri)[0];

    return result;

};


EigMat<Complex> VectorHypersingularOp::compute(
    const Complex k,
    const Triangle<3>& obs_tri,
    const Triangle<3>& src_tri
    ) const
{
    Triangle<3> obs_tri_local;
    Triangle<2> src_tri_local;
    transform_coordinates(obs_tri_local, src_tri_local, obs_tri, src_tri);

    const ObsResult obs_result = obs_integrator_->integrate(
        k, obs_tri_local, src_tri_local, true, true, false, false
        );

    return assemble(k, obs_tri_local, src_tri_local.to_3d(), obs_result);
};


EigMat<Complex> VectorHypersingularOp::assemble(
    const Complex k,
    const Triangle<3>& obs_tri,
    const Triangle<3>& src_tri,
    const ObsResult& obs_result
    ) const
{
    EigMatMN<Complex, 3, 3> result = op_g_.assemble(k, obs_tri, src_tri, obs_result);
    Complex hessg_term = op_hessg_.assemble(k, obs_tri, src_tri, obs_result)(0, 0) / k / k;
    result -= hessg_term * obs_tri.edge_polarities().transpose() * src_tri.edge_polarities();
    return result;
};


EigMat<Complex> RotVectorHypersingularOp::compute(
    const Complex k,
    const Triangle<3>& obs_tri,
    const Triangle<3>& src_tri
    ) const
{
    Triangle<3> obs_tri_local;
    Triangle<2> src_tri_local;
    transform_coordinates(obs_tri_local, src_tri_local, obs_tri, src_tri);

    const ObsResult obs_result = obs_integrator_->integrate(
        k, obs_tri_local, src_tri_local, false, true, true, false
        );

    return assemble(k, obs_tri_local, src_tri_local.to_3d(), obs_result);
};


EigMat<Complex> RotVectorHypersingularOp::assemble(
    const Complex k,
    const Triangle<3>& obs_tri,
    const Triangle<3>& src_tri,
    const ObsResult& obs_result
    ) const
{
    EigMatMN<Complex, 3, 3> result = op_g_.assemble(k, obs_tri, src_tri, obs_result);
    EigMatMN<Complex, 3, 1> hessg_term = op_hessg_.assemble(k, obs_tri, src_tri, obs_result) / k / k;
    result += hessg_term * src_tri.edge_polarities();
    return result;
};


EigMat<Complex> VectorHypersingularShapeOp::compute(
    const Complex k,
    const Triangle<3>& obs_tri,
    const Triangle<3>& src_tri
    ) const
{
    Triangle<3> obs_tri_local;
    Triangle<2> src_tri_local;
    OperatorBase::transform_coordinates(obs_tri_local, src_tri_local, obs_tri, src_tri);

    const ObsResult obs_result = obs_integrator_->integrate(
        k, obs_tri_local, src_tri_local, false, true, false, false, true
        );

    EigMat<Complex> result = assemble(k, obs_tri_local, src_tri_local.to_3d(), obs_result);

    // Vertex gradients back to the global coordinate system
    const EigMatMN<Float, 3, 3> local_uvw = src_tri.local_coordinate_basis();
    for (uint8_t vert = 0; vert < 6; ++vert)
        result.middleCols(3 * vert, 3) = (result.middleCols(3 * vert, 3) * local_uvw.transpose()).eval();

    return result;
};


EigMat<Complex> VectorHypersingularShapeOp::assemble(
    const Complex k,
    const Triangle<3>& obs_tri,
    const Triangle<3>& src_tri,
    const ObsResult& obs_result
    ) const
{
    EigMat<Complex> result = EigMat<Complex>::Zero(9, 18);

    // Pairs skipped by the integrator (e.g. beyond the skin-depth threshold) contribute nothing
    if (obs_result.grad_g_shape.size() == 0)
        return result;

    // Moments of G: rs_g layout is 1|1, 1|x', 1|y', x|1, y|1, z|1, x|x', y|x', z|x', x|y', y|y', z|y'
    const EigRowVecN<Complex, 12>& I = obs_result.rs_g;
    EigMatMN<Complex, 4, 3> gm;
    gm << I[0], I[1], I[2],
          I[3], I[6], I[9],
          I[4], I[7], I[10],
          I[5], I[8], I[11];

    // Moments of d_c G as 10 x 6 matrices (observation by source monomials)
    EigMatMN<Complex, 10, 6> hm[3];
    for (uint8_t c = 0; c < 3; ++c)
        for (uint8_t i = 0; i < 10; ++i)
            for (uint8_t j = 0; j < 6; ++j)
                hm[c](i, j) = obs_result.grad_g_shape(c, 6 * i + j);

    const EigRowVecN<Float, 3> s = Rwg::normalization(obs_tri);
    const EigRowVecN<Float, 3> t = Rwg::normalization(src_tri);
    const Complex inv_k_sq = one / (k * k);

    // Components of the source RWG functions f_n(r') = t (r' - p_n); the local z component is zero
    PolySrc f_p[3][2];
    for (uint8_t jj = 0; jj < 3; ++jj)
    {
        const EigColVecN<Float, 3> pn = src_tri.v((jj + 2) % 3);
        for (uint8_t d = 0; d < 2; ++d)
            f_p[jj][d] = lin_src(-t[jj] * pn[d], d == 0 ? t[jj] : 0, d == 1 ? t[jj] : 0);
    }

    // Contractions of the gradient moments with the source polynomials that do not depend on the vertex
    EigColVecN<Complex, 10> h_fp[3][3][2];
    for (uint8_t c = 0; c < 3; ++c)
        for (uint8_t jj = 0; jj < 3; ++jj)
            for (uint8_t d = 0; d < 2; ++d)
                h_fp[c][jj][d] = hm[c] * f_p[jj][d].cast<Complex>();

    for (uint8_t vert = 0; vert < 6; ++vert)
    {
        const bool obs_vert = vert < 3;

        // Hat function of the moved vertex on the triangle it belongs to: lambda = offset + grad . r
        EigColVecN<Float, 3> grad_lam;
        Float off_lam;
        barycentric(obs_vert ? obs_tri : src_tri, obs_vert ? vert : vert - 3, grad_lam, off_lam);

        if (obs_vert)
        {
            const PolyObs lam_q = lin_obs(off_lam, grad_lam);

            for (uint8_t ii = 0; ii < 3; ++ii)
            {
                const EigColVecN<Float, 3> qm = obs_tri.v((ii + 2) % 3);

                // f_m(r) = s (r - q_m), (grad lambda(r) . f_m), and f_m lambda(r)
                PolyObs f_q[3], f_q_lam[2];
                for (uint8_t d = 0; d < 3; ++d)
                    f_q[d] = lin_obs(-s[ii] * qm[d], s[ii] * EigColVecN<Float, 3>::Unit(d));
                for (uint8_t d = 0; d < 2; ++d)
                    f_q_lam[d] = mul_obs(f_q[d], lam_q);
                const PolyObs gq_f = lin_obs(-s[ii] * grad_lam.dot(qm), s[ii] * grad_lam);

                for (uint8_t jj = 0; jj < 3; ++jj)
                    for (uint8_t c = 0; c < 3; ++c)
                    {
                        // Jacobian term of eq. (26): (f_m . grad lambda(r)) f_n,c
                        Complex d_i1 = (c < 2) ? int_g(gm, gq_f, f_p[jj][c]) : Complex(0);

                        // Kernel term of eq. (26): (f_m . f_n) d_c G lambda(r)
                        for (uint8_t d = 0; d < 2; ++d)
                            d_i1 += f_q_lam[d].cast<Complex>().dot(h_fp[c][jj][d]);

                        // Eq. (24): div f_m div f_n d_c G lambda(r), div f = 2 s
                        const Complex d_i2 = (2 * s[ii]) * (2 * t[jj]) *
                            lam_q.cast<Complex>().dot(hm[c].col(0));

                        result(3 * ii + jj, 3 * vert + c) = d_i1 - d_i2 * inv_k_sq;
                    }
            }
        }
        else
        {
            const PolySrc lam_p = lin_src(off_lam, grad_lam[0], grad_lam[1]);

            // Contractions with lambda(r') and with f_n lambda(r')
            EigColVecN<Complex, 10> h_lam[3], h_fp_lam[3][3][2];
            for (uint8_t c = 0; c < 3; ++c)
            {
                h_lam[c] = hm[c] * lam_p.cast<Complex>();
                for (uint8_t jj = 0; jj < 3; ++jj)
                    for (uint8_t d = 0; d < 2; ++d)
                        h_fp_lam[c][jj][d] = hm[c] * mul_src(f_p[jj][d], lam_p).cast<Complex>();
            }

            for (uint8_t ii = 0; ii < 3; ++ii)
            {
                const EigColVecN<Float, 3> qm = obs_tri.v((ii + 2) % 3);

                PolyObs f_q[3];
                for (uint8_t d = 0; d < 3; ++d)
                    f_q[d] = lin_obs(-s[ii] * qm[d], s[ii] * EigColVecN<Float, 3>::Unit(d));

                for (uint8_t jj = 0; jj < 3; ++jj)
                {
                    const EigColVecN<Float, 3> pn = src_tri.v((jj + 2) % 3);
                    const PolySrc gp_f = lin_src(
                        -t[jj] * grad_lam.dot(pn), t[jj] * grad_lam[0], t[jj] * grad_lam[1]
                        );

                    for (uint8_t c = 0; c < 3; ++c)
                    {
                        // Jacobian term of eq. (26): f_m,c (grad lambda(r') . f_n)
                        Complex d_i1 = int_g(gm, f_q[c], gp_f);

                        // Kernel term of eq. (26): -(f_m . f_n) d_c G lambda(r')
                        for (uint8_t d = 0; d < 2; ++d)
                            d_i1 -= f_q[d].cast<Complex>().dot(h_fp_lam[c][jj][d]);

                        // Eq. (24): -div f_m div f_n d_c G lambda(r'), div f = 2 s
                        const Complex d_i2 = -(2 * s[ii]) * (2 * t[jj]) * h_lam[c][0];

                        result(3 * ii + jj, 3 * vert + c) = d_i1 - d_i2 * inv_k_sq;
                    }
                }
            }
        }
    }

    return result;
};

}
