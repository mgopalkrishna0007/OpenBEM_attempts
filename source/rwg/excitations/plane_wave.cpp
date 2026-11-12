// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Functionality for computing plane wave excitation vector coefficients for RWG-based BEM systems.
*/

#include "rwg/excitations/plane_wave.hpp"

#include <external/Eigen/Dense>

#include "types.hpp"
#include "geometry/primitives/triangle.hpp"
#include "rwg/function_space.hpp"


namespace bem::rwg
{

EigMat<Complex> RwgPlaneWave::compute(const Complex k, const Triangle<3>& obs_tri)
{
    Index rhs = 0;

    auto eval_lambda = [&] (ConstEigRef<EigMatNX<Float, 3>> r_obs) -> EigMatNX<Complex, 3>
        { return eval(k, r_obs, rhs); };

    EigMatNX<Complex, 3> result = EigMatNX<Complex, 3>::Zero(3, pos_.cols());

    for (rhs = 0; rhs < pos_.cols(); ++rhs)
        result.col(rhs) = Rwg::test_field(obs_tri, eval_lambda, *tri_quad_);

    return result;
};


EigMat<Complex> RwgPlaneWave::compute_shape(const Complex k, const Triangle<3>& obs_tri)
{
    QuadratureData<3> qd = tri_quad_->compute(obs_tri);

    // Tangential gradients of the barycentric coordinates and their values at the quadrature points
    const EigColVecN<Float, 3> nrm = obs_tri.normal();
    const Float twice_area = 2 * obs_tri.area();
    EigMatMN<Float, 3, 3> grad_lambda;
    EigMatNX<Float, 3> lambda (3, qd.points.cols());
    for (uint8_t vert = 0; vert < 3; ++vert)
    {
        grad_lambda.col(vert) = nrm.cross(
            obs_tri.v((vert + 2) % 3) - obs_tri.v((vert + 1) % 3)
            ) / twice_area;
        lambda.row(vert) = (
            grad_lambda.col(vert).transpose() * (qd.points.colwise() - obs_tri.v(vert))
            ).array() + one;
    }

    EigMat<Complex> result = EigMat<Complex>::Zero(3, 9 * pos_.cols());

    for (Index rhs = 0; rhs < pos_.cols(); ++rhs)
    {
        const EigMatNX<Complex, 3> field = eval(k, qd.points, rhs);
        const EigColVecN<Complex, 3> jk_dir = J * k * dir_.col(rhs).cast<Complex>();

        for (uint8_t edge = 0; edge < 3; ++edge)
        {
            const EigMatNX<Float, 3> f = Rwg::value(obs_tri, qd.points, edge);
            const EigRowVec<Complex> f_dot_e = (f.cast<Complex>().array() * field.array()).colwise().sum();

            for (uint8_t vert = 0; vert < 3; ++vert)
            {
                const EigRowVec<Float> grad_dot_f = grad_lambda.col(vert).transpose() * f;
                for (uint8_t comp = 0; comp < 3; ++comp)
                {
                    const EigRowVec<Complex> integrand =
                        field.row(comp).array() * grad_dot_f.array().cast<Complex>()
                        - jk_dir[comp] * f_dot_e.array() * lambda.row(vert).array().cast<Complex>();
                    result(edge, 9 * rhs + 3 * vert + comp) = (integrand * qd.weights.transpose())(0, 0);
                }
            }
        }
    }

    return result;
};


EigMat<Complex> NxRwgPlaneWave::compute(const Complex k, const Triangle<3>& obs_tri)
{
    Index rhs = 0;

    auto eval_lambda = [&] (ConstEigRef<EigMatNX<Float, 3>> r_obs) -> EigMatNX<Complex, 3>
        { return eval(k, r_obs, rhs); };

    EigMatNX<Complex, 3> result = EigMatNX<Complex, 3>::Zero(3, pos_.cols());

    for (rhs = 0; rhs < pos_.cols(); ++rhs)
        result.col(rhs) = NxRwg::test_field(obs_tri, eval_lambda, *tri_quad_);

    return result;
};

}

