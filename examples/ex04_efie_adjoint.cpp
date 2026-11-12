// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file Example 4: Adjoint shape sensitivity of the radar cross-section of a PEC sphere with the
* TEFIE. Demonstrates the analytical shape derivatives of the discretized EFIE, the adjoint
* variable method, and the sensitivity of the RCS with respect to the radius of the sphere.
*/

#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>
#include <cmath>

#include "types.hpp"
#include "constants.hpp"

#include "matrix/eigen_matrix.hpp"

#include "geometry/structure.hpp"
#include "geometry/mesh/mesh_transfer.hpp"
#include "geometry/mesh/triangle_mesh.hpp"
#include "geometry/point_cloud.hpp"

#include "rwg/integral_equations/tefie.hpp"

#include "rwg/excitations/plane_wave.hpp"

#include "rwg/function_space.hpp"

// This example builds on example 1, so it is recommended to go through that first. Here, we use
// the same PEC sphere and plane wave, and ask a different question: how does the monostatic
// (backscatter) radar cross-section change when the geometry of the sphere changes?

// The answer is given by shape derivatives. Following J. Kataja and J. I. Toivanen, "On shape
// differentiation of discretized electric field integral equation", Eng. Anal. Bound. Elem. 37
// (2013) 1197-1203, OpenBEM can compute the derivatives of the discretized TEFIE matrix, of the
// excitation vector, and of the field projectors with respect to the coordinates of every mesh
// vertex, analytically. Combined with the adjoint variable method, this gives the sensitivity of
// the RCS with respect to every vertex of the mesh at the cost of one additional (adjoint) solve.


/**
* @brief Solves the TEFIE for the plane wave and returns the backscatter RCS.
*/
template <typename MatrixType>
bem::Float backscatter_rcs(
    bem::rwg::Tefie<MatrixType>& tefie,
    const bem::Float f,
    const bem::Material& material,
    bem::rwg::ExcitationBase& exc,
    const bem::PointCloud<3>& point,
    const bem::Float dist
    )
{
    MatrixType Z = tefie.j_matrix(f, material);
    MatrixType inc = tefie.exc_matrix(f, material, exc);
    MatrixType j;
    Z.factorize();
    Z.mat_solve(j, inc);

    MatrixType e;
    e.set_matmul(tefie.j_projector(f, material, point), j);

    return bem::four_pi * dist * dist * e.raw_matrix().squaredNorm();
}


int main(int argc, char** argv)
{

    std::cout << "\n====================================================" << std::endl;
    std::cout << "OpenBEM example 4" << std::endl;
    std::cout << "====================================================\n" << std::endl;

    using Clock = std::chrono::steady_clock;
    auto seconds = [] (Clock::time_point t0)
    { return std::chrono::duration<double>(Clock::now() - t0).count(); };

    // --- Geometry and problem setup, exactly as in example 1 ------------------------------------

    std::size_t path_pos = std::string(__FILE__).find_last_of("/");
    std::string path = std::string(__FILE__).substr(0, path_pos) + "/";

    bem::Structure<bem::TriangleMesh<3>> structure;
    bem::MeshTransfer::read_gmsh_v2(structure, path + "msh/sphere.msh");

    bem::TriangleMesh<3>& mesh = structure.mesh();

    std::cout << "Number of vertices: " << mesh.num_verts() << std::endl;
    std::cout << "Number of triangles: " << mesh.num_elems() << std::endl;
    std::cout << "Number of edges: " << mesh.num_edges() << std::endl;

    bem::Float f = 250e6;
    const bem::Material& material = structure.background_material();

    using MatrixType = bem::EigenMatrix<bem::Complex>;

    // The `Tefie` object stores a reference to the mesh. If the mesh vertices are moved later on,
    // the `Tefie` automatically computes its operators on the moved geometry.

    bem::rwg::Tefie<MatrixType> tefie (mesh);

    bem::EigColVecN<bem::Float, 3> dir = { 0, 0, 1 };
    bem::EigColVecN<bem::Float, 3> pol_e = { 1, 0, 0 };
    bem::Float dist = 100 * (bem::c0 / f);
    bem::EigColVecN<bem::Float, 3> pos = -dir * dist;
    bem::EigColVecN<bem::Complex, 1> amp_e { 1 };

    bem::rwg::RwgPlaneWave pw_e (dir, pol_e, pos, amp_e);

    // The objective is the monostatic RCS, i.e., the RCS in the direction from which the plane
    // wave arrives. The plane wave originates from `pos`, so we project the scattered field there.

    bem::PointCloud<3> backscatter_point (pos);

    // We also define an arc of observation points as in example 1, to look at the sensitivity of
    // the bistatic RCS at all angles.

    const bem::Index num_arc = 37;
    bem::EigColVecN<bem::Float, 3> arc_begin = { dist, 0, 0 };
    bem::EigColVecN<bem::Float, 3> arc_end = { dist, 0, bem::pi };
    bem::EigColVecN<bem::Float, 3> center_arc = { 0, 0, 0 };
    bem::EigColVecN<bem::Index, 3> num_pts = { 1, 1, num_arc };

    bem::PointCloud<3> arc;
    arc.set_polar_data(arc_begin, arc_end, center_arc, num_pts);

    // --- Forward problem ------------------------------------------------------------------------

    auto t0 = Clock::now();

    MatrixType Z = tefie.j_matrix(f, material);
    MatrixType inc = tefie.exc_matrix(f, material, pw_e);

    MatrixType j;
    Z.factorize();
    Z.mat_solve(j, inc);

    // Projected E-field at the backscatter point (3 x 1), and the RCS
    MatrixType P = tefie.j_projector(f, material, backscatter_point);
    MatrixType e;
    e.set_matmul(P, j);

    const bem::Float rcs_scale = bem::four_pi * dist * dist;
    const bem::Float rcs = rcs_scale * e.raw_matrix().squaredNorm();

    std::cout << "\nForward solve: " << seconds(t0) << " s" << std::endl;
    std::cout << "Backscatter RCS: " << rcs << " m^2" << std::endl;

    // --- Adjoint problem --------------------------------------------------------------------------

    // With sigma = 4 pi d^2 e^H e and e = P j, the derivative of sigma along any deformation is
    //   d sigma = Re[ mu^T (d inc - dZ j) ] + 8 pi d^2 Re[ e^H dP j ],
    // where mu solves the adjoint problem Z^T mu = 8 pi d^2 P^T conj(e) (paper eqs. 11 and 13).
    // The adjoint problem reuses the LU factors of Z.

    t0 = Clock::now();

    MatrixType rhs_adj;
    rhs_adj.raw_matrix() = 2 * rcs_scale * P.raw_matrix().transpose() * e.raw_matrix().conjugate();

    MatrixType mu;
    Z.mat_solve_transpose(mu, rhs_adj);

    std::cout << "\nAdjoint solve: " << seconds(t0) << " s" << std::endl;

    // --- Shape sensitivities ----------------------------------------------------------------------

    // Each of the following calls visits every element pair (or element) once, evaluates the
    // analytical shape derivatives, and contracts them directly with the given vectors, giving the
    // derivative with respect to the x, y, z coordinates of every mesh vertex (3 x num_verts).

    t0 = Clock::now();

    bem::EigMatNX<bem::Complex, 3> sens_Z = tefie.j_matrix_shape_sensitivity(f, material, j, mu);
    bem::EigMatNX<bem::Complex, 3> sens_inc = tefie.exc_matrix_shape_sensitivity(f, material, pw_e, mu);

    MatrixType e_conj;
    e_conj.raw_matrix() = e.raw_matrix().conjugate();
    bem::EigMatNX<bem::Complex, 3> sens_P = tefie.j_projector_shape_sensitivity(
        f, material, backscatter_point, j, e_conj
        );

    bem::EigMatNX<bem::Float, 3> drcs_dr = (sens_inc - sens_Z).real() + 2 * rcs_scale * sens_P.real();

    std::cout << "Shape sensitivities (all vertices): " << seconds(t0) << " s" << std::endl;

    // --- Sensitivity with respect to the radius ---------------------------------------------------

    // Changing the radius moves every vertex along its radial direction. This is the velocity
    // field of the deformation, and the derivative of the RCS with respect to the radius is the
    // sum over the vertices of the velocity dotted with the vertex sensitivity.

    bem::EigColVecN<bem::Float, 3> center = mesh.verts().rowwise().mean();
    bem::EigMatNX<bem::Float, 3> velocity = mesh.verts().colwise() - center;
    bem::Float radius = velocity.colwise().norm().mean();
    velocity.colwise().normalize();

    bem::Float drcs_dradius = (velocity.array() * drcs_dr.array()).sum();

    std::cout << "\nRadius: " << radius << " m (ka = " << material.k(f).real() * radius << ")" << std::endl;
    std::cout << std::setprecision(10);
    std::cout << "d(RCS)/d(radius), adjoint method:      " << drcs_dradius << " m^2/m" << std::endl;

    // --- Verification 1: direct method ------------------------------------------------------------

    // For a single design variable, the derivative can also be computed directly by assembling
    // the shape-derivative matrices for the radial velocity field. This also gives the
    // sensitivity of the bistatic RCS at every angle, with one extra back-substitution.

    t0 = Clock::now();

    MatrixType dZ = tefie.j_matrix_shape_derivative(f, material, velocity);
    MatrixType dinc = tefie.exc_matrix_shape_derivative(f, material, pw_e, velocity);

    MatrixType rhs_dir = dinc;
    rhs_dir.add_matmul(dZ, j, -bem::one);

    MatrixType dj;
    Z.mat_solve(dj, rhs_dir);

    MatrixType dP = tefie.j_projector_shape_derivative(f, material, backscatter_point, velocity);
    MatrixType de;
    de.set_matmul(dP, j);
    de.add_matmul(P, dj);

    bem::Float drcs_dradius_direct = 2 * rcs_scale * (e.raw_matrix().adjoint() * de.raw_matrix())(0, 0).real();

    std::cout << "d(RCS)/d(radius), direct method:       " << drcs_dradius_direct << " m^2/m  ("
              << seconds(t0) << " s)" << std::endl;

    // --- Verification 2: finite differences -------------------------------------------------------

    // Finally, we move every vertex radially by +-h and re-solve the whole problem.

    t0 = Clock::now();

    const bem::EigMatNX<bem::Float, 3> verts0 = mesh.verts();
    const bem::EigMatNX<bem::Index, 3> elems0 = mesh.elems();
    const bem::EigRowVec<bem::Index> tags0 = mesh.elem_tags();
    const bem::Float h = 1e-4 * radius;

    mesh.set_data(verts0 + h * velocity, elems0, tags0);
    bem::Float rcs_plus = backscatter_rcs(tefie, f, material, pw_e, backscatter_point, dist);

    mesh.set_data(verts0 - h * velocity, elems0, tags0);
    bem::Float rcs_minus = backscatter_rcs(tefie, f, material, pw_e, backscatter_point, dist);

    mesh.set_data(verts0, elems0, tags0);

    bem::Float drcs_dradius_fd = (rcs_plus - rcs_minus) / (2 * h);

    std::cout << "d(RCS)/d(radius), finite differences:  " << drcs_dradius_fd << " m^2/m  ("
              << seconds(t0) << " s)" << std::endl;
    std::cout << std::setprecision(4);
    std::cout << "Relative difference adjoint vs. FD: "
              << std::abs(drcs_dradius - drcs_dradius_fd) / std::abs(drcs_dradius_fd) << std::endl;

    // --- Bistatic RCS sensitivity -----------------------------------------------------------------

    MatrixType P_arc = tefie.j_projector(f, material, arc);
    MatrixType dP_arc = tefie.j_projector_shape_derivative(f, material, arc, velocity);

    MatrixType e_arc, de_arc;
    e_arc.set_matmul(P_arc, j);
    de_arc.set_matmul(dP_arc, j);
    de_arc.add_matmul(P_arc, dj);

    std::cout << "\nBistatic RCS and its sensitivity to the radius (phi = 0):" << std::endl;
    std::cout << std::setw(10) << "theta" << std::setw(16) << "RCS [m^2]" << std::setw(20)
              << "dRCS/dR [m^2/m]" << std::endl;
    for (bem::Index pt = 0; pt < num_arc; pt += 3)
    {
        auto e_pt = e_arc.raw_matrix().middleRows(3 * pt, 3);
        auto de_pt = de_arc.raw_matrix().middleRows(3 * pt, 3);
        std::cout << std::setw(10) << 180.0 * pt / (num_arc - 1)
                  << std::setw(16) << rcs_scale * e_pt.squaredNorm()
                  << std::setw(20) << 2 * rcs_scale * (e_pt.adjoint() * de_pt)(0, 0).real() << std::endl;
    }

    // --- Output -----------------------------------------------------------------------------------

    // Surface sensitivity map: the derivative of the backscatter RCS with respect to an outward
    // normal displacement of each vertex, divided by the area associated with the vertex, and then
    // averaged onto the triangles for visualization in Gmsh.

    bem::EigRowVec<bem::Float> vertex_area = bem::EigRowVec<bem::Float>::Zero(1, mesh.num_verts());
    for (bem::Index elem = 0; elem < mesh.num_elems(); ++elem)
        for (uint8_t vert = 0; vert < 3; ++vert)
            vertex_area[mesh.elems(vert, elem)] += mesh.elem_primitive(elem).area() / 3;

    bem::EigRowVec<bem::Float> vertex_sens =
        (velocity.array() * drcs_dr.array()).colwise().sum() / vertex_area.array();

    bem::EigRowVec<bem::Float> elem_sens (1, mesh.num_elems());
    for (bem::Index elem = 0; elem < mesh.num_elems(); ++elem)
        elem_sens[elem] = (vertex_sens[mesh.elems(0, elem)] + vertex_sens[mesh.elems(1, elem)] +
                           vertex_sens[mesh.elems(2, elem)]) / 3;

    bem::MeshTransfer::write_gmsh_v2_scalar_field(
        structure, path + "msh/ex04_rcs_sensitivity", elem_sens, "dRCS/dn [m^2/m^3]"
        );

    bem::EigMatNX<bem::Float, 3> j_surf = bem::Rwg::reconstruct_field(
        mesh, j, mesh.elem_centroids()
        ).array().real();
    bem::MeshTransfer::write_gmsh_v2_vector_field(structure, path + "msh/ex04_jsurf_initial", j_surf);

    // Deformed sphere: change the radius by 5% in the direction that reduces the backscatter RCS,
    // and compare the RCS of the deformed sphere with the linear prediction from the sensitivity.

    bem::Float d_radius = -0.05 * radius * (drcs_dradius > 0 ? 1 : -1);
    mesh.set_data(verts0 + d_radius * velocity, elems0, tags0);

    MatrixType Z_def = tefie.j_matrix(f, material);
    MatrixType inc_def = tefie.exc_matrix(f, material, pw_e);
    MatrixType j_def;
    Z_def.factorize();
    Z_def.mat_solve(j_def, inc_def);

    MatrixType e_def;
    e_def.set_matmul(tefie.j_projector(f, material, backscatter_point), j_def);
    bem::Float rcs_def = rcs_scale * e_def.raw_matrix().squaredNorm();

    std::cout << "\nRadius changed by " << d_radius << " m:" << std::endl;
    std::cout << "  backscatter RCS " << rcs << " -> " << rcs_def << " m^2 ("
              << 10 * std::log10(rcs_def / rcs) << " dB)" << std::endl;
    std::cout << "  linear prediction: " << rcs + drcs_dradius * d_radius << " m^2" << std::endl;

    bem::EigMatNX<bem::Float, 3> j_surf_def = bem::Rwg::reconstruct_field(
        mesh, j_def, mesh.elem_centroids()
        ).array().real();
    bem::MeshTransfer::write_gmsh_v2_vector_field(structure, path + "msh/ex04_jsurf_deformed", j_surf_def);

    // This creates three `.pos` files in the `msh` folder: the surface current on the original
    // and on the deformed sphere, and the RCS sensitivity map on the original sphere.

    return 0;

}
