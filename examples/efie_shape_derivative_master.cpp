// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file efie_shape_derivative_master.cpp
*
* SINGLE-FILE VERIFICATION OF THE ANALYTICAL EFIE SHAPE DERIVATIVES
* =================================================================
*
* This one file contains everything needed to compute and cross-check the analytical shape
* derivatives of the discretized EFIE (TEFIE) of
*
*   J. Kataja, J. I. Toivanen, "On shape differentiation of discretized electric field integral
*   equation", Eng. Anal. Bound. Elem. 37 (2013) 1197-1203  (eqs. 13, 14, 24, 26, 39, 40).
*
* It is a self-contained port of the parts of OpenBEM that are involved (standard C++ only: no
* Eigen, no JSON, no OpenBEM headers). The algorithms, constants, quadrature rules, thresholds
* and conventions are those of OpenBEM, so the numbers are those of the library.
*
* Build and run from the `examples` directory (OpenMP is optional):
*
*   g++ -O3 -march=native -fopenmp -std=c++17 efie_shape_derivative_master.cpp -o efie_shape_derivative_master
*   ./efie_shape_derivative_master                     # element checks + sphere checks (msh/sphere.msh)
*   ./efie_shape_derivative_master --elements-only     # only the element-pair checks (a few seconds)
*   ./efie_shape_derivative_master path/to/mesh.msh    # sphere checks on another Gmsh v2 ASCII mesh
*
* What is verified
* ----------------
*   Part A (paper Sec. 5.1, Table 1): the derivative of the 3x3 element-pair matrix of the TEFIE
*     for the four configurations "near", "point", "edge", "same", computed with the analytical
*     formulas (eqs. 24, 26), against central finite differences and the paper's forward
*     difference (eq. 40, h = 1e-8) of OpenBEM's own element matrix. Also the derivatives of
*     the plane-wave excitation and of the field projector of a single triangle.
*   Part B (assembled system, ex01 sphere): the assembled derivative dZ/ds for a radial
*     deformation against finite differences of the assembled matrix, and the adjoint
*     sensitivity (eqs. 11, 13) of the monostatic RCS with respect to the sphere radius against
*     the direct method and finite differences of the complete solve.
*
* ================================================================================================
* CHANGES WITH RESPECT TO THE ORIGINAL OPENBEM REPOSITORY (commit e0f667c)
* ================================================================================================
* Every section of this file names the OpenBEM file it reproduces, tagged:
*   [ORIGINAL]  unchanged algorithm, ported from OpenBEM
*   [CHANGED]   existing OpenBEM file/class that was extended for the shape derivatives
*   [NEW]       code that does not exist in the original OpenBEM
* The corresponding repository changes (branch `efie-analytical-shape-derivatives`) are:
*
*  1. kernels/base.hpp, kernels/hgf.hpp/.cpp                                         [CHANGED]
*     + ScalarKernelBase::compute_hess() (default throws), HGF::compute_hess(): Hessian of G.
*  2. rwg/integrators/src/base.hpp                                                   [CHANGED]
*     + SrcResult::rs_grad_g    (int x' grad G, int y' grad G)
*     + SrcResult::rsrs_grad_g  (int x'^2 grad G, int x'y' grad G, int y'^2 grad G)
*     + SrcResult::hess_g, rs_hess_g (int grad grad G, int x' grad grad G, int y' grad grad G)
*     + SrcIntegratorBase::integrate(..., shape_grad_terms = false, shape_hess_terms = false)
*  3. rwg/integrators/src/quadrature.*  + quadrature of the new moments                  [CHANGED]
*  4. rwg/integrators/src/singularity.* + integrate_singular_shape(): closed-form singular  [CHANGED]
*     parts of the new moments for the subtracted kernel (1/R - k^2 R/2)/(4 pi), via the
*     surface divergence theorem; throws for Hessian moments (near field not supported).
*  5. rwg/integrators/src/strategic.*, line.*  + pass-through of the flags; the line   [CHANGED]
*     integrator (edges >= 1 wavelength) throws if shape terms are requested.
*  6. rwg/integrators/obs/base.hpp, quadrature.*, strategic.*                         [CHANGED]
*     + ObsResult::grad_g_shape (3 x 60 moments), integrate(..., shape_terms = false).
*  7. rwg/operators/base.hpp            + ShapeOperatorBase                             [CHANGED]
*  8. rwg/operators/single_layer.*      + VectorHypersingularShapeOp (eqs. 24, 26)       [CHANGED]
*  9. rwg/excitations/base.hpp, plane_wave.*  + compute_shape(), RwgPlaneWave::compute_shape() [CHANGED]
* 10. rwg/projectors/base.hpp, single_layer.* + compute_shape(), VectorHypersingularProj::compute_shape() [CHANGED]
* 11. rwg/assemblers/operator_, excitation_, projector_assembler.*                   [CHANGED]
*     + assemble_shape_derivative() and shape_sensitivity()
* 12. matrix/base.hpp, matrix/eigen_matrix.hpp  + mat_solve_transpose()             [CHANGED]
* 13. rwg/integral_equations/tefie.hpp  + j_matrix_/exc_matrix_/j_projector_shape_derivative()
*                                         and the matching *_shape_sensitivity()      [CHANGED]
* 14. examples/ex04_efie_adjoint.cpp, CMakeLists.txt, CHANGELOG.md, README.md          [NEW]/[CHANGED]
* Nothing else in OpenBEM was modified; all new arguments are optional, so existing behaviour is
* unchanged (ex01 reproduces its 2.4444 % before and after).
*
* Additional differences of THIS FILE only (needed to make it standalone; not repo changes):
*   - Eigen matrices/vectors are replaced by small fixed-size structs and std::vector.
*   - The Xiao-Gimbutas triangle rules that OpenBEM loads from quadrature/tri_quad_xiao_gimbutas.json
*     are embedded for the two orders used by the default settings (4 and 8), digit for digit.
*   - The dense LU (EigenMatrix / Eigen::PartialPivLU) is replaced by a hand-written LU with
*     partial pivoting, with solve() and solve_transpose().
*   - Gmsh reading and edge generation follow MeshTransfer::read_gmsh_v2 and
*     TriangleMesh::generate_edges; only what is needed here is kept.
*   - Line integration (SrcLineIntegrator, used for edges >= 1 wavelength) is not ported; such
*     meshes are rejected with an error.
* ================================================================================================
*/

#include <array>
#include <vector>
#include <complex>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <functional>
#include <limits>
#include <memory>

#ifdef _OPENMP
#include <omp.h>
#endif


namespace bem
{

// =================================================================================================
// [ORIGINAL] types.hpp, constants.hpp
// =================================================================================================

using Float = double;
using Complex = std::complex<double>;
using Index = std::size_t;

const Complex J = Complex(0.0, 1.0);
const Float one = 1.0;
const Float half = 0.5;
const Float pi = M_PI;
const Float two_pi = (Float)(2.0 * M_PI);
const Float four_pi = (Float)(4.0 * M_PI);
const Float eps0 = 8.854187817e-12;
const Float mu0 = four_pi * 1.0e-7;
const Float c0 = 1.0 / std::sqrt(eps0 * mu0);
const Float float_eps = std::numeric_limits<Float>::epsilon();

const Float KERNEL_DEFAULT_TOL = 1.0e-6;                    // kernels/base.hpp
const Float TRIANGLE_DEFAULT_TOL = 1.0e-6;                  // geometry/primitives/triangle.hpp
const Float GEOMETRY_DEFAULT_TOL = std::sqrt(float_eps);    // geometry/operations.hpp


// =================================================================================================
// [NEW, this file only] small vector types replacing Eigen
// =================================================================================================

struct V2 { Float x = 0, y = 0; };
struct V3 { Float x = 0, y = 0, z = 0; };

inline V2 operator+(V2 a, V2 b) { return {a.x + b.x, a.y + b.y}; }
inline V2 operator-(V2 a, V2 b) { return {a.x - b.x, a.y - b.y}; }
inline V2 operator*(Float s, V2 a) { return {s * a.x, s * a.y}; }
inline Float dot(V2 a, V2 b) { return a.x * b.x + a.y * b.y; }
inline Float norm(V2 a) { return std::sqrt(dot(a, a)); }

inline V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V3 operator*(Float s, V3 a) { return {s * a.x, s * a.y, s * a.z}; }
inline Float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline Float norm(V3 a) { return std::sqrt(dot(a, a)); }
inline Float comp(V3 a, int i) { return i == 0 ? a.x : (i == 1 ? a.y : a.z); }
inline V3 unit(int i) { V3 e; if (i == 0) e.x = 1; else if (i == 1) e.y = 1; else e.z = 1; return e; }

using C3 = std::array<Complex, 3>;
using CMat3 = std::array<std::array<Complex, 3>, 3>;

// Dense complex matrix (row-major)
struct CMat
{
    Index rows = 0, cols = 0;
    std::vector<Complex> a;
    CMat() {}
    CMat(Index r, Index c): rows(r), cols(c), a(r * c, 0.0) {}
    Complex& operator()(Index i, Index j) { return a[i * cols + j]; }
    const Complex& operator()(Index i, Index j) const { return a[i * cols + j]; }
};

inline void atomic_add(Complex& t, const Complex& v)
{
    Float* p = reinterpret_cast<Float*>(&t);
#pragma omp atomic
    p[0] += v.real();
#pragma omp atomic
    p[1] += v.imag();
}

inline Float frob(const CMat& m)
{
    Float s = 0;
    for (const Complex& v: m.a) s += std::norm(v);
    return std::sqrt(s);
}


// =================================================================================================
// [ORIGINAL] geometry/primitives/triangle.hpp/.cpp (Triangle<2>, Triangle<3>)
// =================================================================================================

struct Triangle3
{
    V3 v[3];
    Float pol[3] = {1, 1, 1};   // edge polarities
    Float area = 0;
    V3 normal, centroid;

    Triangle3() {}
    Triangle3(V3 a, V3 b, V3 c, const Float* p = nullptr) { set(a, b, c, p); }

    void set(V3 a, V3 b, V3 c, const Float* p = nullptr)
    {
        v[0] = a; v[1] = b; v[2] = c;
        if (p) for (int i = 0; i < 3; ++i) pol[i] = p[i];
        area = 0.5 * norm(cross(v[1] - v[0], v[2] - v[0]));
        V3 orth = cross(v[1] - v[0], v[2] - v[1]);          // Triangle::normal()
        normal = (1.0 / norm(orth)) * orth;
        centroid = (1.0 / 3) * (v[0] + v[1] + v[2]);
    }

    Float edge_length(int i) const { return norm(v[(i + 1) % 3] - v[i]); }
    Float longest_edge_length() const { return std::max({edge_length(0), edge_length(1), edge_length(2)}); }
    Float mean_edge_length() const { return (edge_length(0) + edge_length(1) + edge_length(2)) / 3; }

    // Columns u, v, w of the local coordinate basis (Triangle::local_coordinate_basis)
    void local_coordinate_basis(V3 basis[3]) const
    {
        V3 v10 = v[1] - v[0];
        basis[0] = (1.0 / norm(v10)) * v10;
        V3 nxu = cross(normal, basis[0]);
        basis[1] = (1.0 / norm(nxu)) * nxu;
        basis[2] = normal;
    }
};

struct Triangle2
{
    V2 v[3];
    Float pol[3] = {1, 1, 1};
    Float area = 0;
    V2 centroid;

    void set(V2 a, V2 b, V2 c, const Float* p)
    {
        v[0] = a; v[1] = b; v[2] = c;
        for (int i = 0; i < 3; ++i) pol[i] = p[i];
        area = 0.5 * std::abs(v[0].x * (v[1].y - v[2].y) - v[0].y * (v[1].x - v[2].x)
                              + (v[1].x * v[2].y - v[1].y * v[2].x));
        centroid = (1.0 / 3) * (v[0] + v[1] + v[2]);
    }

    Float edge_length(int i) const { return norm(v[(i + 1) % 3] - v[i]); }
    Float longest_edge_length() const { return std::max({edge_length(0), edge_length(1), edge_length(2)}); }
    Float shortest_edge_length() const { return std::min({edge_length(0), edge_length(1), edge_length(2)}); }

    Triangle3 to_3d() const
    {
        return Triangle3({v[0].x, v[0].y, 0}, {v[1].x, v[1].y, 0}, {v[2].x, v[2].y, 0}, pol);
    }
};

// [ORIGINAL] geometry/operations.cpp: GeometryOps<3>::transform_coordinate_system
inline V3 to_local(V3 p, V3 origin, const V3 basis[3])
{
    V3 d = p - origin;
    return {dot(basis[0], d), dot(basis[1], d), dot(basis[2], d)};
}

// [ORIGINAL] geometry/operations.cpp: GeometryOps<3>::common_vertices
inline int common_vertices(const Triangle3& t1, const Triangle3& t2, Float tol = GEOMETRY_DEFAULT_TOL)
{
    Float toln = tol * std::min(t1.mean_edge_length(), t2.mean_edge_length());
    int n = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (norm(t2.v[j] - t1.v[i]) < toln) { n++; break; }
    return n;
}

// [ORIGINAL] rwg/operators/base.hpp: OperatorBase::transform_coordinates
inline void transform_coordinates(Triangle3& obs_local, Triangle2& src_local,
                                  const Triangle3& obs, const Triangle3& src)
{
    V3 basis[3];
    src.local_coordinate_basis(basis);
    V3 o = src.v[0];   // Triangle::local_origin()
    obs_local.set(to_local(obs.v[0], o, basis), to_local(obs.v[1], o, basis), to_local(obs.v[2], o, basis), obs.pol);
    V3 s0 = to_local(src.v[0], o, basis), s1 = to_local(src.v[1], o, basis), s2 = to_local(src.v[2], o, basis);
    src_local.set({s0.x, s0.y}, {s1.x, s1.y}, {s2.x, s2.y}, src.pol);
}


// =================================================================================================
// [ORIGINAL] quadrature/triangle/gauss.hpp/.cpp + quadrature/tri_quad_xiao_gimbutas.json
// (only orders 4 and 8, the defaults of IntegrationSettings / SrcIntegrationSettings / PlaneWaveBase)
// =================================================================================================

struct RefNode { Float xi, eta, w; };

const std::vector<RefNode> XG_ORDER_4 = {
    {0.44594849091596483, 0.44594849091596483, 0.11169079483900572}, {0.44594849091596483, 0.10810301816807033, 0.11169079483900572},
    {0.10810301816807033, 0.44594849091596483, 0.11169079483900572}, {0.09157621350977066, 0.09157621350977066, 0.05497587182766093},
    {0.09157621350977066, 0.8168475729804587, 0.05497587182766093}, {0.8168475729804587, 0.09157621350977066, 0.05497587182766093}};

const std::vector<RefNode> XG_ORDER_8 = {
    {0.3333333333333333, 0.3333333333333333, 0.07215780383889359}, {0.1705693077517601, 0.1705693077517601, 0.05160868526735912},
    {0.1705693077517601, 0.6588613844964798, 0.05160868526735912}, {0.6588613844964798, 0.1705693077517601, 0.05160868526735912},
    {0.4592925882927231, 0.4592925882927231, 0.04754581713364231}, {0.4592925882927231, 0.08141482341455375, 0.04754581713364231},
    {0.08141482341455375, 0.4592925882927231, 0.04754581713364231}, {0.05054722831703086, 0.05054722831703086, 0.01622924881159904},
    {0.05054722831703086, 0.8989055433659383, 0.01622924881159904}, {0.8989055433659383, 0.05054722831703086, 0.01622924881159904},
    {0.26311282963463806, 0.008394777409957678, 0.013615157087217496}, {0.26311282963463806, 0.7284923929554042, 0.013615157087217496},
    {0.008394777409957678, 0.26311282963463806, 0.013615157087217496}, {0.008394777409957678, 0.7284923929554042, 0.013615157087217496},
    {0.7284923929554042, 0.26311282963463806, 0.013615157087217496}, {0.7284923929554042, 0.008394777409957678, 0.013615157087217496}};

inline const std::vector<RefNode>& xg_rule(int order)
{
    if (order == 4) return XG_ORDER_4;
    if (order == 8) return XG_ORDER_8;
    throw std::domain_error("xg_rule(): only orders 4 and 8 are embedded in this file.");
}

// GaussTriangleQuadrature::compute: r = xi v1 + eta v2 + (1 - xi - eta) v0, w = 2 area w_ref
inline void quad_points(const Triangle2& t, int order, std::vector<V2>& pts, std::vector<Float>& w)
{
    const auto& rule = xg_rule(order);
    pts.resize(rule.size()); w.resize(rule.size());
    for (std::size_t q = 0; q < rule.size(); ++q)
    {
        pts[q] = rule[q].xi * t.v[1] + rule[q].eta * t.v[2] + (one - rule[q].xi - rule[q].eta) * t.v[0];
        w[q] = t.area * rule[q].w * 2;
    }
}

inline void quad_points(const Triangle3& t, int order, std::vector<V3>& pts, std::vector<Float>& w)
{
    const auto& rule = xg_rule(order);
    pts.resize(rule.size()); w.resize(rule.size());
    for (std::size_t q = 0; q < rule.size(); ++q)
    {
        pts[q] = rule[q].xi * t.v[1] + rule[q].eta * t.v[2] + (one - rule[q].xi - rule[q].eta) * t.v[0];
        w[q] = t.area * rule[q].w * 2;
    }
}


// =================================================================================================
// [ORIGINAL + CHANGED] kernels/base.hpp, kernels/hgf.hpp/.cpp
//   CHANGED: compute_hess() added (base: throws; HGF: implemented)
// =================================================================================================

class ScalarKernelBase
{
public:
    virtual std::vector<Complex> compute(V3 r_obs, const std::vector<V3>& r_src, Complex k) const = 0;
    virtual std::vector<C3> compute_grad(V3 r_obs, const std::vector<V3>& r_src, Complex k) const = 0;

    // [CHANGED] Hessian, components (xx, xy, xz, yy, yz, zz)
    virtual std::vector<std::array<Complex, 6>> compute_hess(V3, const std::vector<V3>&, Complex) const
    { throw std::runtime_error("ScalarKernelBase::compute_hess(): Not implemented."); }

    virtual ~ScalarKernelBase() = default;
};

// G = exp(-jkR) / (4 pi R)
class HGF: public ScalarKernelBase
{
public:
    std::vector<Complex> compute(V3 r_obs, const std::vector<V3>& r_src, Complex k) const override
    {
        std::vector<Complex> out (r_src.size());
        for (std::size_t i = 0; i < r_src.size(); ++i)
        {
            Float r = norm(r_src[i] - r_obs);
            out[i] = std::exp(-J * k * r) / r / four_pi;
        }
        return out;
    }

    std::vector<C3> compute_grad(V3 r_obs, const std::vector<V3>& r_src, Complex k) const override
    {
        std::vector<C3> out (r_src.size());
        for (std::size_t i = 0; i < r_src.size(); ++i)
        {
            V3 d = r_obs - r_src[i];
            Float r = norm(d);
            Complex jkr = J * k * r;
            Complex s = -std::exp(-jkr) * (one + jkr) / (r * r * r) / four_pi;
            out[i] = {s * d.x, s * d.y, s * d.z};
        }
        return out;
    }

    // [CHANGED] e^{-jkR}/(4 pi) [ (3 + 3jkR - k^2R^2)/R^5 R R^T - (1 + jkR)/R^3 I ]
    std::vector<std::array<Complex, 6>> compute_hess(V3 r_obs, const std::vector<V3>& r_src, Complex k) const override
    {
        std::vector<std::array<Complex, 6>> out (r_src.size());
        for (std::size_t i = 0; i < r_src.size(); ++i)
        {
            V3 d = r_obs - r_src[i];
            Float r = norm(d), r2 = r * r;
            Complex jkr = J * k * r;
            Complex e = std::exp(-jkr) / four_pi;
            Complex c_rr = e * (3.0 + 3.0 * jkr + jkr * jkr) / (r2 * r2 * r);
            Complex c_id = -e * (one + jkr) / (r2 * r);
            out[i] = {c_rr * d.x * d.x + c_id, c_rr * d.x * d.y, c_rr * d.x * d.z,
                      c_rr * d.y * d.y + c_id, c_rr * d.y * d.z, c_rr * d.z * d.z + c_id};
        }
        return out;
    }
};

// G - (1/R - jk - k^2 R/2)/(4 pi) by its Taylor series (vectorized convergence check as in OpenBEM)
class SingularitySubtractedTaylorHGF: public ScalarKernelBase
{
public:
    std::vector<Complex> compute(V3 r_obs, const std::vector<V3>& r_src, Complex k) const override
    {
        const std::size_t n = r_src.size();
        std::vector<Complex> val (n, 0.0);
        if (std::real(k) == 0.0) return val;
        const Float tol = KERNEL_DEFAULT_TOL;
        const Complex jk = J * k;
        std::vector<Complex> jkr (n), mult (n, -jk);
        for (std::size_t i = 0; i < n; ++i) { jkr[i] = jk * norm(r_src[i] - r_obs); val[i] = mult[i]; }
        for (unsigned jj = 2; true; jj++)
        {
            bool done = true;
            for (std::size_t i = 0; i < n; ++i)
            {
                mult[i] *= -jkr[i] / (Float)jj;
                val[i] += mult[i];
                if (!(std::norm(mult[i]) <= tol * tol * std::norm(val[i]))) done = false;
            }
            if (done) break;
        }
        for (auto& v: val) v /= four_pi;
        return val;
    }

    std::vector<C3> compute_grad(V3 r_obs, const std::vector<V3>& r_src, Complex k) const override
    {
        const std::size_t n = r_src.size();
        std::vector<C3> out (n, C3{0.0, 0.0, 0.0});
        if (std::real(k) == 0.0) return out;
        const Float tol = KERNEL_DEFAULT_TOL;
        const Complex jk = J * k;
        const Complex jkc = jk * k * k;
        std::vector<Complex> jkr (n), mult (n), sc (n);
        for (std::size_t i = 0; i < n; ++i)
        {
            jkr[i] = jk * norm(r_obs - r_src[i]);
            mult[i] = jkc * (one + jkr[i]) / (Float)6.0;
            sc[i] = -jkc / 2.0 + mult[i];
        }
        for (unsigned jj = 4; true; jj++)
        {
            bool done = true;
            for (std::size_t i = 0; i < n; ++i)
            {
                mult[i] *= -jkr[i] / (Float)jj;
                sc[i] += mult[i];
                if (!(std::norm(mult[i]) <= tol * tol * std::norm(sc[i]))) done = false;
            }
            if (done) break;
        }
        for (std::size_t i = 0; i < n; ++i)
        {
            V3 d = r_obs - r_src[i];
            Complex s = sc[i] * (-1.0 / four_pi);
            out[i] = {s * d.x, s * d.y, s * d.z};
        }
        return out;
    }
};


// =================================================================================================
// [ORIGINAL + CHANGED] rwg/integrators/src/base.hpp: SrcResult, SrcIntegratorBase
//   CHANGED: rs_grad_g, rsrs_grad_g, hess_g, rs_hess_g and the shape flags
// =================================================================================================

struct SrcResult
{
    std::vector<Complex> g;
    std::vector<std::array<Complex, 2>> rs_g;
    std::vector<C3> grad_g;
    std::vector<std::array<Complex, 6>> rs_grad_g;     // [CHANGED] x' grad G (0-2), y' grad G (3-5)
    std::vector<std::array<Complex, 9>> rsrs_grad_g;   // [CHANGED] x'x', x'y', y'y' times grad G
    std::vector<std::array<Complex, 6>> hess_g;        // [CHANGED] grad grad G (xx, xy, xz, yy, yz, zz)
    std::vector<std::array<Complex, 12>> rs_hess_g;    // [CHANGED] x' grad grad G (0-5), y' grad grad G (6-11)

    void resize(std::size_t n)
    {
        g.assign(n, 0.0);
        rs_g.assign(n, {0.0, 0.0});
        grad_g.assign(n, {0.0, 0.0, 0.0});
        rs_grad_g.assign(n, {}); rsrs_grad_g.assign(n, {});
        hess_g.assign(n, {}); rs_hess_g.assign(n, {});
    }
};

class SrcIntegratorBase
{
public:
    virtual SrcResult integrate(Complex k, const Triangle2& src_tri, const std::vector<V3>& r_obs,
                                bool g_terms = true, bool grad_g_terms = true,
                                bool shape_grad_terms = false, bool shape_hess_terms = false) const = 0;  // [CHANGED]
    virtual ~SrcIntegratorBase() = default;
};


// [ORIGINAL + CHANGED] rwg/integrators/src/quadrature.cpp: SrcQuadrature
class SrcQuadrature: public SrcIntegratorBase
{
public:
    SrcQuadrature(int order, std::shared_ptr<ScalarKernelBase> kernel): order_(order), kernel_(kernel) {}

    SrcResult integrate(Complex k, const Triangle2& src_tri, const std::vector<V3>& r_obs,
                        bool g_terms = true, bool grad_g_terms = true,
                        bool shape_grad_terms = false, bool shape_hess_terms = false) const override
    {
        std::vector<V2> pts; std::vector<Float> w;
        quad_points(src_tri, order_, pts, w);
        std::vector<V3> pts3 (pts.size());
        for (std::size_t q = 0; q < pts.size(); ++q) pts3[q] = {pts[q].x, pts[q].y, 0};

        SrcResult res;
        res.resize(r_obs.size());

        for (std::size_t ro = 0; ro < r_obs.size(); ++ro)
        {
            if (g_terms)
            {
                std::vector<Complex> gk = kernel_->compute(r_obs[ro], pts3, k);
                for (std::size_t q = 0; q < pts.size(); ++q)
                {
                    Complex gw = gk[q] * w[q];
                    res.g[ro] += gw;
                    res.rs_g[ro][0] += pts[q].x * gw;
                    res.rs_g[ro][1] += pts[q].y * gw;
                }
            }
            if (grad_g_terms || shape_grad_terms)
            {
                std::vector<C3> gr = kernel_->compute_grad(r_obs[ro], pts3, k);
                for (std::size_t q = 0; q < pts.size(); ++q)
                {
                    const Float x = pts[q].x, y = pts[q].y;
                    for (int c = 0; c < 3; ++c)
                    {
                        if (grad_g_terms) res.grad_g[ro][c] += gr[q][c] * w[q];
                        if (shape_grad_terms)                                         // [CHANGED]
                        {
                            res.rs_grad_g[ro][c] += gr[q][c] * (w[q] * x);
                            res.rs_grad_g[ro][3 + c] += gr[q][c] * (w[q] * y);
                            res.rsrs_grad_g[ro][c] += gr[q][c] * (w[q] * x * x);
                            res.rsrs_grad_g[ro][3 + c] += gr[q][c] * (w[q] * x * y);
                            res.rsrs_grad_g[ro][6 + c] += gr[q][c] * (w[q] * y * y);
                        }
                    }
                }
            }
            if (shape_hess_terms)                                                    // [CHANGED]
            {
                auto hs = kernel_->compute_hess(r_obs[ro], pts3, k);
                for (std::size_t q = 0; q < pts.size(); ++q)
                    for (int h = 0; h < 6; ++h)
                    {
                        res.hess_g[ro][h] += hs[q][h] * w[q];
                        res.rs_hess_g[ro][h] += hs[q][h] * (w[q] * pts[q].x);
                        res.rs_hess_g[ro][6 + h] += hs[q][h] * (w[q] * pts[q].y);
                    }
            }
        }
        return res;
    }

private:
    int order_;
    std::shared_ptr<ScalarKernelBase> kernel_;
};


// [ORIGINAL + CHANGED] rwg/integrators/src/singularity.cpp: SrcSingularity
//   CHANGED: integrate() combines the shape moments; integrate_singular_shape() is new.
class SrcSingularity: public SrcIntegratorBase
{
public:
    explicit SrcSingularity(int order): src_quad_(order, std::make_shared<SingularitySubtractedTaylorHGF>()) {}

    SrcResult integrate(Complex k, const Triangle2& src_tri, const std::vector<V3>& r_obs,
                        bool g_terms = true, bool grad_g_terms = true,
                        bool shape_grad_terms = false, bool shape_hess_terms = false) const override
    {
        if (shape_hess_terms)
            throw std::runtime_error("SrcSingularity::integrate(): Hessian shape terms are only supported "
                                     "for observation points away from the source triangle.");

        SrcResult sing = integrate_singular(k, src_tri, r_obs, g_terms, grad_g_terms);
        SrcResult res = src_quad_.integrate(k, src_tri, r_obs, g_terms, grad_g_terms, shape_grad_terms, false);
        for (std::size_t i = 0; i < r_obs.size(); ++i)
        {
            res.g[i] += sing.g[i];
            for (int c = 0; c < 2; ++c) res.rs_g[i][c] += sing.rs_g[i][c];
            for (int c = 0; c < 3; ++c) res.grad_g[i][c] += sing.grad_g[i][c];
        }
        if (shape_grad_terms)
            integrate_singular_shape(k, src_tri, r_obs, res);   // adds the analytical singular parts
        return res;
    }

private:

    // [ORIGINAL] SrcSingularity::integrate_singular: analytical int 1/R terms (Wilton/Graglia)
    SrcResult integrate_singular(Complex k, const Triangle2& tri, const std::vector<V3>& r_obs,
                                 bool g_terms, bool grad_g_terms) const
    {
        SrcResult res;
        res.resize(r_obs.size());
        for (std::size_t ro = 0; ro < r_obs.size(); ++ro)
        {
            const V2 rho = {r_obs[ro].x, r_obs[ro].y};
            const Float z = r_obs[ro].z, z_abs = std::abs(z), z_sq = z * z;
            Float beta = 0, t0_f2 = 0;
            V2 u_f2, u_f3;
            for (int idx = 0; idx < 3; ++idx)
            {
                V2 vp = tri.v[idx], vm = tri.v[(idx + 1) % 3];      // Edge<2>(v_plus, v_minus)
                Float len = norm(vm - vp);
                V2 uv = (1.0 / len) * (vm - vp);
                V2 u_hat = {uv.y, -uv.x};
                V2 plus_to_rho = rho - vp, minus_to_rho = rho - vm;
                Float s_plus = dot(uv, plus_to_rho);
                Float s_minus = s_plus - len;
                Float t0 = -dot(u_hat, plus_to_rho);
                Float R0 = std::sqrt(t0 * t0 + z_sq);
                Float R_plus = std::sqrt(dot(plus_to_rho, plus_to_rho) + z_sq);
                Float R_minus = std::sqrt(dot(minus_to_rho, minus_to_rho) + z_sq);
                Float R0_sq = R0 * R0;
                Float atan_plus = std::atan2(t0 * s_plus, R0_sq + z_abs * R_plus);
                Float atan_minus = std::atan2(t0 * s_minus, R0_sq + z_abs * R_minus);
                Float f2 = std::log((R_plus + s_plus) / (R_minus + s_minus));
                if (!std::isfinite(f2)) f2 = 0;
                Float f3 = s_plus * R_plus - s_minus * R_minus + R0_sq * f2;
                beta += atan_plus - atan_minus;
                t0_f2 += t0 * f2;
                u_f2 = u_f2 + f2 * u_hat;
                u_f3 = u_f3 + f3 * u_hat;
            }
            if (g_terms)
            {
                res.g[ro] = (t0_f2 - z_abs * beta) / four_pi;
                res.rs_g[ro][0] = 0.5 * u_f3.x / four_pi + rho.x * res.g[ro];
                res.rs_g[ro][1] = 0.5 * u_f3.y / four_pi + rho.y * res.g[ro];
            }
            if (grad_g_terms)
            {
                Float sign = (z_abs < TRIANGLE_DEFAULT_TOL * tri.shortest_edge_length()) ? 1.0 : z / z_abs;
                C3 gg = {-u_f2.x, -u_f2.y, -sign * beta};
                Float gr[3] = {0.5 * u_f3.x, 0.5 * u_f3.y, z * z_abs * beta - z * t0_f2};
                for (int c = 0; c < 3; ++c)
                    res.grad_g[ro][c] = (gg[c] + (half * k * k) * gr[c]) / four_pi;
            }
        }
        return res;
    }

    // [CHANGED / NEW] analytical singular parts of int w grad G, w in {x', y', x'^2, x'y', y'^2},
    // for the subtracted kernel (1/R - k^2 R/2)/(4 pi), via the surface divergence theorem.
    void integrate_singular_shape(Complex k, const Triangle2& tri, const std::vector<V3>& r_obs, SrcResult& res) const
    {
        struct M2 { Float a = 0, b = 0, c = 0, d = 0; };   // 2x2 matrix [a b; c d]
        auto outer = [] (V2 p, V2 q) { M2 m; m.a = p.x * q.x; m.b = p.x * q.y; m.c = p.y * q.x; m.d = p.y * q.y; return m; };
        auto add = [] (M2& m, const M2& n, Float s) { m.a += s * n.a; m.b += s * n.b; m.c += s * n.c; m.d += s * n.d; };
        auto get = [] (const M2& m, int i, int j) { return i == 0 ? (j == 0 ? m.a : m.b) : (j == 0 ? m.c : m.d); };
        auto mv = [&] (const M2& m, V2 v) { return V2{m.a * v.x + m.b * v.y, m.c * v.x + m.d * v.y}; };

        const Complex half_k_sq = half * k * k;
        const Float tol_z = TRIANGLE_DEFAULT_TOL * tri.shortest_edge_length();

        for (std::size_t ro = 0; ro < r_obs.size(); ++ro)
        {
            const V2 rho = {r_obs[ro].x, r_obs[ro].y};
            const Float z = r_obs[ro].z, z_abs = std::abs(z);
            const Float sign = (z_abs < tol_z) ? one : z / z_abs;

            Float sum_beta = 0, sum_t0_f0 = 0, sum_t0_l0 = 0;
            V2 u_f0, u_l0, u_l3;
            M2 a_ipp, a_m2, a_pppR[2], a_ppp3[2];

            for (int idx = 0; idx < 3; ++idx)
            {
                const V2 v_start = tri.v[idx], v_end = tri.v[(idx + 1) % 3];
                const Float len = norm(v_end - v_start);
                const V2 l_hat = (1.0 / len) * (v_end - v_start);
                const V2 u_hat = {l_hat.y, -l_hat.x};          // outward edge normal

                const Float lm = dot(v_start - rho, l_hat), lp = lm + len;
                const Float t0 = dot(v_start - rho, u_hat);
                const Float r0_sq = t0 * t0 + z * z;
                const Float rm = std::sqrt(lm * lm + r0_sq), rp = std::sqrt(lp * lp + r0_sq);

                Float f0 = (lm + lp >= 0) ? std::log((rp + lp) / (rm + lm)) : std::log((rm - lm) / (rp - lp));
                if (!std::isfinite(f0)) f0 = 0;
                Float beta = 0;
                if (r0_sq > float_eps * len * len)
                    beta = std::atan2(t0 * lp, r0_sq + z_abs * rp) - std::atan2(t0 * lm, r0_sq + z_abs * rm);

                const Float f1 = rp - rm;                                             // int l/R
                const Float f2 = half * (lp * rp - lm * rm) - half * r0_sq * f0;      // int l^2/R
                const Float l0 = half * (lp * rp - lm * rm) + half * r0_sq * f0;      // int R
                const Float l1 = (rp * rp * rp - rm * rm * rm) / 3;                   // int l R
                const Float l2 = (lp * (2 * lp * lp + r0_sq) * rp - lm * (2 * lm * lm + r0_sq) * rm) / 8
                                 - r0_sq * r0_sq * f0 / 8;                            // int l^2 R
                const Float l3 = (lp * (2 * lp * lp + 5 * r0_sq) * rp - lm * (2 * lm * lm + 5 * r0_sq) * rm) / 8
                                 + 3 * r0_sq * r0_sq * f0 / 8;                        // int R^3

                sum_beta += beta; sum_t0_f0 += t0 * f0; sum_t0_l0 += t0 * l0;
                u_f0 = u_f0 + f0 * u_hat; u_l0 = u_l0 + l0 * u_hat; u_l3 = u_l3 + l3 * u_hat;
                add(a_ipp, outer(t0 * l0 * u_hat + l1 * l_hat, u_hat), 1);
                add(a_m2, outer(t0 * f0 * u_hat + f1 * l_hat, u_hat), 1);

                M2 uu = outer(u_hat, u_hat), ll = outer(l_hat, l_hat), ul = outer(u_hat, l_hat);
                add(ul, outer(l_hat, u_hat), 1);
                M2 ppR, pp3;
                add(ppR, uu, t0 * t0 * l0); add(ppR, ul, t0 * l1); add(ppR, ll, l2);
                add(pp3, uu, t0 * t0 * f0); add(pp3, ul, t0 * f1); add(pp3, ll, f2);
                add(a_pppR[0], ppR, u_hat.x); add(a_pppR[1], ppR, u_hat.y);
                add(a_ppp3[0], pp3, u_hat.x); add(a_ppp3[1], pp3, u_hat.y);
            }

            const Float i0 = sum_t0_f0 - z_abs * sum_beta;             // int 1/R
            const Float iR = (sum_t0_l0 + z * z * i0) / 3;             // int R
            const V2 ip = u_l0, ipR = (1.0 / 3) * u_l3, m1 = (-1.0) * u_f0;
            M2 ipp = a_ipp; ipp.a -= iR; ipp.d -= iR;                  // int p p^T/R
            M2 m2; add(m2, a_m2, -1); m2.a += i0; m2.d += i0;          // int p p^T/R^3
            const Float z_m0 = sign * sum_beta;                        // z int 1/R^3
            Float ippp[2][2][2], mppp[2][2][2];                        // [c][a][b]
            for (int c = 0; c < 2; ++c)
                for (int a = 0; a < 2; ++a)
                    for (int b = 0; b < 2; ++b)
                    {
                        Float ipRa = a == 0 ? ipR.x : ipR.y, ipRb = b == 0 ? ipR.x : ipR.y;
                        Float ipa = a == 0 ? ip.x : ip.y, ipb = b == 0 ? ip.x : ip.y;
                        ippp[c][a][b] = get(a_pppR[c], a, b) - ((a == c ? ipRb : 0) + (b == c ? ipRa : 0));
                        mppp[c][a][b] = -get(a_ppp3[c], a, b) + ((a == c ? ipb : 0) + (b == c ? ipa : 0));
                    }

            // w = w0 + w1 . p + p^T W2 p
            auto weighted = [&] (Float w0, V2 w1, const M2& w2) -> C3
            {
                Float t_m[2], t_i[2];
                for (int c = 0; c < 2; ++c)
                {
                    t_m[c] = 0; t_i[c] = 0;
                    for (int a = 0; a < 2; ++a)
                        for (int b = 0; b < 2; ++b)
                        {
                            t_m[c] += get(w2, a, b) * mppp[c][a][b];
                            t_i[c] += get(w2, a, b) * ippp[c][a][b];
                        }
                }
                Float w2_m2 = 0, w2_ipp = 0;
                for (int a = 0; a < 2; ++a)
                    for (int b = 0; b < 2; ++b) { w2_m2 += get(w2, a, b) * get(m2, a, b); w2_ipp += get(w2, a, b) * get(ipp, a, b); }

                V2 s_inv_par = w0 * m1 + mv(m2, w1) + V2{t_m[0], t_m[1]};
                Float s_inv_z = -(w0 * z_m0 + z * (dot(w1, m1) + w2_m2));
                V2 s_r_par = (-1.0) * (w0 * ip + mv(ipp, w1) + V2{t_i[0], t_i[1]});
                Float s_r_z = z * (w0 * i0 + dot(w1, ip) + w2_ipp);
                return {(s_inv_par.x - half_k_sq * s_r_par.x) / four_pi,
                        (s_inv_par.y - half_k_sq * s_r_par.y) / four_pi,
                        (s_inv_z - half_k_sq * s_r_z) / four_pi};
            };

            const V2 ex = {1, 0}, ey = {0, 1};
            M2 zero2, w_xx, w_xy, w_yy;
            w_xx.a = 1; w_xy.b = half; w_xy.c = half; w_yy.d = 1;
            const C3 sx = weighted(rho.x, ex, zero2), sy = weighted(rho.y, ey, zero2);
            const C3 sxx = weighted(rho.x * rho.x, 2 * rho.x * ex, w_xx);
            const C3 sxy = weighted(rho.x * rho.y, rho.y * ex + rho.x * ey, w_xy);
            const C3 syy = weighted(rho.y * rho.y, 2 * rho.y * ey, w_yy);
            for (int c = 0; c < 3; ++c)
            {
                res.rs_grad_g[ro][c] += sx[c]; res.rs_grad_g[ro][3 + c] += sy[c];
                res.rsrs_grad_g[ro][c] += sxx[c]; res.rsrs_grad_g[ro][3 + c] += sxy[c]; res.rsrs_grad_g[ro][6 + c] += syy[c];
            }
        }
    }

    SrcQuadrature src_quad_;
};


// [ORIGINAL + CHANGED] rwg/integrators/src/strategic.cpp: SrcStrategic (default settings)
//   CHANGED: passes the shape flags through
class SrcStrategic: public SrcIntegratorBase
{
public:
    SrcStrategic(): src_hgf_(4, std::make_shared<HGF>()), src_sthgf_(8) {}

    SrcResult integrate(Complex k, const Triangle2& src_tri, const std::vector<V3>& r_obs,
                        bool g_terms = true, bool grad_g_terms = true,
                        bool shape_grad_terms = false, bool shape_hess_terms = false) const override
    {
        V3 c = {src_tri.centroid.x, src_tri.centroid.y, 0};
        Float min_dist = std::numeric_limits<Float>::max();
        for (const V3& r: r_obs) min_dist = std::min(min_dist, norm(r - c));
        Float longest_edge_wvl = src_tri.longest_edge_length() * std::real(k) / two_pi;
        if (longest_edge_wvl >= 1)
            throw std::runtime_error("SrcStrategic: line integration (edges >= 1 wavelength) is not ported.");
        Float threshold_dist = src_tri.longest_edge_length() * 5;
        if (min_dist <= threshold_dist)
            return src_sthgf_.integrate(k, src_tri, r_obs, g_terms, grad_g_terms, shape_grad_terms, shape_hess_terms);
        return src_hgf_.integrate(k, src_tri, r_obs, g_terms, grad_g_terms, shape_grad_terms, shape_hess_terms);
    }

private:
    SrcQuadrature src_hgf_;
    SrcSingularity src_sthgf_;
};


// =================================================================================================
// [ORIGINAL + CHANGED] rwg/integrators/obs/base.hpp, quadrature.cpp, strategic.cpp
//   CHANGED: ObsResult::grad_g_shape and the `shape_terms` flag
// =================================================================================================

struct ObsResult
{
    Complex g = 0;
    std::array<Complex, 12> rs_g {};
    // [CHANGED] grad_g_shape[c][6 a + b] = int_obs m_a int_src n_b d_c G, with
    // m = (1, x, y, z, x^2, xy, xz, y^2, yz, z^2) and n = (1, x', y', x'^2, x'y', y'^2)
    bool has_shape = false;
    std::array<std::array<Complex, 60>, 3> grad_g_shape {};
};

class ObsQuadrature
{
public:
    ObsQuadrature(int order, std::shared_ptr<SrcIntegratorBase> src): order_(order), src_(src) {}

    ObsResult integrate(Complex k, const Triangle3& obs_tri, const Triangle2& src_tri,
                        bool g_term, bool rs_g_terms, bool shape_terms) const
    {
        std::vector<V3> pts; std::vector<Float> w;
        quad_points(obs_tri, order_, pts, w);
        SrcResult s = src_->integrate(k, src_tri, pts, g_term || rs_g_terms, shape_terms, shape_terms, false);

        ObsResult r;
        for (std::size_t q = 0; q < pts.size(); ++q)
        {
            const Complex wg = s.g[q] * w[q];
            const Complex wx = s.rs_g[q][0] * w[q], wy = s.rs_g[q][1] * w[q];
            if (g_term) r.g += wg;
            if (rs_g_terms)   // ObsQuadrature::compute_rs_g_terms
            {
                r.rs_g[0] += wg; r.rs_g[1] += wx; r.rs_g[2] += wy;
                r.rs_g[3] += wg * pts[q].x; r.rs_g[4] += wg * pts[q].y; r.rs_g[5] += wg * pts[q].z;
                r.rs_g[6] += wx * pts[q].x; r.rs_g[7] += wx * pts[q].y; r.rs_g[8] += wx * pts[q].z;
                r.rs_g[9] += wy * pts[q].x; r.rs_g[10] += wy * pts[q].y; r.rs_g[11] += wy * pts[q].z;
            }
            if (shape_terms)  // [CHANGED] ObsQuadrature::compute_grad_g_shape_terms
            {
                const Float x = pts[q].x, y = pts[q].y, z = pts[q].z;
                const Float mono[10] = {1, x, y, z, x * x, x * y, x * z, y * y, y * z, z * z};
                for (int c = 0; c < 3; ++c)
                {
                    const Complex src_m[6] = {s.grad_g[q][c], s.rs_grad_g[q][c], s.rs_grad_g[q][3 + c],
                                              s.rsrs_grad_g[q][c], s.rsrs_grad_g[q][3 + c], s.rsrs_grad_g[q][6 + c]};
                    for (int a = 0; a < 10; ++a)
                        for (int b = 0; b < 6; ++b)
                            r.grad_g_shape[c][6 * a + b] += (w[q] * mono[a]) * src_m[b];
                }
            }
        }
        r.has_shape = shape_terms;
        return r;
    }

private:
    int order_;
    std::shared_ptr<SrcIntegratorBase> src_;
};

// Default IntegrationSettings: obs far 4, near 8; src far 4, near 8; threshold 2 x longest edge
class ObsStrategic
{
public:
    ObsStrategic():
        hgf_(4, std::make_shared<SrcQuadrature>(4, std::make_shared<HGF>())),
        sthgf_(8, std::make_shared<SrcSingularity>(8)) {}

    ObsResult integrate(Complex k, const Triangle3& obs_tri, const Triangle2& src_tri,
                        bool g_term, bool rs_g_terms, bool shape_terms = false) const
    {
        const Triangle3 src3 = src_tri.to_3d();
        Float dist = norm(obs_tri.centroid - src3.centroid);
        Float longest_edge_wvl = src_tri.longest_edge_length() * std::real(k) / two_pi;
        if (longest_edge_wvl >= 1)
            throw std::runtime_error("ObsStrategic: line integration (edges >= 1 wavelength) is not ported.");
        Float threshold_dist = src_tri.longest_edge_length() * 2;
        bool singularity_subtraction = common_vertices(obs_tri, src3) > 0 || dist <= threshold_dist;
        return singularity_subtraction ? sthgf_.integrate(k, obs_tri, src_tri, g_term, rs_g_terms, shape_terms)
                                       : hgf_.integrate(k, obs_tri, src_tri, g_term, rs_g_terms, shape_terms);
    }

private:
    ObsQuadrature hgf_, sthgf_;
};


// =================================================================================================
// [ORIGINAL] rwg/function_space: Rwg::normalization = polarity / area / 2, free vertex (i + 2) % 3
// =================================================================================================

inline Float rwg_norm(const Triangle3& t, int i) { return t.pol[i] / t.area / 2; }


// =================================================================================================
// [ORIGINAL] rwg/operators/single_layer.cpp: VectorSingleLayerOp, ScalarSingleLayerOp,
//            VectorHypersingularOp  (T = int int f.f G - (1/k^2) int int div f div f G)
// =================================================================================================

class VectorHypersingularOp
{
public:
    CMat3 compute(Complex k, const Triangle3& obs_tri, const Triangle3& src_tri) const
    {
        Triangle3 obs_l; Triangle2 src_l;
        transform_coordinates(obs_l, src_l, obs_tri, src_tri);
        ObsResult r = integ_.integrate(k, obs_l, src_l, true, true, false);
        return assemble(k, obs_l, src_l.to_3d(), r);
    }

    CMat3 assemble(Complex k, const Triangle3& obs, const Triangle3& src, const ObsResult& r) const
    {
        const auto& I = r.rs_g;
        CMat3 res;
        const Complex hessg = r.g * (one / obs.area) * (one / src.area) / k / k;   // ScalarSingleLayerOp / k^2
        for (int jj = 0; jj < 3; ++jj)
        {
            Float xn = src.v[(jj + 2) % 3].x, yn = src.v[(jj + 2) % 3].y;
            for (int ii = 0; ii < 3; ++ii)
            {
                Float xm = obs.v[(ii + 2) % 3].x, ym = obs.v[(ii + 2) % 3].y;
                Complex vsl = (xm * xn + ym * yn) * I[0] - xm * I[1] - ym * I[2] - xn * I[3] - yn * I[4] + I[6] + I[10];
                vsl *= rwg_norm(obs, ii) * rwg_norm(src, jj);
                res[ii][jj] = vsl - hessg * obs.pol[ii] * src.pol[jj];
            }
        }
        return res;
    }

private:
    ObsStrategic integ_;
};


// =================================================================================================
// [NEW] rwg/operators/base.hpp ShapeOperatorBase + single_layer.cpp VectorHypersingularShapeOp
//   Paper eqs. (24) and (26), summed into vertex-coordinate gradients. Output: 9 x 18,
//   row 3 ii + jj, column 3 vertex + component (vertices 0-2 obs, 3-5 src), global frame.
// =================================================================================================

using Shape9x18 = std::array<std::array<Complex, 18>, 9>;

class VectorHypersingularShapeOp
{
public:
    Shape9x18 compute(Complex k, const Triangle3& obs_tri, const Triangle3& src_tri) const
    {
        Triangle3 obs_l; Triangle2 src_l;
        transform_coordinates(obs_l, src_l, obs_tri, src_tri);
        ObsResult r = integ_.integrate(k, obs_l, src_l, false, true, true);
        Shape9x18 res = assemble(k, obs_l, src_l.to_3d(), r);

        // vertex gradients back to the global frame: g_global = U g_local
        V3 U[3];
        src_tri.local_coordinate_basis(U);
        for (int row = 0; row < 9; ++row)
            for (int vert = 0; vert < 6; ++vert)
            {
                Complex gl[3] = {res[row][3 * vert], res[row][3 * vert + 1], res[row][3 * vert + 2]};
                for (int c = 0; c < 3; ++c)
                    res[row][3 * vert + c] = comp(U[0], c) * gl[0] + comp(U[1], c) * gl[1] + comp(U[2], c) * gl[2];
            }
        return res;
    }

    Shape9x18 assemble(Complex k, const Triangle3& obs, const Triangle3& src, const ObsResult& r) const
    {
        Shape9x18 res {};
        for (auto& row: res) row.fill(0.0);
        if (!r.has_shape) return res;

        // Moments of G as 4 x 3 (obs 1,x,y,z by src 1,x',y') from rs_g
        const auto& I = r.rs_g;
        const Complex gm[4][3] = {{I[0], I[1], I[2]}, {I[3], I[6], I[9]}, {I[4], I[7], I[10]}, {I[5], I[8], I[11]}};
        auto int_g = [&] (const Float* a, const Float* b) {
            Complex s = 0;
            for (int i = 0; i < 4; ++i) for (int j = 0; j < 3; ++j) s += a[i] * b[j] * gm[i][j];
            return s;
        };
        // int int a(r) b(r') d_c G from grad_g_shape
        auto hm_vec = [&] (int c, const Float* b, Complex out[10]) {
            for (int i = 0; i < 10; ++i)
            {
                out[i] = 0;
                for (int j = 0; j < 6; ++j) out[i] += r.grad_g_shape[c][6 * i + j] * b[j];
            }
        };
        auto dot10 = [] (const Float* a, const Complex* h) { Complex s = 0; for (int i = 0; i < 10; ++i) s += a[i] * h[i]; return s; };
        auto lin_obs = [] (Float c0, V3 g, Float* p) { std::fill(p, p + 10, 0.0); p[0] = c0; p[1] = g.x; p[2] = g.y; p[3] = g.z; };
        auto lin_src = [] (Float c0, Float gx, Float gy, Float* p) { std::fill(p, p + 6, 0.0); p[0] = c0; p[1] = gx; p[2] = gy; };
        auto mul_obs = [] (const Float* a, const Float* b, Float* p) {
            static const int quad[3][3] = {{4, 5, 6}, {5, 7, 8}, {6, 8, 9}};
            std::fill(p, p + 10, 0.0); p[0] = a[0] * b[0];
            for (int i = 0; i < 3; ++i) { p[1 + i] = a[0] * b[1 + i] + b[0] * a[1 + i]; for (int j = 0; j < 3; ++j) p[quad[i][j]] += a[1 + i] * b[1 + j]; }
        };
        auto mul_src = [] (const Float* a, const Float* b, Float* p) {
            static const int quad[2][2] = {{3, 4}, {4, 5}};
            std::fill(p, p + 6, 0.0); p[0] = a[0] * b[0];
            for (int i = 0; i < 2; ++i) { p[1 + i] = a[0] * b[1 + i] + b[0] * a[1 + i]; for (int j = 0; j < 2; ++j) p[quad[i][j]] += a[1 + i] * b[1 + j]; }
        };
        // barycentric coordinate of vertex a: lambda = offset + grad . r
        auto barycentric = [] (const Triangle3& t, int a, V3& grad, Float& off) {
            V3 c = cross(t.v[1] - t.v[0], t.v[2] - t.v[0]);
            Float ta = norm(c);
            grad = (1.0 / ta) * cross((1.0 / ta) * c, t.v[(a + 2) % 3] - t.v[(a + 1) % 3]);
            off = one - dot(grad, t.v[a]);
        };

        const Complex inv_k_sq = one / (k * k);
        Float s[3], t[3];
        for (int i = 0; i < 3; ++i) { s[i] = rwg_norm(obs, i); t[i] = rwg_norm(src, i); }

        // f_n(r') = t (r' - p_n), in-plane components
        Float f_p[3][2][6];
        for (int jj = 0; jj < 3; ++jj)
        {
            V3 pn = src.v[(jj + 2) % 3];
            lin_src(-t[jj] * pn.x, t[jj], 0, f_p[jj][0]);
            lin_src(-t[jj] * pn.y, 0, t[jj], f_p[jj][1]);
        }
        Complex h_fp[3][3][2][10];
        for (int c = 0; c < 3; ++c) for (int jj = 0; jj < 3; ++jj) for (int d = 0; d < 2; ++d) hm_vec(c, f_p[jj][d], h_fp[c][jj][d]);
        Complex h_one[3][10];
        const Float one_src[6] = {1, 0, 0, 0, 0, 0};
        for (int c = 0; c < 3; ++c) hm_vec(c, one_src, h_one[c]);

        for (int vert = 0; vert < 6; ++vert)
        {
            const bool obs_vert = vert < 3;
            V3 gl; Float off;
            barycentric(obs_vert ? obs : src, obs_vert ? vert : vert - 3, gl, off);

            if (obs_vert)
            {
                Float lam_q[10]; lin_obs(off, gl, lam_q);
                for (int ii = 0; ii < 3; ++ii)
                {
                    V3 qm = obs.v[(ii + 2) % 3];
                    Float f_q[3][10], f_q_lam[2][10], gq_f[10];
                    for (int d = 0; d < 3; ++d) lin_obs(-s[ii] * comp(qm, d), s[ii] * unit(d), f_q[d]);
                    for (int d = 0; d < 2; ++d) mul_obs(f_q[d], lam_q, f_q_lam[d]);
                    lin_obs(-s[ii] * dot(gl, qm), s[ii] * gl, gq_f);
                    for (int jj = 0; jj < 3; ++jj)
                        for (int c = 0; c < 3; ++c)
                        {
                            // eq. (26): (f_m . grad lambda(r)) f_n,c G + (f_m . f_n) d_c G lambda(r)
                            Complex d_i1 = (c < 2) ? int_g(gq_f, f_p[jj][c]) : Complex(0);
                            for (int d = 0; d < 2; ++d) d_i1 += dot10(f_q_lam[d], h_fp[c][jj][d]);
                            // eq. (24): div f_m div f_n d_c G lambda(r), div f = 2 s
                            Complex d_i2 = (2 * s[ii]) * (2 * t[jj]) * dot10(lam_q, h_one[c]);
                            res[3 * ii + jj][3 * vert + c] = d_i1 - d_i2 * inv_k_sq;
                        }
                }
            }
            else
            {
                Float lam_p[6]; lin_src(off, gl.x, gl.y, lam_p);
                Complex h_lam[3][10], h_fp_lam[3][3][2][10];
                for (int c = 0; c < 3; ++c)
                {
                    hm_vec(c, lam_p, h_lam[c]);
                    for (int jj = 0; jj < 3; ++jj)
                        for (int d = 0; d < 2; ++d) { Float fl[6]; mul_src(f_p[jj][d], lam_p, fl); hm_vec(c, fl, h_fp_lam[c][jj][d]); }
                }
                for (int ii = 0; ii < 3; ++ii)
                {
                    V3 qm = obs.v[(ii + 2) % 3];
                    Float f_q[3][10];
                    for (int d = 0; d < 3; ++d) lin_obs(-s[ii] * comp(qm, d), s[ii] * unit(d), f_q[d]);
                    for (int jj = 0; jj < 3; ++jj)
                    {
                        V3 pn = src.v[(jj + 2) % 3];
                        Float gp_f[6]; lin_src(-t[jj] * dot(gl, pn), t[jj] * gl.x, t[jj] * gl.y, gp_f);
                        for (int c = 0; c < 3; ++c)
                        {
                            // eq. (26): f_m,c (grad lambda(r') . f_n) G - (f_m . f_n) d_c G lambda(r')
                            Complex d_i1 = int_g(f_q[c], gp_f);
                            for (int d = 0; d < 2; ++d) d_i1 -= dot10(f_q[d], h_fp_lam[c][jj][d]);
                            // eq. (24): -div f_m div f_n d_c G lambda(r')
                            Complex d_i2 = -(2 * s[ii]) * (2 * t[jj]) * h_lam[c][0];
                            res[3 * ii + jj][3 * vert + c] = d_i1 - d_i2 * inv_k_sq;
                        }
                    }
                }
            }
        }
        return res;
    }

private:
    ObsStrategic integ_;
};


// =================================================================================================
// [ORIGINAL + CHANGED] rwg/excitations/plane_wave: PlaneWaveBase::eval, RwgPlaneWave::compute
//   CHANGED: RwgPlaneWave::compute_shape (3 edges x 9: 3 vertex + component)
// =================================================================================================

class RwgPlaneWave
{
public:
    RwgPlaneWave(V3 dir, V3 pol, V3 pos, Complex amp):
        dir_((1.0 / norm(dir)) * dir), pol_((1.0 / norm(pol)) * pol), pos_(pos), amp_(amp) {}

    C3 eval(Complex k, V3 r) const
    {
        Complex e = amp_ * std::exp(-J * k * dot(dir_, r - pos_));
        return {e * pol_.x, e * pol_.y, e * pol_.z};
    }

    // Rwg::test_field with GaussTriangleQuadrature<3> of order 4
    C3 compute(Complex k, const Triangle3& tri) const
    {
        std::vector<V3> pts; std::vector<Float> w;
        quad_points(tri, 4, pts, w);
        C3 res {0.0, 0.0, 0.0};
        for (std::size_t q = 0; q < pts.size(); ++q)
        {
            C3 E = eval(k, pts[q]);
            for (int e = 0; e < 3; ++e)
            {
                V3 f = rwg_norm(tri, e) * (pts[q] - tri.v[(e + 2) % 3]);
                res[e] += w[q] * (f.x * E[0] + f.y * E[1] + f.z * E[2]);
            }
        }
        return res;
    }

    // [CHANGED] db_n/dr_m = int [ E (grad lambda_m . f_n) - jk d (f_n . E) lambda_m ]
    std::array<std::array<Complex, 9>, 3> compute_shape(Complex k, const Triangle3& tri) const
    {
        std::vector<V3> pts; std::vector<Float> w;
        quad_points(tri, 4, pts, w);
        V3 gl[3];
        for (int a = 0; a < 3; ++a)
            gl[a] = (1.0 / (2 * tri.area)) * cross(tri.normal, tri.v[(a + 2) % 3] - tri.v[(a + 1) % 3]);
        std::array<std::array<Complex, 9>, 3> res {};
        for (std::size_t q = 0; q < pts.size(); ++q)
        {
            C3 E = eval(k, pts[q]);
            for (int e = 0; e < 3; ++e)
            {
                V3 f = rwg_norm(tri, e) * (pts[q] - tri.v[(e + 2) % 3]);
                Complex fE = f.x * E[0] + f.y * E[1] + f.z * E[2];
                for (int a = 0; a < 3; ++a)
                {
                    Float lam = one + dot(gl[a], pts[q] - tri.v[a]);
                    Float gf = dot(gl[a], f);
                    for (int c = 0; c < 3; ++c)
                        res[e][3 * a + c] += w[q] * (E[c] * gf - J * k * comp(dir_, c) * fE * lam);
                }
            }
        }
        return res;
    }

private:
    V3 dir_, pol_, pos_;
    Complex amp_;
};


// =================================================================================================
// [ORIGINAL + CHANGED] rwg/projectors/single_layer: VectorHypersingularProj::compute
//   (= VectorSingleLayerProj + GradScalarSingleLayerProj / k^2 x polarity), with SrcStrategic
//   CHANGED: VectorHypersingularProj::compute_shape
// =================================================================================================

class VectorHypersingularProj
{
public:
    // rows 3 pt + comp, columns: source edge
    CMat compute(Complex k, const std::vector<V3>& pts, const Triangle3& src) const
    {
        V3 U[3]; src.local_coordinate_basis(U);
        Triangle3 dummy; Triangle2 src_l;
        transform_coordinates(dummy, src_l, src, src);
        std::vector<V3> pl (pts.size());
        for (std::size_t i = 0; i < pts.size(); ++i) pl[i] = to_local(pts[i], src.v[0], U);
        SrcResult s = integ_.integrate(k, src_l, pl, true, true);
        CMat res (3 * pts.size(), 3);
        for (std::size_t p = 0; p < pts.size(); ++p)
            for (int jj = 0; jj < 3; ++jj)
            {
                V2 vf = src_l.v[(jj + 2) % 3];
                Complex rx = s.rs_g[p][0] - vf.x * s.g[p], ry = s.rs_g[p][1] - vf.y * s.g[p];
                for (int c = 0; c < 3; ++c)
                {
                    Complex vsl = (comp(U[0], c) * rx + comp(U[1], c) * ry) * rwg_norm(src, jj);
                    Complex gsl = (comp(U[0], c) * s.grad_g[p][0] + comp(U[1], c) * s.grad_g[p][1]
                                   + comp(U[2], c) * s.grad_g[p][2]) * (one / src.area);
                    res(3 * p + c, jj) = vsl + gsl / k / k * src.pol[jj];
                }
            }
        return res;
    }

    // [CHANGED] rows 3 pt + comp, columns 9 jj + 3 vertex + c
    CMat compute_shape(Complex k, const std::vector<V3>& pts, const Triangle3& src) const
    {
        V3 U[3]; src.local_coordinate_basis(U);
        Triangle3 dummy; Triangle2 sl;
        transform_coordinates(dummy, sl, src, src);
        std::vector<V3> pl (pts.size());
        for (std::size_t i = 0; i < pts.size(); ++i) pl[i] = to_local(pts[i], src.v[0], U);
        SrcResult s = integ_.integrate(k, sl, pl, true, true, true, true);

        static const int hidx[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
        const Complex inv_k_sq = one / (k * k);
        Float gx[3], gy[3], g0[3];
        for (int a = 0; a < 3; ++a)
        {
            V2 e = sl.v[(a + 2) % 3] - sl.v[(a + 1) % 3];
            gx[a] = -e.y / (2 * sl.area); gy[a] = e.x / (2 * sl.area);
            g0[a] = one - (gx[a] * sl.v[a].x + gy[a] * sl.v[a].y);
        }
        CMat res (3 * pts.size(), 27);
        for (std::size_t p = 0; p < pts.size(); ++p)
        {
            Complex dg[6][3];
            for (int c = 0; c < 3; ++c)
            {
                dg[0][c] = s.grad_g[p][c]; dg[1][c] = s.rs_grad_g[p][c]; dg[2][c] = s.rs_grad_g[p][3 + c];
                dg[3][c] = s.rsrs_grad_g[p][c]; dg[4][c] = s.rsrs_grad_g[p][3 + c]; dg[5][c] = s.rsrs_grad_g[p][6 + c];
            }
            auto hess = [&] (int wi, int a, int b) {
                int h = hidx[a][b];
                return wi == 0 ? s.hess_g[p][h] : s.rs_hess_g[p][6 * (wi - 1) + h];
            };
            for (int jj = 0; jj < 3; ++jj)
            {
                V2 pn = sl.v[(jj + 2) % 3];
                Float t = rwg_norm(src, jj);
                for (int vert = 0; vert < 3; ++vert)
                {
                    Float ax = gx[vert], ay = gy[vert], a0 = g0[vert];
                    Complex gl_f_g = t * (ax * s.rs_g[p][0] + ay * s.rs_g[p][1] - (ax * pn.x + ay * pn.y) * s.g[p]);
                    Complex D[3][3] = {};
                    for (int c = 0; c < 3; ++c)
                    {
                        D[c][c] += gl_f_g;
                        for (int m = 0; m < 2; ++m)
                        {
                            Float pm = m == 0 ? pn.x : pn.y;
                            Float wv[6] = {-pm * a0, -pm * ax, -pm * ay, 0, 0, 0};
                            if (m == 0) { wv[1] += a0; wv[3] += ax; wv[4] += ay; }
                            else        { wv[2] += a0; wv[4] += ax; wv[5] += ay; }
                            Complex val = 0;
                            for (int q = 0; q < 6; ++q) val += wv[q] * dg[q][c];
                            D[m][c] -= t * val;
                        }
                        for (int m = 0; m < 3; ++m)
                            D[m][c] -= 2 * t * inv_k_sq * (a0 * hess(0, m, c) + ax * hess(1, m, c) + ay * hess(2, m, c));
                    }
                    // rotate field and vertex components: U D U^T
                    for (int mg = 0; mg < 3; ++mg)
                        for (int cg = 0; cg < 3; ++cg)
                        {
                            Complex v = 0;
                            for (int m = 0; m < 3; ++m) for (int c = 0; c < 3; ++c)
                                v += comp(U[m], mg) * D[m][c] * comp(U[c], cg);
                            res(3 * p + mg, 9 * jj + 3 * vert + cg) = v;
                        }
                }
            }
        }
        return res;
    }

private:
    SrcStrategic integ_;
};


// =================================================================================================
// [ORIGINAL] geometry/mesh: MeshTransfer::read_gmsh_v2 (triangles, physical tag) and
//            TriangleMesh::generate_edges (local edge i = vertices (i, i+1), polarity +1 first)
// =================================================================================================

struct Mesh
{
    std::vector<V3> verts;
    std::vector<std::array<Index, 3>> elems;
    std::vector<Index> tags;
    std::vector<std::array<Index, 3>> elem_edges;
    std::vector<std::array<Float, 3>> elem_pol;
    Index num_edges = 0;

    Triangle3 elem_primitive(Index e) const
    {
        return Triangle3(verts[elems[e][0]], verts[elems[e][1]], verts[elems[e][2]], elem_pol[e].data());
    }

    void generate_edges()
    {
        elem_edges.assign(elems.size(), {0, 0, 0});
        elem_pol.assign(elems.size(), {1, 1, 1});
        num_edges = 0;
        std::set<Index> unique_tags (tags.begin(), tags.end());
        for (Index tag: unique_tags)
        {
            std::map<std::pair<Index, Index>, Index> unique_edges;
            for (Index e = 0; e < elems.size(); ++e)
            {
                if (tags[e] != tag) continue;
                for (int i = 0; i < 3; ++i)
                {
                    std::pair<Index, Index> v = {elems[e][i], elems[e][(i + 1) % 3]};
                    if (v.first > v.second) std::swap(v.first, v.second);
                    auto ins = unique_edges.insert({v, num_edges});
                    if (ins.second) { elem_edges[e][i] = num_edges; elem_pol[e][i] = 1; num_edges++; }
                    else { elem_edges[e][i] = ins.first->second; elem_pol[e][i] = -1; }
                }
            }
        }
    }
};

inline bool read_gmsh_v2(const std::string& file, Mesh& mesh)
{
    std::ifstream in (file);
    if (!in) return false;
    std::string line;
    std::map<long, Index> node;
    while (std::getline(in, line))
    {
        if (line.rfind("$MeshFormat", 0) == 0)
        {
            std::getline(in, line);
            std::istringstream ss (line); double ver; int ft; ss >> ver >> ft;
            if (ft != 0 || ver >= 3) throw std::runtime_error("Only ASCII Gmsh v2 meshes are supported.");
        }
        else if (line.rfind("$Nodes", 0) == 0)
        {
            std::getline(in, line); long n = std::stol(line);
            for (long i = 0; i < n; ++i)
            {
                std::getline(in, line); std::istringstream ss (line);
                long id; V3 p; ss >> id >> p.x >> p.y >> p.z;
                node[id] = mesh.verts.size(); mesh.verts.push_back(p);
            }
        }
        else if (line.rfind("$Elements", 0) == 0)
        {
            std::getline(in, line); long n = std::stol(line);
            for (long i = 0; i < n; ++i)
            {
                std::getline(in, line); std::istringstream ss (line);
                long id, type, ntags; ss >> id >> type >> ntags;
                std::vector<long> tg (ntags); for (auto& t: tg) ss >> t;
                if (type == 2)
                {
                    long a, b, c; ss >> a >> b >> c;
                    mesh.elems.push_back({node.at(a), node.at(b), node.at(c)});
                    mesh.tags.push_back(ntags > 0 ? (Index)(tg[0] - 1) : 0);
                }
            }
        }
    }
    mesh.generate_edges();
    return !mesh.elems.empty();
}


// =================================================================================================
// [ORIGINAL + CHANGED] rwg/assemblers: OperatorAssembler, ExcitationAssembler, ProjectorAssembler
//   CHANGED: assemble_shape_derivative() and shape_sensitivity()
// =================================================================================================

inline CMat assemble_T(const Mesh& m, Complex k, const VectorHypersingularOp& op)
{
    CMat Z (m.num_edges, m.num_edges);
    const long ne = m.elems.size();
#pragma omp parallel for schedule(dynamic, 8)
    for (long q = 0; q < ne; ++q)
    {
        Triangle3 tq = m.elem_primitive(q);
        for (long p = 0; p < ne; ++p)
        {
            CMat3 v = op.compute(k, tq, m.elem_primitive(p));
            for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j)
                atomic_add(Z(m.elem_edges[q][i], m.elem_edges[p][j]), v[i][j]);
        }
    }
    return Z;
}

// [CHANGED] dT/ds for the vertex velocity field V (one 3-vector per vertex)
inline CMat assemble_T_shape_derivative(const Mesh& m, Complex k, const VectorHypersingularShapeOp& op,
                                        const std::vector<V3>& V)
{
    CMat D (m.num_edges, m.num_edges);
    const long ne = m.elems.size();
#pragma omp parallel for schedule(dynamic, 8)
    for (long q = 0; q < ne; ++q)
    {
        Triangle3 tq = m.elem_primitive(q);
        for (long p = 0; p < ne; ++p)
        {
            Shape9x18 d = op.compute(k, tq, m.elem_primitive(p));
            Float vel[18];
            for (int a = 0; a < 3; ++a)
                for (int c = 0; c < 3; ++c)
                {
                    vel[3 * a + c] = comp(V[m.elems[q][a]], c);
                    vel[3 * (a + 3) + c] = comp(V[m.elems[p][a]], c);
                }
            for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j)
            {
                Complex s = 0;
                for (int t = 0; t < 18; ++t) s += d[3 * i + j][t] * vel[t];
                atomic_add(D(m.elem_edges[q][i], m.elem_edges[p][j]), s);
            }
        }
    }
    return D;
}

// [CHANGED] y^T (dT/dr_v) x for every vertex v
inline std::vector<C3> T_shape_sensitivity(const Mesh& m, Complex k, const VectorHypersingularShapeOp& op,
                                           const std::vector<Complex>& x, const std::vector<Complex>& y)
{
    std::vector<C3> sens (m.verts.size(), C3{0.0, 0.0, 0.0});
    const long ne = m.elems.size();
#pragma omp parallel for schedule(dynamic, 8)
    for (long q = 0; q < ne; ++q)
    {
        Triangle3 tq = m.elem_primitive(q);
        for (long p = 0; p < ne; ++p)
        {
            Shape9x18 d = op.compute(k, tq, m.elem_primitive(p));
            Complex yx[9];
            for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j)
                yx[3 * i + j] = y[m.elem_edges[q][i]] * x[m.elem_edges[p][j]];
            for (int t = 0; t < 18; ++t)
            {
                Complex s = 0;
                for (int r = 0; r < 9; ++r) s += yx[r] * d[r][t];
                Index v = t < 9 ? m.elems[q][t / 3] : m.elems[p][(t - 9) / 3];
                atomic_add(sens[v][t % 3], s);
            }
        }
    }
    return sens;
}

inline std::vector<Complex> assemble_exc(const Mesh& m, Complex k, const RwgPlaneWave& pw)
{
    std::vector<Complex> b (m.num_edges, 0.0);
    for (Index e = 0; e < m.elems.size(); ++e)
    {
        C3 v = pw.compute(k, m.elem_primitive(e));
        for (int i = 0; i < 3; ++i) b[m.elem_edges[e][i]] += v[i];
    }
    return b;
}

// [CHANGED]
inline std::vector<Complex> assemble_exc_shape_derivative(const Mesh& m, Complex k, const RwgPlaneWave& pw,
                                                          const std::vector<V3>& V)
{
    std::vector<Complex> db (m.num_edges, 0.0);
    for (Index e = 0; e < m.elems.size(); ++e)
    {
        auto d = pw.compute_shape(k, m.elem_primitive(e));
        for (int i = 0; i < 3; ++i)
            for (int a = 0; a < 3; ++a)
                for (int c = 0; c < 3; ++c)
                    db[m.elem_edges[e][i]] += d[i][3 * a + c] * comp(V[m.elems[e][a]], c);
    }
    return db;
}

// [CHANGED]
inline std::vector<C3> exc_shape_sensitivity(const Mesh& m, Complex k, const RwgPlaneWave& pw, const std::vector<Complex>& y)
{
    std::vector<C3> sens (m.verts.size(), C3{0.0, 0.0, 0.0});
    for (Index e = 0; e < m.elems.size(); ++e)
    {
        auto d = pw.compute_shape(k, m.elem_primitive(e));
        for (int i = 0; i < 3; ++i)
            for (int a = 0; a < 3; ++a)
                for (int c = 0; c < 3; ++c)
                    sens[m.elems[e][a]][c] += y[m.elem_edges[e][i]] * d[i][3 * a + c];
    }
    return sens;
}

inline CMat assemble_proj(const Mesh& m, Complex k, const VectorHypersingularProj& pr, const std::vector<V3>& pts)
{
    CMat P (3 * pts.size(), m.num_edges);
    for (Index e = 0; e < m.elems.size(); ++e)
    {
        CMat v = pr.compute(k, pts, m.elem_primitive(e));
        for (Index r = 0; r < v.rows; ++r) for (int j = 0; j < 3; ++j) P(r, m.elem_edges[e][j]) += v(r, j);
    }
    return P;
}

// [CHANGED]
inline CMat assemble_proj_shape_derivative(const Mesh& m, Complex k, const VectorHypersingularProj& pr,
                                           const std::vector<V3>& pts, const std::vector<V3>& V)
{
    CMat dP (3 * pts.size(), m.num_edges);
    for (Index e = 0; e < m.elems.size(); ++e)
    {
        CMat d = pr.compute_shape(k, pts, m.elem_primitive(e));
        for (Index r = 0; r < d.rows; ++r)
            for (int j = 0; j < 3; ++j)
                for (int a = 0; a < 3; ++a)
                    for (int c = 0; c < 3; ++c)
                        dP(r, m.elem_edges[e][j]) += d(r, 9 * j + 3 * a + c) * comp(V[m.elems[e][a]], c);
    }
    return dP;
}

// [CHANGED] y^T (dP/dr_v) x
inline std::vector<C3> proj_shape_sensitivity(const Mesh& m, Complex k, const VectorHypersingularProj& pr,
                                              const std::vector<V3>& pts, const std::vector<Complex>& x,
                                              const std::vector<Complex>& y)
{
    std::vector<C3> sens (m.verts.size(), C3{0.0, 0.0, 0.0});
    for (Index e = 0; e < m.elems.size(); ++e)
    {
        CMat d = pr.compute_shape(k, pts, m.elem_primitive(e));
        for (Index r = 0; r < d.rows; ++r)
            for (int j = 0; j < 3; ++j)
                for (int a = 0; a < 3; ++a)
                    for (int c = 0; c < 3; ++c)
                        sens[m.elems[e][a]][c] += y[r] * d(r, 9 * j + 3 * a + c) * x[m.elem_edges[e][j]];
    }
    return sens;
}


// =================================================================================================
// [NEW, this file only] dense LU with partial pivoting (replaces EigenMatrix::factorize,
// mat_solve and the [CHANGED] mat_solve_transpose)
// =================================================================================================

struct LU
{
    Index N = 0;
    std::vector<Complex> a;
    std::vector<Index> perm;   // (PA)[i] = A[perm[i]]

    explicit LU(const CMat& A): N(A.rows), a(A.a), perm(A.rows)
    {
        for (Index i = 0; i < N; ++i) perm[i] = i;
        for (Index k = 0; k < N; ++k)
        {
            Index piv = k; Float best = std::abs(a[k * N + k]);
            for (Index i = k + 1; i < N; ++i) if (std::abs(a[i * N + k]) > best) { best = std::abs(a[i * N + k]); piv = i; }
            if (piv != k) { std::swap_ranges(a.begin() + k * N, a.begin() + (k + 1) * N, a.begin() + piv * N); std::swap(perm[k], perm[piv]); }
            const Complex inv = 1.0 / a[k * N + k];
            const Float* rk = reinterpret_cast<const Float*>(&a[k * N]);
#pragma omp parallel for schedule(static) if (N - k > 256)
            for (long i = (long)k + 1; i < (long)N; ++i)
            {
                Complex& lik = a[i * N + k];
                lik *= inv;
                const Float fr = lik.real(), fi = lik.imag();
                Float* ri = reinterpret_cast<Float*>(&a[i * N]);
                for (Index j = k + 1; j < N; ++j)
                {
                    const Float ar = rk[2 * j], ai = rk[2 * j + 1];
                    ri[2 * j] -= fr * ar - fi * ai;
                    ri[2 * j + 1] -= fr * ai + fi * ar;
                }
            }
        }
    }

    std::vector<Complex> solve(const std::vector<Complex>& b) const
    {
        std::vector<Complex> y (N);
        for (Index i = 0; i < N; ++i) { Complex s = b[perm[i]]; for (Index j = 0; j < i; ++j) s -= a[i * N + j] * y[j]; y[i] = s; }
        for (Index i = N; i-- > 0;) { Complex s = y[i]; for (Index j = i + 1; j < N; ++j) s -= a[i * N + j] * y[j]; y[i] = s / a[i * N + i]; }
        return y;
    }

    std::vector<Complex> solve_transpose(const std::vector<Complex>& r) const
    {
        std::vector<Complex> z = r;
        for (Index i = 0; i < N; ++i) { z[i] /= a[i * N + i]; const Complex zi = z[i]; for (Index j = i + 1; j < N; ++j) z[j] -= a[i * N + j] * zi; }
        for (Index i = N; i-- > 0;) { const Complex zi = z[i]; for (Index j = 0; j < i; ++j) z[j] -= a[i * N + j] * zi; }
        std::vector<Complex> x (N);
        for (Index i = 0; i < N; ++i) x[perm[i]] = z[i];
        return x;
    }
};

}   // namespace bem


// =================================================================================================
// [NEW] Verification driver
// =================================================================================================

using namespace bem;

static double seconds_since(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

static Float rel(const CMat3& a, const CMat3& ref)
{
    Float n = 0, d = 0;
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) { n += std::norm(a[i][j] - ref[i][j]); d += std::norm(ref[i][j]); }
    return std::sqrt(n / d);
}

// ------------------------------------------------------------------------------------------------
// Part A: element-pair checks (paper Sec. 5.1, Table 1)
// ------------------------------------------------------------------------------------------------
static void part_a()
{
    std::printf("\n================================================================================\n");
    std::printf("PART A - element-pair shape derivatives (paper Sec. 5.1, Table 1)\n");
    std::printf("================================================================================\n");

    // lambda = 1 m (k = 2 pi), triangles of about lambda/10 as in the paper's examples
    const Float f = c0;
    const Complex k = two_pi * f / c0;
    const Float scale = 0.1;

    std::vector<V3> X = {{0, 0, 0}, {1, 0, 0}, {0.3, 0.9, 0}, {0.9, 0.8, 0.5}, {-0.6, -0.3, 0.5},
                         {0.1, 0.2, -0.6}, {0.9, 0.3, -0.9}, {0.4, 0.8, -0.7}, {-0.2, -0.9, 0.3}};
    for (V3& x: X) x = scale * x;

    const V3 tau = (1.0 / std::sqrt(13.0)) * V3{0, 2, 3};   // variation direction used in the paper

    struct Case { const char* name; std::array<int, 3> q, p; int m; };
    const std::vector<Case> cases = {
        {"(a) near : disjoint triangles, node of T_q moved      ", {0, 1, 2}, {5, 6, 7}, 0},
        {"(b) point: shared vertex moved                         ", {0, 1, 2}, {0, 4, 8}, 0},
        {"(c) edge : shared-edge vertex moved                    ", {0, 1, 2}, {2, 1, 3}, 1},
        {"(d) same : coincident triangles, vertex moved          ", {0, 1, 2}, {0, 1, 2}, 2},
        {"(e) edge : non-shared vertex of T_p moved              ", {0, 1, 2}, {2, 1, 3}, 3},
        {"(f) point: non-shared vertex of T_p moved              ", {0, 1, 2}, {0, 4, 8}, 4}};

    VectorHypersingularOp op;
    VectorHypersingularShapeOp sop;
    auto tri = [] (const std::vector<V3>& Y, std::array<int, 3> t) { return Triangle3(Y[t[0]], Y[t[1]], Y[t[2]]); };

    std::printf("\nElement matrix T of the TEFIE (Z = -j omega mu T), relative Frobenius difference\n");
    std::printf("between the analytical derivative (eqs. 24, 26) and finite differences of OpenBEM's T:\n\n");
    std::printf("  %-56s %14s %14s %14s\n", "configuration", "central 1e-5", "central 1e-6", "forward 1e-8");

    for (const Case& c: cases)
    {
        Shape9x18 D = sop.compute(k, tri(X, c.q), tri(X, c.p));
        CMat3 an {};
        for (int a = 0; a < 3; ++a)
            for (int r = 0; r < 9; ++r)
                for (int cc = 0; cc < 3; ++cc)
                {
                    if (c.q[a] == c.m) an[r / 3][r % 3] += D[r][3 * a + cc] * comp(tau, cc);
                    if (c.p[a] == c.m) an[r / 3][r % 3] += D[r][3 * (a + 3) + cc] * comp(tau, cc);
                }

        auto fd = [&] (Float h, bool central) {
            std::vector<V3> Xp = X, Xm = X;
            Xp[c.m] = X[c.m] + (h * scale) * tau;
            if (central) Xm[c.m] = X[c.m] - (h * scale) * tau;
            CMat3 A = op.compute(k, tri(Xp, c.q), tri(Xp, c.p)), B = op.compute(k, tri(Xm, c.q), tri(Xm, c.p));
            CMat3 r;
            for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) r[i][j] = (A[i][j] - B[i][j]) / ((central ? 2 : 1) * h * scale);
            return r;
        };
        std::printf("  %-56s %14.2e %14.2e %14.2e\n", c.name,
                    rel(fd(1e-5, true), an), rel(fd(1e-6, true), an), rel(fd(1e-8, false), an));
    }
    std::printf("\n  The difference does not change with the step: it is the quadrature difference between\n"
                "  integrating the derivative kernels (singularity subtraction, eqs. 24/26) and\n"
                "  differentiating OpenBEM's integrated matrix, largest for touching triangles.\n");

    // ---- plane-wave excitation of one triangle
    RwgPlaneWave pw ({0, 0, 1}, {1, 0, 0}, {0, 0, -100}, 1.0);
    std::printf("\nPlane-wave excitation b = int f . E (one triangle), analytical vs central FD (h = 1e-6):\n");
    {
        Triangle3 t0 = tri(X, {0, 1, 2});
        auto d = pw.compute_shape(k, t0);
        for (int vert = 0; vert < 3; ++vert)
        {
            C3 an {0.0, 0.0, 0.0};
            for (int e = 0; e < 3; ++e) for (int c = 0; c < 3; ++c) an[e] += d[e][3 * vert + c] * comp(tau, c);
            const Float h = 1e-6 * scale;
            std::vector<V3> Xp = X, Xm = X; Xp[vert] = X[vert] + h * tau; Xm[vert] = X[vert] - h * tau;
            C3 bp = pw.compute(k, tri(Xp, {0, 1, 2})), bm = pw.compute(k, tri(Xm, {0, 1, 2}));
            Float n = 0, dd = 0;
            for (int e = 0; e < 3; ++e) { Complex fdv = (bp[e] - bm[e]) / (2 * h); n += std::norm(an[e] - fdv); dd += std::norm(fdv); }
            std::printf("  vertex %d moved: relative difference %.2e\n", vert, std::sqrt(n / dd));
        }
    }

    // ---- field projector of one triangle at far points
    VectorHypersingularProj pr;
    std::vector<V3> far_pts = {{0, 0, -100}, {30, 20, 90}, {-80, 10, 50}};
    std::printf("\nE-field projector at far points (one triangle), analytical vs central FD (h = 1e-6):\n");
    {
        Triangle3 t0 = tri(X, {0, 1, 2});
        CMat d = pr.compute_shape(k, far_pts, t0);
        for (int vert = 0; vert < 3; ++vert)
        {
            const Float h = 1e-6 * scale;
            std::vector<V3> Xp = X, Xm = X; Xp[vert] = X[vert] + h * tau; Xm[vert] = X[vert] - h * tau;
            CMat Pp = pr.compute(k, far_pts, tri(Xp, {0, 1, 2})), Pm = pr.compute(k, far_pts, tri(Xm, {0, 1, 2}));
            Float n = 0, dd = 0;
            for (Index r = 0; r < Pp.rows; ++r)
                for (int j = 0; j < 3; ++j)
                {
                    Complex an = 0;
                    for (int c = 0; c < 3; ++c) an += d(r, 9 * j + 3 * vert + c) * comp(tau, c);
                    Complex fdv = (Pp(r, j) - Pm(r, j)) / (2 * h);
                    n += std::norm(an - fdv); dd += std::norm(fdv);
                }
            std::printf("  vertex %d moved: relative difference %.2e\n", vert, std::sqrt(n / dd));
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Part B: assembled system on the sphere of example 1
// ------------------------------------------------------------------------------------------------
static void part_b(const std::string& mesh_file)
{
    std::printf("\n================================================================================\n");
    std::printf("PART B - assembled TEFIE on %s\n", mesh_file.c_str());
    std::printf("================================================================================\n");

    Mesh mesh;
    if (!read_gmsh_v2(mesh_file, mesh))
    {
        std::printf("  Mesh not found; skipping Part B (run from the `examples` directory).\n");
        return;
    }
    std::printf("  vertices %zu, triangles %zu, edges %zu\n", mesh.verts.size(), mesh.elems.size(), mesh.num_edges);

    // Problem of example 1: 250 MHz, plane wave along +z, E along +x, vacuum
    const Float f = 250e6;
    const Complex k = two_pi * f / c0;
    const Complex scale_Z = -J * two_pi * f * mu0;               // Tefie::j_matrix scaling
    const Float dist = 100 * (c0 / f);
    const V3 pos = {0, 0, -dist};
    const RwgPlaneWave pw ({0, 0, 1}, {1, 0, 0}, pos, 1.0);
    const std::vector<V3> obs = {pos};                           // backscatter point
    const Float rcs_scale = four_pi * dist * dist;

    VectorHypersingularOp op;
    VectorHypersingularShapeOp sop;
    VectorHypersingularProj pr;

    // radial velocity field: d/d(radius)
    V3 center;
    for (const V3& p: mesh.verts) center = center + p;
    center = (1.0 / mesh.verts.size()) * center;
    std::vector<V3> Vel (mesh.verts.size());
    Float radius = 0;
    for (std::size_t i = 0; i < mesh.verts.size(); ++i)
    {
        V3 r = mesh.verts[i] - center;
        radius += norm(r);
        Vel[i] = (1.0 / norm(r)) * r;
    }
    radius /= mesh.verts.size();
    std::printf("  radius %.6f m, ka = %.4f\n", radius, std::real(k) * radius);

    auto moved = [&] (Float h) { Mesh m2 = mesh; for (std::size_t i = 0; i < m2.verts.size(); ++i) m2.verts[i] = mesh.verts[i] + h * Vel[i]; return m2; };

    auto t0 = std::chrono::steady_clock::now();
    CMat T = assemble_T(mesh, k, op);
    std::printf("\n  assembly of T: %.2f s\n", seconds_since(t0));

    // ---- B1: assembled matrix derivative vs FD of the assembled matrix
    t0 = std::chrono::steady_clock::now();
    CMat dT = assemble_T_shape_derivative(mesh, k, sop, Vel);
    std::printf("  analytical dT/dR (radial field): %.2f s\n", seconds_since(t0));
    const Float hm = 1e-4 * radius;
    {
        CMat Tp = assemble_T(moved(hm), k, op);
        CMat Tm = assemble_T(moved(-hm), k, op);
        Float n = 0, d = 0;
        for (std::size_t i = 0; i < T.a.size(); ++i)
        {
            Complex fdv = (Tp.a[i] - Tm.a[i]) / (2 * hm);
            n += std::norm(dT.a[i] - fdv); d += std::norm(fdv);
        }
        std::printf("\n  [B1] || dT/dR (analytical) - dT/dR (central FD) ||_F / || dT/dR (FD) ||_F = %.3e\n", std::sqrt(n / d));
    }

    // ---- B2: adjoint sensitivity of the backscatter RCS
    CMat Z = T;
    for (Complex& v: Z.a) v *= scale_Z;
    std::vector<Complex> b = assemble_exc(mesh, k, pw);
    t0 = std::chrono::steady_clock::now();
    LU lu (Z);
    std::vector<Complex> x = lu.solve(b);
    std::printf("\n  LU factorization and solve: %.2f s\n", seconds_since(t0));

    CMat P = assemble_proj(mesh, k, pr, obs);
    for (Complex& v: P.a) v *= scale_Z;                      // Tefie::j_projector scaling
    std::vector<Complex> e (3, 0.0);
    for (int r = 0; r < 3; ++r) for (Index j = 0; j < mesh.num_edges; ++j) e[r] += P(r, j) * x[j];
    Float rcs = rcs_scale * (std::norm(e[0]) + std::norm(e[1]) + std::norm(e[2]));
    std::printf("  backscatter RCS: %.6f m^2\n", rcs);

    // adjoint: Z^T mu = 2 rcs_scale P^T conj(e)   (paper eq. 11)
    std::vector<Complex> rhs (mesh.num_edges, 0.0);
    for (Index j = 0; j < mesh.num_edges; ++j)
        for (int r = 0; r < 3; ++r) rhs[j] += 2 * rcs_scale * P(r, j) * std::conj(e[r]);
    std::vector<Complex> mu = lu.solve_transpose(rhs);

    // paper eq. 13: d rcs = Re[ mu^T (db - dZ x) ] + 2 rcs_scale Re[ e^H dP x ]
    t0 = std::chrono::steady_clock::now();
    std::vector<C3> sZ = T_shape_sensitivity(mesh, k, sop, x, mu);
    std::vector<C3> sb = exc_shape_sensitivity(mesh, k, pw, mu);
    std::vector<Complex> e_conj = {std::conj(e[0]), std::conj(e[1]), std::conj(e[2])};
    std::vector<C3> sP = proj_shape_sensitivity(mesh, k, pr, obs, x, e_conj);
    Float drcs_adj = 0;
    for (std::size_t v = 0; v < mesh.verts.size(); ++v)
        for (int c = 0; c < 3; ++c)
        {
            Float g = std::real(sb[v][c] - scale_Z * sZ[v][c]) + 2 * rcs_scale * std::real(scale_Z * sP[v][c]);
            drcs_adj += g * comp(Vel[v], c);
        }
    std::printf("  adjoint sensitivities for all %zu vertices: %.2f s\n", mesh.verts.size(), seconds_since(t0));

    // direct method: dx = Z^{-1}(db - dZ x)
    std::vector<Complex> db = assemble_exc_shape_derivative(mesh, k, pw, Vel);
    std::vector<Complex> r_dir = db;
    for (Index i = 0; i < mesh.num_edges; ++i)
        for (Index j = 0; j < mesh.num_edges; ++j) r_dir[i] -= scale_Z * dT(i, j) * x[j];
    std::vector<Complex> dx = lu.solve(r_dir);
    CMat dP = assemble_proj_shape_derivative(mesh, k, pr, obs, Vel);
    std::vector<Complex> de (3, 0.0);
    for (int r = 0; r < 3; ++r)
        for (Index j = 0; j < mesh.num_edges; ++j) de[r] += scale_Z * dP(r, j) * x[j] + P(r, j) * dx[j];
    Float drcs_dir = 2 * rcs_scale * std::real(std::conj(e[0]) * de[0] + std::conj(e[1]) * de[1] + std::conj(e[2]) * de[2]);

    // finite differences of the whole solve
    auto rcs_of = [&] (const Mesh& m) {
        CMat Zm = assemble_T(m, k, op);
        for (Complex& v: Zm.a) v *= scale_Z;
        LU l (Zm);
        std::vector<Complex> xm = l.solve(assemble_exc(m, k, pw));
        CMat Pm = assemble_proj(m, k, pr, obs);
        Float s = 0;
        for (int r = 0; r < 3; ++r) { Complex er = 0; for (Index j = 0; j < m.num_edges; ++j) er += scale_Z * Pm(r, j) * xm[j]; s += std::norm(er); }
        return rcs_scale * s;
    };
    const Float h = 1e-4 * radius;
    Float drcs_fd = (rcs_of(moved(h)) - rcs_of(moved(-h))) / (2 * h);

    std::printf("\n  [B2] d(RCS)/d(radius):\n");
    std::printf("       adjoint (analytical, eqs. 11, 13) : %+.10e m^2/m\n", drcs_adj);
    std::printf("       direct  (analytical dZ, db, dP)   : %+.10e m^2/m\n", drcs_dir);
    std::printf("       central FD of the complete solve  : %+.10e m^2/m\n", drcs_fd);
    std::printf("       relative difference adjoint vs FD : %.3e\n", std::abs(drcs_adj - drcs_fd) / std::abs(drcs_fd));
}


int main(int argc, char** argv)
{
    std::printf("EFIE analytical shape derivatives (Kataja & Toivanen 2013) - single-file verification\n");
#ifdef _OPENMP
    std::printf("OpenMP threads: %d\n", omp_get_max_threads());
#endif

    bool elements_only = false;
    std::string mesh_file = "msh/sphere.msh";
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--elements-only") elements_only = true;
        else mesh_file = a;
    }

    try
    {
        part_a();
        if (!elements_only)
            part_b(mesh_file);
    }
    catch (const std::exception& ex)
    {
        std::fprintf(stderr, "Error: %s\n", ex.what());
        return 1;
    }

    std::printf("\nDone.\n");
    return 0;
}
