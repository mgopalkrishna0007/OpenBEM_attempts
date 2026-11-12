# Changes

## [Unreleased]

### Added

- Analytical shape derivatives of the TEFIE following Kataja and Toivanen, "On shape differentiation of discretized electric field integral equation", Eng. Anal. Bound. Elem. 37 (2013).
- `ShapeOperatorBase` and `VectorHypersingularShapeOp` for element-pair shape derivatives of the vector hypersingular operator.
- Shape-derivative assembly (`assemble_shape_derivative()`) and adjoint sensitivity contraction (`shape_sensitivity()`) in `OperatorAssembler`, `ExcitationAssembler`, and `ProjectorAssembler`.
- `compute_shape()` for `RwgPlaneWave` and `VectorHypersingularProj`, with default (throwing) implementations in `ExcitationBase` and `ProjectorBase`.
- Shape-derivative methods in `Tefie`: `j_matrix_shape_derivative()`, `j_matrix_shape_sensitivity()`, `exc_matrix_shape_derivative()`, `exc_matrix_shape_sensitivity()`, `j_projector_shape_derivative()`, and `j_projector_shape_sensitivity()`.
- Moments of the kernel gradient (`SrcResult::rs_grad_g`, `SrcResult::rsrs_grad_g`, `ObsResult::grad_g_shape`) with analytical singularity subtraction in `SrcSingularity`, and kernel Hessian moments (`SrcResult::hess_g`, `SrcResult::rs_hess_g`).
- `HGF::compute_hess()` and a default `ScalarKernelBase::compute_hess()`.
- `MatrixBase::mat_solve_transpose()` and its `EigenMatrix` implementation, for adjoint solves that reuse the forward factorization.
- Example 4: adjoint sensitivity of the radar cross-section of a PEC sphere with respect to its radius.

### Changed

- `SrcIntegratorBase::integrate()` and `ObsIntegratorBase::integrate()` take optional flags for the shape-derivative terms; existing calls are unaffected.

## [1.1.0] - July 11, 2026

### Added

- Changelog.
- `IndexGenerator` helper class.
- `BoundingBox` geometry class.
- `IndexSet` to manage index blocks.
- `BlockAssembler` to allow assembling (not necessarily contiguous) index blocks.

### Changed

- Simplified the Eigen matrix wrapper class and unified its sparse and dense interface.
- Improved interoperability between sparse and dense Eigen matrices.
- `factorize()` must now be called explicitly before calling `mat_solve` in `EigenMatrix`.
- Line integration is now only vectorized in source points, not observation points, which is a little faster.
- Line integration speed improvements and fixed a bug in the directed angle computation.
- Simplified `TriangleMesh` interface and added a map from edges to associated elements.
- Refactored operator and assembler classes and added support for multiple operator generation.
- Changed assembler file names for consistency with class names.
- Refactored operator, integrator, and quadrature classes to remove templates, use cpp instead of tpp files, and make the computational routines all thread-safe.

### Fixed

- AEFIE's divergence matrix orientation fixed.

## [1.0.2] - April 05, 2025

### Added

- Virtual destructors added to all base classes.

### Changed

- Vectorized kernels for source points.
- Improved and simplified CMakeLists compile options.

### Fixed

- Fixed compiler warnings.
- Typos fixed in docs.

## [1.0.0] - March 30, 2026

Initial release.
