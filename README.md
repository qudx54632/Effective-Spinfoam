# Effective Spinfoam

`EffectiveSpinfoam` is a Julia package for the numerical effective Spinfoam action on Lorentzian Regge geometries. It constructs the Regge and non-Regge quadratic terms in a chosen length-perturbation space, evaluates the hinge-space correction kernel `K_kernel`, and compares the analytic expansion with the exact action evaluated at a fixed-length complex critical point.

The package uses the local `LorentzianSimplexSolver` package in `packages/LorentzianSimplexSolver` for Lorentzian simplex geometry, face matching, gauge fixing, Regge data, symbolic Spinfoam actions, equations of motion, and Hessians. The two packages are kept together so that this repository can be used from a local clone without a sibling checkout.

## Two calculations

The analytic effective-action calculation and the complex-critical-point calculation use two different forms of the bulk action.

The analytic workflow uses the logarithmic bulk action, selected by `bulk_sum_form=false`:

```julia
solver = build_solver_data(
    simplices,
    vertex_coords;
    bulk_sum_form=false,
)
data = build_effective_data(solver, simplices, vertex_coords)
terms = compute_effective_terms(data, gamma)
```

The returned `terms` contains the Regge terms, the non-Regge correction, and the kernel:

```julia
terms.linear_regge
terms.quadratic_regge
terms.non_regge_correction
terms.spinfoam_quadratic
terms.K_kernel
```

The complex-critical-point workflow uses the exact bulk action, selected by `bulk_sum_form=true`. It reconstructs the perturbed local simplex coordinates, fixes the selected lengths, solves the complex equations of motion, and evaluates the action difference:

```julia
data = prepare_action_difference(
    simplices,
    flat_coordinates,
    boundary_edge,
    delta_lengths;
    bulk_sum_form=true,
    gamma=gamma,
)
result = solve_action_difference(
    data,
    gamma;
    tol=5e-14,
    maxiter=30,
)
```

The numerical result is stored in `result.action_difference`. The exact action at the perturbed complex critical point is `result.complex_action`, the flat reference action is `result.flat_action`, and the final fixed-length equation-of-motion residual is `maximum(abs.(result.residual))`.

## Installation from a local clone

This repository is intended to be run from a local clone. From the package root, instantiate the project with:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Then load the package with:

```julia
using EffectiveSpinfoam
using LorentzianSimplexSolver
```

The path dependency in `Project.toml` points to `packages/LorentzianSimplexSolver`, so the whole repository must be kept together.

## Input and precision

Coordinates use the order `(t, x, y, z)`. For a flat geometry, provide one coordinate vector for each global vertex. The helper `build_vertex_coordinates` reads coordinate strings into the selected scalar type:

```julia
vertex_coords = build_vertex_coordinates(simplices, coords_lines, Float64)
```

For a curved Regge geometry, provide five local coordinate vectors for every 4-simplex, in the same order as the five vertex labels in that simplex. Every 4-simplex is flat internally, but the complete triangulation need not have one global Minkowski embedding.

Set the scalar type before constructing coordinates or numerical values:

```julia
configure_precision!(Float64)
configure_precision!(BigFloat; precision=256)
```

Use the same scalar type for `gamma` and for the length perturbations. In `prepare_action_difference`, `delta_lengths` is ordered as all bulk edges followed by the selected boundary edge or edges. The exact order is available as `data.perturb_edges`.

Flat and curved coordinate examples are collected in `packages/LorentzianSimplexSolver/examples/example-complex.txt`.

## Examples

Run commands from the package root.

```bash
julia --project=. examples/analytic_action_difference.jl
julia --project=. examples/complex_action_difference.jl
julia --project=. examples/action_difference_table.jl
```

`analytic_action_difference.jl` evaluates the analytic linear and quadratic terms for a prescribed length perturbation. `complex_action_difference.jl` evaluates the exact action difference at the corresponding complex critical point. `action_difference_table.jl` compares the two results for the double-Delta3 and 1-5 examples, including bulk-only, boundary-only, and simultaneous bulk-and-boundary perturbations.

The two scan scripts produce the remainder plots used in the numerical checks:

```bash
julia --project=. examples/double_delta3_action_difference_scan.jl
julia --project=. examples/one_five_action_difference_scan.jl
```

The script `examples/run_gamma_scan_Kkernel.jl` scans `gamma` for the geometry and coordinates defined in that file and saves the plot of `gamma^2 * ||K||` in `results/`. To use another triangulation, replace the simplex and coordinate data in the script.

## Tests

Run the package test from the package root:

```bash
julia --project=. test/runtests.jl
```

The test constructs the 1-5 flat geometry, builds the analytic effective-action data, and checks the kernel dimensions, finite numerical values, and decomposition of the quadratic Spinfoam term.

## Repository layout

```text
Project.toml                         package environment
Manifest.toml                        resolved local environment
src/                                EffectiveSpinfoam source code
  EffectiveSpinfoam.jl               public package entry point
  workflow/                           public analytic workflow
  derivatives/                        area, face, and angle derivatives
  perturbations/                      Regge/transverse perturbation data
  effective_action/                   fixed-length complex action workflow
  geometry/                           local coordinate reconstruction
packages/LorentzianSimplexSolver/    local geometry and Spinfoam solver package
examples/                            runnable numerical examples and scans
notebooks/                            research notebooks
test/                                package tests
results/                             generated numerical data and figures
```

The files in `results/` are output artifacts and are not required for loading the package. The core public entry point is `src/EffectiveSpinfoam.jl`; the lower-level files under `src/` are included by the package and are not meant to be loaded individually.

## Scope

The package is intended for numerical studies of Lorentzian Regge geometries, effective Spinfoam corrections, and complex critical points. It is a research package rather than a general-purpose geometry library.
