# EffectiveSpinfoam

`EffectiveSpinfoam.jl` is a Julia package accompanying the numerical study of the effective EPRL spinfoam action. It computes the Regge and non-Regge quadratic terms for selected length perturbations, evaluates the hinge-space kernel, and compares the quadratic expansion with the exact action at a complex critical point.

`EffectiveSpinfoam` depends on [`LorentzianSimplexSolver`](https://github.com/qudx54632/LorentzianSimplexSolver) for Lorentzian geometry reconstruction, the Regge and EPRL spinfoam actions, equations of motion, Hessians, and complex critical-point calculations. For reproducibility, this repository includes the exact solver source used for the paper in `packages/LorentzianSimplexSolver`. The `[sources]` entry in `Project.toml` resolves the dependency to this local copy, so the `packages/LorentzianSimplexSolver` directory must remain in a cloned repository.

## Requirements and installation

The project requires Julia 1.11 or later and has been tested locally with Julia 1.12.3. Clone the repository and instantiate its environment:

```bash
git clone https://github.com/qudx54632/Spinfoam-effective-project.git
cd Spinfoam-effective-project
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The committed `Manifest.toml` records the Julia 1.12.3 environment used for the paper. With another Julia minor version, run `julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'` to update the environment for that version.

Check that both packages load:

```bash
julia --project=. -e 'using EffectiveSpinfoam, LorentzianSimplexSolver'
```

The `LorentzianSimplexSolver` package can also be installed and used independently from its GitHub repository; see its README for that workflow.

## Action conventions

The analytic effective-action workflow uses the logarithmic bulk form, selected explicitly with `bulk_sum_form=false`:

```julia
using EffectiveSpinfoam, LorentzianSimplexSolver

solver = build_solver_data(simplices, vertex_coords; bulk_sum_form=false)
data = build_effective_data(solver, simplices, vertex_coords)
terms = compute_effective_terms(data, gamma)
```

The direct complex critical-point calculation uses the exact bulk form, selected with `bulk_sum_form=true`:

```julia
data = prepare_action_difference(
    simplices,
    flat_coordinates,
    boundary_edge,
    delta_lengths;
    bulk_sum_form=true,
    gamma=gamma,
)
result = solve_action_difference(data, gamma; tol=5e-14, maxiter=30)
```

These snippets use the simplices, coordinates, `gamma`, and perturbations supplied in the examples below. The analytic result is obtained by contracting `terms.linear_regge` with the length-perturbation vector and `terms.quadratic_regge + terms.non_regge_correction` with two copies of that vector. The direct action difference is `result.action_difference`. Check `result.converged` before using the result. The field `result.residual` contains the complex equations-of-motion residuals.

## Reproducing the calculations

Run the following commands from the repository root. They overwrite the corresponding generated files in `results/`.

| Script | Calculation and output |
| --- | --- |
| `examples/run_gamma_scan_Kkernel.jl` | Hypercube scan of `gamma^2 * \|K\|`; writes a PDF and the scan data to `results/`. |
| `examples/action_difference_table.jl` | Double-Delta3 and 1-5 move comparisons at `gamma=1` and `delta_l=1e-5`; writes `results/action_difference_table.txt`. |
| `examples/action_difference_uniform_direction_scan.jl` | Uniform perturbation directions for Double-Delta3 and 1-5 move at `gamma=1` and `gamma=0.01`; writes PDF plots and text data to `results/`. |
| `examples/action_difference_nonuniform_direction_scan.jl` | Two non-uniform bulk perturbation profiles for the 1-5 move at `gamma=1`; writes PDF plots and text data to `results/`. |

For example, run one calculation with:

```bash
julia --project=. examples/action_difference_uniform_direction_scan.jl
```

The tracked files in `results/` are the numerical data and figures accompanying the paper. The uniform script produces four PDFs and four text files. The non-uniform script produces two PDFs and two text files. Both use 160-bit `BigFloat` arithmetic and fit over `1e-6 <= delta_L <= 1e-4`. The table script uses `Float64`.

The gamma-scan script currently contains the hypercube input. The other tracked gamma-scan PDFs come from earlier runs with different simplex and coordinate inputs. To scan another geometry, change `simplices`, `coords_lines`, the title, and both output filenames in that script.

## Precision and input data

Coordinates use the order `(t, x, y, z)`. A flat geometry takes one coordinate vector per global vertex. A curved Regge geometry takes five local coordinate vectors per 4-simplex, ordered as the vertex labels in that simplex. Configure the numeric type before creating coordinates or perturbations:

```julia
configure_precision!(Float64)
configure_precision!(BigFloat; precision=160)
```

Choose one precision setting before constructing the input. For `BigFloat`, parse decimal strings, for example `BigFloat("0.01")`, to avoid first rounding the input to `Float64`.

Use the same numeric type for coordinates, `gamma`, and length perturbations. The analytic perturbation order is stored in `data.perturb_edges`. For the object returned by `prepare_action_difference`, the order is stored in `data.split.perturb_edges`. Both list the bulk edges and selected boundary edge.

## Tests

Run the package tests from the repository root:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

The current test suite checks the analytic 1-5 move workflow, transverse-basis dimensions and orthogonality, finite kernel values, and the decomposition of the quadratic spinfoam term. The long scans and plotting scripts are run separately.

To run tests from an existing Julia session, activate the repository first:

```julia
using Pkg
Pkg.activate(".")  # Run from the repository root.
include("test/runtests.jl")
```

## Repository layout

```text
Project.toml                         package environment
Manifest.toml                        resolved paper environment
src/                                 EffectiveSpinfoam source
packages/LorentzianSimplexSolver/    local source dependency
examples/                            paper calculation scripts
notebooks/                           exploratory calculations
test/                                package tests
results/                             paper figures and numerical data
```
