module EffectiveSpinfoamWorkflow

using LinearAlgebra
using SymEngine

using LorentzianSimplexSolver

include(joinpath(@__DIR__, "derivatives", "EtaLengthDerivatives.jl"))
include(joinpath(@__DIR__, "derivatives", "LogFaceDerivatives.jl"))
include(joinpath(@__DIR__, "perturbations", "TransverseBasis.jl"))
include(joinpath(@__DIR__, "perturbations", "BoundaryAreaDerivatives.jl"))
include(joinpath(@__DIR__, "perturbations", "DihedralAngleDerivatives.jl"))

# Complex critical-point action difference for a length perturbation.
include(joinpath(@__DIR__, "geometry", "CoordinateReconstruction.jl"))
include(joinpath(@__DIR__, "effective_action", "SpinfoamData.jl"))
include(joinpath(@__DIR__, "effective_action", "ReggeTransverseSplit.jl"))
include(joinpath(@__DIR__, "effective_action", "FixedBulkLengthComplexPoint.jl"))
include(joinpath(@__DIR__, "effective_action", "ActionDifference.jl"))

using .EtaLengthDerivatives
using .LogFaceDerivatives
using .TransverseBasis
using .BoundaryAreaDerivatives
using .DihedralAngleDerivatives

include(joinpath(@__DIR__, "workflow", "SolverStage.jl"))
include(joinpath(@__DIR__, "workflow", "EffectiveData.jl"))
include(joinpath(@__DIR__, "workflow", "EffectiveMatrices.jl"))
include(joinpath(@__DIR__, "workflow", "BoundaryTaylor.jl"))

export build_vertex_coordinates,
       build_solver_data,
       build_effective_data,
       evaluate_effective_blocks,
       compute_effective_matrices,
       compute_effective_terms,
       compute_boundary_taylor_terms,
       print_effective_terms,
       scan_K_kernel,
       prepare_action_data,
       prepare_action_difference,
       solve_action_difference

end # module
