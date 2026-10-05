module EffectiveSpinfoam

include("EffectiveSpinfoamWorkflow.jl")

using .EffectiveSpinfoamWorkflow:
    build_vertex_coordinates,
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

end
