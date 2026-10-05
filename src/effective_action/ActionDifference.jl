function prepare_action_difference(
    simplices,
    flat_coordinates,
    boundary_edge,
    delta_lengths;
    bulk_sum_form::Bool=true,
    gamma=1.0,
    verbose::Bool=false,
    check::Bool=false,
)
    flat_data = prepare_action_data(
        simplices,
        flat_coordinates;
        bulk_sum_form=bulk_sum_form,
        curved=false,
        gamma=gamma,
        verbose=verbose,
        check=check,
    )

    split = prepare_regge_transverse_split(
        flat_data,
        simplices,
        flat_coordinates,
        boundary_edge,
        gamma,
    )

    T = eltype(first(values(flat_data.coordinates)))
    delta_lengths = T.(delta_lengths)
    length(delta_lengths) == length(split.perturb_edges) ||
        error("Give one length perturbation for every edge in split.perturb_edges.")

    curved_coordinates = reconstruct_local_coordinates(
        simplices,
        flat_data.coordinates,
        split.perturb_edges,
        delta_lengths,
    )

    curved_data = prepare_action_data(
        simplices,
        curved_coordinates;
        bulk_sum_form=bulk_sum_form,
        curved=true,
        gamma=gamma,
        verbose=verbose,
        check=check,
    )

    problem = prepare_fixed_bulk_length_problem(curved_data, split)

    return (
        simplices=simplices,
        flat_coordinates=flat_coordinates,
        curved_coordinates=curved_coordinates,
        boundary_edge=boundary_edge,
        delta_lengths=delta_lengths,
        bulk_sum_form=bulk_sum_form,
        flat_data=flat_data,
        curved_data=curved_data,
        split=split,
        problem=problem,
    )
end

function evaluate_action_difference(data, complex_point, gamma)
    T = eltype(first(values(data.flat_data.coordinates)))
    # This only substitutes gamma. The symbolic action was constructed above
    # with bulk_sum_form=true.
    flat_exact_spinfoam = LorentzianSimplexSolver.compute_spinfoam_action(
        data.flat_data.exact_spinfoam;
        gamma=gamma,
    )
    flat_action = LorentzianSimplexSolver.ActionEvaluation.to_complex_T(
        flat_exact_spinfoam.action,
        T,
    )

    return (
        gamma=gamma,
        flat_action=flat_action,
        complex_action=complex_point.action,
        action_difference=complex_point.action - flat_action,
    )
end

function solve_action_difference(
    data,
    gamma;
    tol,
    maxiter::Int=30,
    verbose::Bool=false,
    initial=nothing,
)
    complex_point = solve_fixed_bulk_length_complex_point(
        data.problem,
        gamma;
        tol=tol,
        maxiter=maxiter,
        verbose=verbose,
        initial=initial,
    )
    values = evaluate_action_difference(data, complex_point, gamma)

    return merge(complex_point, values)
end
