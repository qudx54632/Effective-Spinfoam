Base.@kwdef struct SolverData
    geom::Any
    regge::Any
    bulk_sum_form::Bool
    log_spinfoam::Any
    log_eom::Any
    log_hessian_symbols::Any
    X_variables::Any
    eta_variables::Any
    all_variables::Any
end

function build_vertex_coordinates(simplices, coords_lines, ::Type{T}) where {T<:Real}
    vertices = sort(unique(Iterators.flatten(simplices)))
    length(vertices) == length(coords_lines) ||
        error("Expected $(length(vertices)) coordinate lines, got $(length(coords_lines)).")

    return Dict{Int, Vector{T}}(
        v => LorentzianSimplexSolver.PrecisionUtils.parse_numeric_line(line, T)
        for (v, line) in zip(vertices, coords_lines)
    )
end

"""Construct the matched flat geometry and logarithmic action used in the analytic expansion."""
function build_solver_data(
    simplices,
    vertex_coords;
    bulk_sum_form::Bool=false,
    verbose::Bool=false,
    sector::Symbol=:ref,
    match_faces::Bool=true,
    gauge_fix::Bool=true,
    check::Bool=false,
    γ=nothing,
    gamma=nothing,
)
    bulk_sum_form && error(
        "The analytic effective-action workflow uses bulk_sum_form=false (the logarithmic bulk action).",
    )

    geom = LorentzianSimplexSolver.construct_geometry(
        simplices,
        vertex_coords;
        verbose=verbose,
    )
    LorentzianSimplexSolver.prepare_global_geometry!(
        geom,
        simplices;
        sector=sector,
        match_faces=match_faces,
        gauge_fix=gauge_fix,
        check=check,
        verbose=verbose,
    )

    regge = LorentzianSimplexSolver.compute_regge_action(
        geom,
        simplices,
        vertex_coords,
    )
    log_spinfoam = LorentzianSimplexSolver.compute_spinfoam_action(
        geom,
        regge;
        γ=γ,
        gamma=gamma,
        bulk_sum_form=bulk_sum_form,
    )

    X_variables = vcat(geom.varias[:g_var], geom.varias[:z_var])
    eta_variables = geom.varias[:η_var]
    all_variables = vcat(X_variables, eta_variables)
    log_eom = LorentzianSimplexSolver.compute_eom(log_spinfoam; variables=all_variables)
    log_hessian = LorentzianSimplexSolver.compute_hessian(
        geom,
        log_spinfoam;
        variables=all_variables,
        eom=log_eom,
    )

    return SolverData(
        geom=geom,
        regge=regge,
        bulk_sum_form=bulk_sum_form,
        log_spinfoam=log_spinfoam,
        log_eom=log_eom,
        log_hessian_symbols=log_hessian.symbols,
        X_variables=X_variables,
        eta_variables=eta_variables,
        all_variables=all_variables,
    )
end

solver_real_type(::LorentzianSimplexSolver.SolveVars.SolveData{T}) where {T<:Real} = T
solver_real_type(action::LorentzianSimplexSolver.SpinfoamActionData) =
    solver_real_type(action.solve_data)
solver_real_type(data::SolverData) = solver_real_type(data.log_spinfoam)

function validate_solver_number(value, ::Type{T}, name::AbstractString) where {T<:Real}
    LorentzianSimplexSolver.PrecisionUtils.validate_number_precision(value, T, name)
    return nothing
end

function resolve_gamma_value(γ, gamma, default)
    if γ !== nothing && gamma !== nothing
        error("Pass either γ or gamma, not both.")
    end

    value = γ !== nothing ? γ : gamma
    return value === nothing ? default : value
end

function critical_value_dict(
    solve_data,
    gamma_symbol::Basic;
    γ=nothing,
    gamma=nothing,
)
    T = solver_real_type(solve_data)
    gamma_value = resolve_gamma_value(γ, gamma, one(T))
    validate_solver_number(gamma_value, T, "gamma")
    return LorentzianSimplexSolver.ActionEvaluation.build_value_dict(
        solve_data,
        gamma_symbol;
        γval=gamma_value,
    )
end

critical_value_dict(action::LorentzianSimplexSolver.SpinfoamActionData; γ=nothing, gamma=nothing) =
    critical_value_dict(action.solve_data, action.gamma_symbol; γ=γ, gamma=gamma)

critical_value_dict(data::SolverData; γ=nothing, gamma=nothing) =
    critical_value_dict(data.log_spinfoam; γ=γ, gamma=gamma)

function evaluate_log_hessian(data::SolverData; γ=nothing, gamma=nothing)
    T = solver_real_type(data)
    gamma_value = resolve_gamma_value(γ, gamma, one(T))
    validate_solver_number(gamma_value, T, "gamma")
    return LorentzianSimplexSolver.Hessian.evaluate_hessian_block(
        data.log_hessian_symbols,
        data.log_spinfoam.solve_data;
        γ=gamma_value,
    )
end

function evaluate_log_action(
    data::SolverData,
    critical_values::Dict{Basic,Basic},
)
    return SymEngine.expand(
        LorentzianSimplexSolver.ActionEvaluation.eval_symbolic(
            data.log_spinfoam.action_no_phase,
            critical_values,
        ),
    )
end

to_solver_complex(value, ::Type{T}) where {T<:Real} =
    LorentzianSimplexSolver.ActionEvaluation.to_complex_T(value, T)
