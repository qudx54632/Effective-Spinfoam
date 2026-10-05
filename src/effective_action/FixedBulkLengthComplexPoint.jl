const ComplexCriticalPoints = LorentzianSimplexSolver.ComplexCriticalPoints
const SymBasic = ComplexCriticalPoints.Basic

symbolic_number(x) = SymBasic(string(x))

function scaled_symbolic_value(value, flag, gamma_symbol)
    number = symbolic_number(value)
    return flag ? number / gamma_symbol : number
end

function fixed_boundary_substitutions(solve_data, gamma_symbol)
    Dict(
        solve_data.labels_bdry[i] => scaled_symbolic_value(
            solve_data.values_bdry[i],
            solve_data.flags_bdry[i],
            gamma_symbol,
        )
        for i in eachindex(solve_data.labels_bdry)
    )
end

function prepare_fixed_bulk_length_problem(curved_data, split)
    exact_spinfoam = curved_data.exact_spinfoam
    solve_data = exact_spinfoam.solve_data
    gamma_symbol = exact_spinfoam.gamma_symbol
    eta_variables = collect(curved_data.geometry.varias[:η_var])
    eta_set = Set(eta_variables)
    X_variables = [
        variable for variable in solve_data.labels_vars
        if !(variable in eta_set)
    ]
    variables = vcat(X_variables, eta_variables)

    action = exact_spinfoam.action_no_phase

    eom_dict = ComplexCriticalPoints.compute_EOMs(action, variables)
    eom_X = [eom_dict[variable] for variable in X_variables]
    eom_eta = [eom_dict[variable] for variable in eta_variables]

    hessian = ComplexCriticalPoints.compute_Hessian_block(
        action,
        variables;
        eom=eom_dict,
    )

    nX = length(X_variables)
    H_XX = hessian[1:nX, 1:nX]
    H_Xeta = hessian[1:nX, (nX + 1):end]
    H_etaeta = hessian[(nX + 1):end, (nX + 1):end]

    eta_indices = [
        findfirst(==(variable), solve_data.labels_η)
        for variable in eta_variables
    ]

    return (
        action=action,
        eom_X=eom_X,
        eom_eta=eom_eta,
        H_XX=H_XX,
        H_Xeta=H_Xeta,
        H_etaeta=H_etaeta,
        variables=variables,
        X_variables=X_variables,
        eta_variables=eta_variables,
        gamma_symbol=gamma_symbol,
        solve_data=solve_data,
        transverse_basis=split.transverse_basis,
        eta_values=solve_data.values_η[eta_indices],
        eta_flags=solve_data.flags_η[eta_indices],
        X_indices=[
            findfirst(==(variable), solve_data.labels_vars)
            for variable in X_variables
        ],
    )
end

function pseudo_critical_point(problem, gamma)
    solve_data = problem.solve_data
    T = promote_type(typeof(float(gamma)), eltype(solve_data.values_vars))
    X_values = Complex{T}[
        solve_data.flags_vars[index] ?
        solve_data.values_vars[index] / gamma :
        solve_data.values_vars[index]
        for index in problem.X_indices
    ]
    t_values = zeros(Complex{T}, size(problem.transverse_basis, 2))
    return vcat(X_values, t_values)
end

function pseudo_critical_eta(problem, gamma, ::Type{T}) where {T<:Real}
    return Complex{T}[
        problem.eta_flags[i] ?
        problem.eta_values[i] / gamma :
        problem.eta_values[i]
        for i in eachindex(problem.eta_values)
    ]
end

function variables_from_X_t(problem, gamma, point, ::Type{T}) where {T<:Real}
    number_of_X = length(problem.X_variables)
    X = point[1:number_of_X]
    t = point[(number_of_X + 1):end]
    eta = pseudo_critical_eta(problem, gamma, T) + problem.transverse_basis * t
    return vcat(X, eta)
end

function evaluate_at_X_t(expressions, problem, gamma, point, ::Type{T}) where {T<:Real}
    gamma_value = symbolic_number(gamma)
    fixed = fixed_boundary_substitutions(
        problem.solve_data,
        gamma_value,
    )
    fixed[problem.gamma_symbol] = gamma_value
    values = fixed
    X_eta = variables_from_X_t(problem, gamma, point, T)
    for i in eachindex(problem.variables)
        values[problem.variables[i]] = SymBasic(X_eta[i])
    end

    output = zeros(Complex{T}, length(expressions))
    for i in eachindex(expressions)
        expression = expressions[i]
        expression == 0 && continue
        value = LorentzianSimplexSolver.ActionEvaluation.eval_symbolic(
            expression,
            values,
        )
        output[i] = LorentzianSimplexSolver.ActionEvaluation.to_complex_T(value, T)
    end
    return output
end

function evaluate_X_t_hessian(problem, gamma, point, ::Type{T}) where {T<:Real}
    e_hat = problem.transverse_basis
    H_XX = reshape(
        evaluate_at_X_t(vec(problem.H_XX), problem, gamma, point, T),
        size(problem.H_XX),
    )
    H_Xeta = reshape(
        evaluate_at_X_t(vec(problem.H_Xeta), problem, gamma, point, T),
        size(problem.H_Xeta),
    )
    H_etaeta = reshape(
        evaluate_at_X_t(vec(problem.H_etaeta), problem, gamma, point, T),
        size(problem.H_etaeta),
    )

    H_Xt = H_Xeta * e_hat
    H_tX = transpose(e_hat) * transpose(H_Xeta)
    H_tt = transpose(e_hat) * H_etaeta * e_hat

    return [
        H_XX H_Xt
        H_tX H_tt
    ]
end

function solve_fixed_bulk_length_complex_point(
    problem,
    gamma;
    tol,
    maxiter::Int=30,
    verbose::Bool=true,
    initial=nothing,
)
    T = promote_type(
        typeof(float(gamma)),
        eltype(problem.solve_data.values_vars),
    )
    point = initial === nothing ?
        pseudo_critical_point(problem, gamma) : copy(initial)
    starting_point = copy(point)
    residual_X = evaluate_at_X_t(problem.eom_X, problem, gamma, point, T)
    residual_eta = evaluate_at_X_t(problem.eom_eta, problem, gamma, point, T)
    residual = vcat(
        residual_X,
        transpose(problem.transverse_basis) * residual_eta,
    )
    iterations = 0

    for iteration in 1:maxiter
        iterations = iteration
        residual_size = maximum(abs.(residual))
        verbose && println("complex Newton $iteration: max |dS| = $residual_size")
        residual_size < tol && break

        hessian = evaluate_X_t_hessian(problem, gamma, point, T)
        point -= hessian \ residual
        residual_X = evaluate_at_X_t(problem.eom_X, problem, gamma, point, T)
        residual_eta = evaluate_at_X_t(problem.eom_eta, problem, gamma, point, T)
        residual = vcat(
            residual_X,
            transpose(problem.transverse_basis) * residual_eta,
        )
    end

    action = evaluate_at_X_t(
        [problem.action],
        problem,
        gamma,
        point,
        T,
    )[1]

    return (
        initial=starting_point,
        solution=point,
        residual=residual,
        action=action,
        iterations=iterations,
        converged=maximum(abs.(residual)) < tol,
    )
end
