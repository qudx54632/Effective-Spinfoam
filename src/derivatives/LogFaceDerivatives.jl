module LogFaceDerivatives

export compute_dlogEh_dX, evaluate_dlogEh_dX

using SymEngine
using LorentzianSimplexSolver

function evaluate_at_critical_point(
    expr,
    evaluation_values,
    ::Type{T},
) where {T<:Real}
    expr == 0 && return zero(Complex{T})
    value = subs(expr, evaluation_values)
    return to_solver_complex(value, T)
end

function critical_values_for_gamma(sd, gamma_value, critical_values)
    critical_values !== nothing && return critical_values

    gamma_symbol = LorentzianSimplexSolver.DefineAction.γsym()
    return LorentzianSimplexSolver.ActionEvaluation.build_value_dict(
        sd,
        gamma_symbol;
        γval=gamma_value,
    )
end

function add_boundary_phase_values(critical_values, phase_solution)
    phase_solution === nothing && return critical_values

    # The phase solution was constructed at the critical geometry and only
    # retains gamma_symbol. Evaluate gamma once, before the large derivative loops.
    gamma_symbol = LorentzianSimplexSolver.DefineAction.γsym()
    gamma_values = Dict{Basic,Basic}(
        gamma_symbol => critical_values[gamma_symbol],
    )
    evaluation_values = copy(critical_values)
    for (phase_symbol, phase_expression) in phase_solution
        evaluation_values[phase_symbol] =
            subs(phase_expression, gamma_values)
    end
    return evaluation_values
end

function internal_face_product(zlist, glist, κlist, indices; gamma_symbol::Basic)
    product = LorentzianSimplexSolver.DefineAction.symone(κlist[1])
    for i in indices
        previous = i == 1 ? length(zlist) : i - 1
        product *= LorentzianSimplexSolver.DefineAction.Eehss(
            (zlist[previous], zlist[i + 1]),
            (glist[i], glist[i + 1]),
            (κlist[i], κlist[i + 1]);
            γ=gamma_symbol,
        )
    end
    return product
end

# ------------------------------------------------------------
# 1. log E_h symbolic
# ------------------------------------------------------------
function log_Eh(
    gvariablesall,
    zvariablesall,
    kappaMat,
    OrderBulkFaces;
    gamma_symbol::Basic=LorentzianSimplexSolver.DefineAction.γsym(),
)

    logEh_list = Vector{Basic}(undef, length(OrderBulkFaces))

    for idx in eachindex(OrderBulkFaces)
        faces = OrderBulkFaces[idx]
        zlist = [zvariablesall[k][i][j] for (k,i,j) in faces]
        glist = [gvariablesall[k][i] for (k,i,j) in faces]
        κlist = [kappaMat[k][i][j] for (k,i,j) in faces]

        product = internal_face_product(
            zlist,
            glist,
            κlist,
            1:2:length(faces);
            gamma_symbol=gamma_symbol,
        )
        logEh = LorentzianSimplexSolver.DefineAction.slog(product)

        logEh_list[idx] = logEh
    end

    return logEh_list
end

function compute_dlogEh_dX(logEh::Basic, X_vars::Vector{Basic})
    dlogEh_dX = Dict{Basic,Basic}()
    for v in X_vars
        dlogEh_dX[v] = SymEngine.diff(logEh, v)
    end
    return dlogEh_dX
end

function compute_dlogEh_dX(geom)
    z_mat  = geom.varias[:z_mat]
    g_mat  = geom.varias[:g_mat]

    kappa  = [geom.simplex[i].kappa for i in eachindex(geom.simplex)]

    OrderBulkFaces = geom.connectivity[1]["OrderBulkFaces"]

    # --- symbolic log E_h ---
    gamma_symbol = LorentzianSimplexSolver.DefineAction.γsym()
    logEh_list = log_Eh(
        g_mat,
        z_mat,
        kappa,
        OrderBulkFaces;
        gamma_symbol=gamma_symbol,
    )
    
    # --- background solution ---
    X_vars = vcat(geom.varias[:g_var], geom.varias[:z_var])

    dlogEh_dX_sym = [compute_dlogEh_dX(logEh, X_vars) for logEh in logEh_list]

    return dlogEh_dX_sym
end

function evaluate_dlogEh_dX(
    dlogEh_dX_sym,
    geom,
    sd;
    gamma_value=nothing,
    critical_values=nothing,
)
    T = solver_real_type(sd)
    critical_values = critical_values_for_gamma(sd, gamma_value, critical_values)

    X_vars = vcat(geom.varias[:g_var], geom.varias[:z_var])

    nh = length(dlogEh_dX_sym)
    ng = length(X_vars)

    dlogEh_dX_vals = Matrix{Complex{T}}(undef, nh, ng)

    for h in 1:nh
        for (i, v) in enumerate(X_vars)

            dlogEh_dX_vals[h, i] = evaluate_at_critical_point(
                dlogEh_dX_sym[h][v], critical_values, T,
            )
        end
    end

    return dlogEh_dX_vals
end

# ------------------------------------------------------------
# 2. log E_b
# ------------------------------------------------------------
function log_Eb(
    gvariablesall,
    zvariablesall,
    xivariablesall,
    kappaMat,
    OrderBDryFaces;
    gamma_symbol::Basic=LorentzianSimplexSolver.DefineAction.γsym(),
)

    logEb_list = Vector{Basic}(undef, length(OrderBDryFaces))

    for idx in eachindex(OrderBDryFaces)
        faces = OrderBDryFaces[idx]

        zlist  = [zvariablesall[k][i][j] for (k,i,j) in faces]
        glist  = [gvariablesall[k][i] for (k,i,j) in faces]
        xilist = [xivariablesall[k][i][j] for (k,i,j) in faces]
        κlist  = [kappaMat[k][i][j] for (k,i,j) in faces]

        product = internal_face_product(
            zlist,
            glist,
            κlist,
            2:2:(length(faces)-1);
            gamma_symbol=gamma_symbol,
        )

        term = LorentzianSimplexSolver.DefineAction.EebssSource(
                   glist[1], zlist[1], xilist[1], κlist[1]; γ=gamma_symbol,
               ) *
               product *
               LorentzianSimplexSolver.DefineAction.EebssTarget(
                   glist[end], zlist[end-1], xilist[end], κlist[end]; γ=gamma_symbol,
               )

        logEb_list[idx] = LorentzianSimplexSolver.DefineAction.slog(term)
    end

    return logEb_list
end


function compute_dlogEb_dXY(logEb::Basic, X_vars::Vector{Basic}, Y_vars::Vector{Basic})
    return (
        Dict(v => SymEngine.diff(logEb, v) for v in X_vars),
        Dict(v => SymEngine.diff(logEb, v) for v in Y_vars),
    )
end


function compute_dlogEb_dXY(geom)
    z_mat  = geom.varias[:z_mat]
    g_mat  = geom.varias[:g_mat]
    xi_mat = geom.varias[:xi_mat]

    kappa  = [geom.simplex[i].kappa for i in eachindex(geom.simplex)]
    OrderBDryFaces = geom.connectivity[1]["OrderBDryFaces"]

    gamma_symbol = LorentzianSimplexSolver.DefineAction.γsym()

    logEb_list = log_Eb(
        g_mat,
        z_mat,
        xi_mat,
        kappa,
        OrderBDryFaces;
        gamma_symbol=gamma_symbol,
    )

    # flatten variable containers
    X_vars = vcat(vec(geom.varias[:g_var]), vec(geom.varias[:z_var]))

    eta_bdry = [LorentzianSimplexSolver.DefineSymbols.make_symbol("η_$(faces[1][1])$(faces[1][2])$(faces[1][3])") for faces in OrderBDryFaces]
    
    xi_bdry = Basic[]
    for i in eachindex(OrderBDryFaces)
        faces = OrderBDryFaces[i]
        a,i,j = faces[1]
        za = LorentzianSimplexSolver.DefineSymbols.make_symbol("zeta_$(a)$(i)$(j)$(:a)")
        zb = LorentzianSimplexSolver.DefineSymbols.make_symbol("zeta_$(a)$(i)$(j)$(:b)")

        a,i,j = faces[end]
        zaend = LorentzianSimplexSolver.DefineSymbols.make_symbol("zeta_$(a)$(i)$(j)$(:a)")
        zbend = LorentzianSimplexSolver.DefineSymbols.make_symbol("zeta_$(a)$(i)$(j)$(:b)")
        push!(xi_bdry, za, zb, zaend, zbend)
    end

    Y_vars = vcat(vec(eta_bdry), vec(xi_bdry))

    dlogEb_dXY_sym = [
        compute_dlogEb_dXY(logEb, X_vars, Y_vars)
        for logEb in logEb_list
    ]

    dlogEb_dX_sym  = [t[1] for t in dlogEb_dXY_sym]
    dlogEb_dY_sym  = [t[2] for t in dlogEb_dXY_sym]

    return dlogEb_dX_sym, dlogEb_dY_sym, Y_vars
end

function evaluate_dlogEb_dXY(dlogEb_dX_sym, dlogEb_dY_sym, Y_vars,
                               geom, sd, phase_soln;
                               gamma_value=nothing,
                               critical_values=nothing)
    T = solver_real_type(sd)
    critical_values = critical_values_for_gamma(sd, gamma_value, critical_values)
    evaluation_values = add_boundary_phase_values(critical_values, phase_soln)

    X_vars = vcat(vec(geom.varias[:g_var]), vec(geom.varias[:z_var]))
    
    nh = length(dlogEb_dX_sym)
    nx = length(X_vars)
    ny = length(Y_vars)
    
    dlogEb_dX_vals  = Matrix{Complex{T}}(undef, nh, nx)
    dlogEb_dY_vals  = Matrix{Complex{T}}(undef, nh, ny)
    
    for h in 1:nh
        for (i, v) in enumerate(X_vars)
            dlogEb_dX_vals[h, i] = evaluate_at_critical_point(
                dlogEb_dX_sym[h][v], evaluation_values, T,
            )
        end

        for (i, v) in enumerate(Y_vars)
            dlogEb_dY_vals[h, i] = evaluate_at_critical_point(
                dlogEb_dY_sym[h][v], evaluation_values, T,
            )
        end

    end

    return dlogEb_dX_vals, dlogEb_dY_vals
end

# ------------------------------------------------------------
# 3. log(k_b E_b)
# ------------------------------------------------------------
function logkbEb_func(
    gvariablesall,
    zvariablesall,
    ηlabelsMat,
    zetabdryall,
    kappaMat,
    OrderBDryFaces;
    gamma_symbol::Basic=LorentzianSimplexSolver.DefineAction.γsym(),
)

    kbEb_list = Vector{Basic}(undef, length(OrderBDryFaces))

    for idx in eachindex(OrderBDryFaces)
        faces = OrderBDryFaces[idx]

        xilist  = [zetabdryall[k][i][j] for (k,i,j) in faces]
        zlist   = [zvariablesall[k][i][j] for (k,i,j) in faces]
        glist   = [gvariablesall[k][i] for (k,i,j) in faces]
        κlist   = [kappaMat[k][i][j] for (k,i,j) in faces]

        ηvals = ηlabelsMat[faces[1][1]][faces[1][2]][faces[1][3]]

        product = internal_face_product(
            zlist,
            glist,
            κlist,
            2:2:(length(faces)-1);
            gamma_symbol=gamma_symbol,
        )

        term = LorentzianSimplexSolver.DefineAction.EebssSource(
                   glist[1], zlist[1], xilist[1], κlist[1]; γ=gamma_symbol,
               ) *
               product *
               LorentzianSimplexSolver.DefineAction.EebssTarget(
                   glist[end], zlist[end-1], xilist[end], κlist[end]; γ=gamma_symbol,
               )

        kbEb_list[idx] = ηvals * LorentzianSimplexSolver.DefineAction.slog(term)
    end

    return kbEb_list
end

function derivatives_of_log_kbEb(logkbEb::Basic, X_vars::Vector{Basic}, Y_vars::Vector{Basic})
    # A boundary face factor depends only on variables belonging to that face.
    # Derivatives with respect to all other variables vanish exactly.
    variables_in_log = Set(free_symbols(logkbEb))
    zero_symbol = Basic(0)
    dX = Dict(
        x => x in variables_in_log ? SymEngine.diff(logkbEb, x) : zero_symbol
        for x in X_vars
    )
    dY = Dict(
        y => y in variables_in_log ? SymEngine.diff(logkbEb, y) : zero_symbol
        for y in Y_vars
    )

    dXY = Dict{Tuple{Basic,Basic},Basic}()
    for x in X_vars
        variables_in_dX = Set(free_symbols(dX[x]))
        for y in Y_vars
            dXY[(x, y)] =
                y in variables_in_dX ? SymEngine.diff(dX[x], y) : zero_symbol
        end
    end

    dYY = Dict{Tuple{Basic,Basic},Basic}()
    for i in eachindex(Y_vars)
        y1 = Y_vars[i]
        variables_in_dY = Set(free_symbols(dY[y1]))
        for j in i:length(Y_vars)
            y2 = Y_vars[j]
            derivative =
                y2 in variables_in_dY ? SymEngine.diff(dY[y1], y2) : zero_symbol
            dYY[(y1, y2)] = derivative
            dYY[(y2, y1)] = derivative
        end
    end
    return dX, dY, dXY, dYY
end

function compute_kblogEb_dXY(geom, Y_vars)
    z_mat      = geom.varias[:z_mat]
    g_mat      = geom.varias[:g_mat]
    xi_mat    = geom.varias[:xi_mat]
    ηlabelsMat = geom.varias[:η_mat]
    kappa      = [geom.simplex[i].kappa for i in eachindex(geom.simplex)]
    OrderBDryFaces = geom.connectivity[1]["OrderBDryFaces"]

    gamma_symbol = LorentzianSimplexSolver.DefineAction.γsym()

    kbEb_list = logkbEb_func(
        g_mat,
        z_mat,
        ηlabelsMat,
        xi_mat,
        kappa,
        OrderBDryFaces;
        gamma_symbol=gamma_symbol,
    )

    # flatten variables
    X_vars = vcat(geom.varias[:g_var], geom.varias[:z_var])

    derivatives = [derivatives_of_log_kbEb(kbEb, X_vars, Y_vars) for kbEb in kbEb_list]
    dkbEb_dX_sym = [t[1] for t in derivatives]
    dkbEb_dY_sym = [t[2] for t in derivatives]
    d2kbEb_dXdY_sym = [t[3] for t in derivatives]
    d2kbEb_dYdY_sym = [t[4] for t in derivatives]

    return dkbEb_dX_sym, dkbEb_dY_sym, d2kbEb_dXdY_sym, d2kbEb_dYdY_sym
end

function evaluate_kblogEb_all(
    dkbEb_dX_sym,
    dkbEb_dY_sym,
    d2kbEb_dXdY_sym,
    d2kbEb_dYdY_sym,
    geom,
    sd, Y_vars, phase_soln;
    gamma_value=nothing,
    critical_values=nothing,
)
    T = solver_real_type(sd)
    critical_values = critical_values_for_gamma(sd, gamma_value, critical_values)
    evaluation_values = add_boundary_phase_values(critical_values, phase_soln)

    # --------------------------------------------------
    # Variables
    # --------------------------------------------------
    X_vars = vcat(vec(geom.varias[:g_var]), vec(geom.varias[:z_var]))

    nfaces = length(dkbEb_dX_sym)
    nx = length(X_vars)
    ny = length(Y_vars)

    # --------------------------------------------------
    # Allocate outputs
    # --------------------------------------------------
    dkbEb_dX_vals   = Matrix{Complex{T}}(undef, nfaces, nx)
    dkbEb_dY_vals   = Matrix{Complex{T}}(undef, nfaces, ny)
    d2kbEb_dXdY_vals = Array{Complex{T}}(undef, nfaces, nx, ny)
    d2kbEb_dYdY_vals = Array{Complex{T}}(undef, nfaces, ny, ny)

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------
    for h in 1:nfaces
        # first derivatives wrt X
        for (i, v) in enumerate(X_vars)
            dkbEb_dX_vals[h, i] = evaluate_at_critical_point(
                dkbEb_dX_sym[h][v], evaluation_values, T,
            )
        end

        # first derivatives wrt Y
        for (i, v) in enumerate(Y_vars)
            dkbEb_dY_vals[h, i] = evaluate_at_critical_point(
                dkbEb_dY_sym[h][v], evaluation_values, T,
            )
        end

        # second derivatives wrt X and Y
        for (ix, x) in enumerate(X_vars)
            for (iy, y) in enumerate(Y_vars)
                expr = d2kbEb_dXdY_sym[h][(x, y)]
                d2kbEb_dXdY_vals[h, ix, iy] = evaluate_at_critical_point(
                    expr, evaluation_values, T,
                )
            end
        end

        # second derivatives wrt Y and Y
        for iy1 in eachindex(Y_vars)
            y1 = Y_vars[iy1]
            for iy2 in iy1:length(Y_vars)
                y2 = Y_vars[iy2]
                expr = d2kbEb_dYdY_sym[h][(y1, y2)]
                value = evaluate_at_critical_point(
                    expr, evaluation_values, T,
                )
                d2kbEb_dYdY_vals[h, iy1, iy2] = value
                d2kbEb_dYdY_vals[h, iy2, iy1] = value
            end
        end
    end

    return dkbEb_dX_vals, dkbEb_dY_vals, d2kbEb_dXdY_vals, d2kbEb_dYdY_vals
end

solver_real_type(::LorentzianSimplexSolver.SolveVars.SolveData{T}) where {T<:Real} = T

to_solver_complex(value, ::Type{T}) where {T<:Real} =
    LorentzianSimplexSolver.ActionEvaluation.to_complex_T(value, T)

end # module
