"""
Compute the boundary Taylor pieces used in the Regge-sector comparison.

The returned object contains the linear and quadratic coefficients of
`sum_b k_b log(E_b)` under Regge-like boundary-length perturbations, together
with the Gaussian source terms needed for the term-by-term checks.
"""
function compute_boundary_taylor_terms(data, gamma_value; blocks=nothing, tol=nothing)
    blocks = blocks === nothing ? evaluate_effective_blocks(data, gamma_value) : blocks
    (; T, critical_values, h_frak, inv_h_frak, dηdl, dlogEh_dX, dkb_dl, d2kb_dldl) = blocks
    tol_value = tol === nothing ? T(LorentzianSimplexSolver.PrecisionUtils.get_tolerance()) : T(tol)

    bdy = data.boundary_symbols
    Y_vars = bdy.Y_variables

    # Boundary derivatives evaluated at the critical configuration.
    dlogEb_dX_vals, dlogEb_dY_vals = LogFaceDerivatives.evaluate_dlogEb_dXY(
        bdy.dlogEb_dX,
        bdy.dlogEb_dY,
        Y_vars,
        data.geom,
        data.log_spinfoam.solve_data,
        data.log_spinfoam.phase_solution;
        gamma_value=gamma_value,
        critical_values=critical_values,
    )

    _, dkbEb_dY_vals, d2kbEb_dXdY_vals, d2kbEb_dYdY_vals =
        LogFaceDerivatives.evaluate_kblogEb_all(
            bdy.dkbEb_dX,
            bdy.dkbEb_dY,
            bdy.d2kbEb_dXdY,
            bdy.d2kbEb_dYdY,
            data.geom,
            data.log_spinfoam.solve_data,
            Y_vars,
            data.log_spinfoam.phase_solution;
            gamma_value=gamma_value,
            critical_values=critical_values,
        )

    nb = size(dlogEb_dY_vals, 1)
    nl = length(data.perturb_edges)

    d2kbEb_dXdY_vals_sumb = sum(d2kbEb_dXdY_vals[i, :, :] for i in 1:nb)
    d2kbEb_dYdY_vals_sumb = sum(d2kbEb_dYdY_vals[i, :, :] for i in 1:nb)
    dkbEb_dY_vals_sumb = sum(dkbEb_dY_vals[i, :] for i in 1:nb)

    # Gaussian source from the bulk area variation.
    B = transpose(transpose(dηdl) * dlogEh_dX)

    # First boundary response.
    # Linear system for the boundary spinor/phase response Dℓξ.
    # It comes from linearizing
    #     log E_b(X_c(ℓ), ξ(ℓ)) = (iγ / 2) Θ_b(ℓ)
    # and eliminating DℓX with the linearized X-EOM,
    #     DℓX = -𝔥^{-1}(B + C).
    # The second block imposes the compatible bulk-face response Dℓε_h.
    boundary_system = vcat(
        dlogEb_dY_vals[:, nb+1:end] -
            dlogEb_dX_vals * inv_h_frak * d2kbEb_dXdY_vals_sumb[:, nb+1:end],
        -dlogEh_dX * inv_h_frak * d2kbEb_dXdY_vals_sumb[:, nb+1:end],
    )

    boundary_rhs = vcat(
        im * gamma_value / 2 * data.DΘDl +
            dlogEb_dX_vals * inv_h_frak *
            (B + d2kbEb_dXdY_vals_sumb[:, 1:nb] * dkb_dl),
        im * gamma_value / 2 * data.DεDl +
            dlogEh_dX * inv_h_frak *
            (B + d2kbEb_dXdY_vals_sumb[:, 1:nb] * dkb_dl),
    )

    DξDl = pinv(boundary_system, atol=tol_value, rtol=tol_value) * boundary_rhs
    DYDl = vcat(dkb_dl, DξDl)
    C = d2kbEb_dXdY_vals_sumb * DYDl
    DXDl = -inv_h_frak * (B + C)

    # Second boundary response.
    kb_vals = [
        real(to_solver_complex(subs(Y_vars[i], critical_values), T))
        for i in 1:nb
    ]

    Mm = [
        kb_vals[b] * dlogEb_dY_vals[b, j]
        for b in 1:nb, j in nb+1:length(Y_vars)
    ]

    R = [
        dkb_dl[i, j] * transpose(dlogEb_dY_vals[i, nb+1:end]) * DξDl[:, j] +
        transpose(DXDl[:, j]) * d2kbEb_dXdY_vals[i, :, nb+1:end] * DξDl[:, j] +
        transpose(DξDl[:, j]) * d2kbEb_dYdY_vals[i, nb+1:end, nb+1:end] * DξDl[:, j]
        for i in 1:nb, j in 1:nl
    ]

    D2ξDl2 = -pinv(Mm, atol=tol_value, rtol=tol_value) * R[:, end]
    nxi = length(Y_vars) - nb
    D2ξDl2_matrix = [
        j == k == nl ? D2ξDl2[b] : 0
        for b in 1:nxi, j in 1:nl, k in 1:nl
    ]
    D2YDl2_matrix = vcat(d2kb_dldl, D2ξDl2_matrix)

    # Taylor expansion of sum_b k_b log(E_b).
    I_bdy_linear = transpose(dkbEb_dY_vals_sumb) * DYDl
    I_bdy_quadratic =
        1 / 2 * transpose(DYDl) * d2kbEb_dYdY_vals_sumb * DYDl +
        1 / 2 * sum(
            dkbEb_dY_vals_sumb[i] * D2YDl2_matrix[i, :, :]
            for i in 1:length(Y_vars)
        )

    return (
        h_frak=h_frak,
        inv_h_frak=inv_h_frak,
        dηdl=dηdl,
        B=B,
        C=C,
        dkb_dl=dkb_dl,
        d2kb_dldl=d2kb_dldl,
        I_bdy_linear=I_bdy_linear,
        I_bdy_quadratic=I_bdy_quadratic,
    )
end
