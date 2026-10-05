"""
Evaluate the matrices entering the non-Regge correction at one Immirzi parameter.
The expensive geometry and symbolic derivatives in `data` are reused.
"""
function compute_effective_matrices(data, gamma_value; blocks=nothing)
    blocks = blocks === nothing ? evaluate_effective_blocks(data, gamma_value) : blocks
    (; T, critical_values, log_hessian, h_frak, inv_h_frak, dηdl, dlogEh_dX) = blocks
    e_hat = TransverseBasis.compute_transverse_basis(dηdl)
    V = transpose(dlogEh_dX)

    eta_values = [
        real(to_solver_complex(subs(eta, critical_values), T))
        for eta in data.eta_symbols
    ]
    A = Diagonal(eta_values)
    invA = inv(A)

    H = h_frak + V * A * transpose(V)
    D = transpose(e_hat) * transpose(V)
    G = -D * inv(H) * transpose(D)
    w = -transpose(V) * inv_h_frak * V
    rho = inv(invA - w)
    kappa = e_hat * inv(G) * transpose(e_hat)
    K_kernel = rho - rho * invA * kappa * invA * rho
    calH = [H transpose(D); D zeros(eltype(H), size(D, 1), size(D, 1))]

    return (
        K_kernel=K_kernel,
        A=A,
        H=H,
        D=D,
        G=G,
        calH=calH,
        V=V,
        w=w,
        rho=rho,
        kappa=kappa,
        log_hessian=log_hessian,
        h_frak=h_frak,
        inv_h_frak=inv_h_frak,
        dηdl=dηdl,
        dkb_dl=blocks.dkb_dl,
        d2kb_dldl=blocks.d2kb_dldl,
        transverse_basis=e_hat,
        dlogEh_dX=dlogEh_dX,
        critical_values=critical_values,
    )
end

function compute_effective_terms(data, gamma_value)
    matrices = compute_effective_matrices(data, gamma_value)

    log_action = evaluate_log_action(
        data.solver,
        matrices.critical_values,
    )

    linear_regge =
        im / 2 * gamma_value * transpose(matrices.dkb_dl) * data.regge.dihedral_angles

    quadratic_regge = im * (
        gamma_value / 4 * transpose(matrices.dηdl) * data.DεDl +
        gamma_value / 4 * (
            transpose(matrices.dkb_dl) * data.DΘDl +
            sum(
                data.regge.dihedral_angles[i] * matrices.d2kb_dldl[i, :, :]
                for i in eachindex(data.regge.dihedral_angles)
            )
        )
    )

    non_regge_correction =
        -gamma_value^2 / 8 * transpose(data.DεDl) * matrices.K_kernel * data.DεDl

    return (
        linear_regge=linear_regge,
        quadratic_regge=quadratic_regge,
        non_regge_correction=non_regge_correction,
        spinfoam_quadratic=quadratic_regge + non_regge_correction,
        log_action=log_action,
        K_kernel=matrices.K_kernel,
        rho=matrices.rho,
        kappa=matrices.kappa,
        log_hessian=matrices.log_hessian,
        A=matrices.A,
        H=matrices.H,
        D=matrices.D,
        G=matrices.G,
        calH=matrices.calH,
        V=matrices.V,
        w=matrices.w,
        h_frak=matrices.h_frak,
        inv_h_frak=matrices.inv_h_frak,
        dηdl=matrices.dηdl,
        dkb_dl=matrices.dkb_dl,
        d2kb_dldl=matrices.d2kb_dldl,
        transverse_basis=matrices.transverse_basis,
        dlogEh_dX=matrices.dlogEh_dX,
    )
end

function scan_K_kernel(data, gamma_values)
    return [
        compute_effective_matrices(data, gamma_value).K_kernel
        for gamma_value in gamma_values
    ]
end

function print_effective_terms(result; io::IO=stdout, show_matrices::Bool=false)
    print_term(io, "iS_Regge^(1): linear Regge term", result.linear_regge; show_matrix=show_matrices)
    print_term(io, "iS_Regge^(2): quadratic Regge term", result.quadratic_regge; show_matrix=show_matrices)
    print_term(io, "Delta_non-Regge^(2): non-Regge correction term", result.non_regge_correction; show_matrix=show_matrices)
    return nothing
end

function print_term(io::IO, title::AbstractString, value; show_matrix::Bool=false)
    println(io)
    println(io, title)
    show(io, "text/plain", value)
    println(io)
    return nothing
end
