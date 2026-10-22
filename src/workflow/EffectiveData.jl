function build_effective_data(
    solver,
    simplices,
    vertex_coords;
    boundary_edges_perturbed=nothing,
    verbose::Bool=false,
)
    geom = solver.geom
    log_spinfoam = solver.log_spinfoam

    dlogEh_dX_symbols = LogFaceDerivatives.compute_dlogEh_dX(geom)
    dlogEb_dX_symbols, dlogEb_dY_symbols, boundary_Y_variables =
        LogFaceDerivatives.compute_dlogEb_dXY(geom)
    dkbEb_dX_symbols, dkbEb_dY_symbols, d2kbEb_dXdY_symbols, d2kbEb_dYdY_symbols =
        LogFaceDerivatives.compute_kblogEb_dXY(geom, boundary_Y_variables)

    eta_h_vertices = EtaLengthDerivatives.get_bulk_faces_vertices(geom)
    bulk_edges, boundary_edges = EtaLengthDerivatives.get_bulk_edges(geom, eta_h_vertices)
    selected_boundary_edges =
        boundary_edges_perturbed === nothing ? [boundary_edges[1]] : boundary_edges_perturbed
    perturb_edges = vcat(bulk_edges, selected_boundary_edges)

    boundary_face_vertices = [
        geom.connectivity[1]["TetFaces"][face[1][1]][face[1][2]][face[1][3]]
        for face in geom.connectivity[1]["OrderBDryFaces"]
    ]

    T = eltype(first(values(vertex_coords)))
    angle_derivatives = DihedralAngleDerivatives.compute_dθdl(
        simplices,
        vcat(eta_h_vertices, boundary_face_vertices),
        perturb_edges,
        vertex_coords,
        geom.connectivity[1]["Tets"],
        T,
    )
    nh = length(eta_h_vertices)
    DεDl = angle_derivatives[1:nh, :]
    DΘDl = angle_derivatives[(nh + 1):end, :]

    dkb_dl, d2kb_dldl = BoundaryAreaDerivatives.compute_dkb_dl(
        geom,
        perturb_edges,
        selected_boundary_edges,
        vertex_coords;
        gamma_symbol=log_spinfoam.gamma_symbol,
    )

    return (
        solver=solver,
        geom=geom,
        vertex_coords=vertex_coords,
        regge=solver.regge,
        log_spinfoam=log_spinfoam,
        dlogEh_dX_symbols=dlogEh_dX_symbols,
        boundary_symbols=(
            Y_variables=boundary_Y_variables,
            dlogEb_dX=dlogEb_dX_symbols,
            dlogEb_dY=dlogEb_dY_symbols,
            dkbEb_dX=dkbEb_dX_symbols,
            dkbEb_dY=dkbEb_dY_symbols,
            d2kbEb_dXdY=d2kbEb_dXdY_symbols,
            d2kbEb_dYdY=d2kbEb_dYdY_symbols,
        ),
        eta_symbols=solver.eta_variables,
        eta_h_vertices=eta_h_vertices,
        bulk_edges=bulk_edges,
        boundary_edges=boundary_edges,
        boundary_edges_perturbed=selected_boundary_edges,
        perturb_edges=perturb_edges,
        DεDl=DεDl,
        DΘDl=DΘDl,
        dkb_dl=dkb_dl,
        d2kb_dldl=d2kb_dldl,
        X_variables=solver.X_variables,
        all_variables=solver.all_variables,
    )
end

function _evaluate_boundary_area_derivatives(data, critical_values, ::Type{T}) where {T<:Real}
    dkb = map(
        x -> to_solver_complex(subs(x, critical_values), T),
        data.dkb_dl,
    )
    d2kb = map(
        x -> to_solver_complex(subs(x, critical_values), T),
        data.d2kb_dldl,
    )
    return dkb, d2kb
end

"""
Evaluate the gamma-dependent blocks shared by the effective-matrix and
boundary-Taylor workflows.
"""
function evaluate_effective_blocks(data, gamma_value)
    T = solver_real_type(data.solver)
    validate_solver_number(gamma_value, T, "gamma_value")

    critical_values = critical_value_dict(
        data.solver;
        gamma=gamma_value,
    )
    log_hessian = evaluate_log_hessian(
        data.solver;
        gamma=gamma_value,
    )

    nX = length(data.X_variables)
    h_frak = log_hessian[1:nX, 1:nX]
    dηdl = EtaLengthDerivatives.build_dηdl_matrix(
        data.eta_h_vertices,
        data.perturb_edges,
        data.vertex_coords,
        T,
        gamma_value,
    )
    dηdl_bulk = dηdl[:, 1:length(data.bulk_edges)]
    dlogEh_dX = LogFaceDerivatives.evaluate_dlogEh_dX(
        data.dlogEh_dX_symbols,
        data.geom,
        data.log_spinfoam.solve_data;
        gamma_value=gamma_value,
        critical_values=critical_values,
    )
    dkb_dl, d2kb_dldl = _evaluate_boundary_area_derivatives(data, critical_values, T)

    return (
        T=T,
        critical_values=critical_values,
        log_hessian=log_hessian,
        h_frak=h_frak,
        inv_h_frak=inv(h_frak),
        dηdl=dηdl,
        dηdl_bulk=dηdl_bulk,
        dlogEh_dX=dlogEh_dX,
        dkb_dl=dkb_dl,
        d2kb_dldl=d2kb_dldl,
    )
end
