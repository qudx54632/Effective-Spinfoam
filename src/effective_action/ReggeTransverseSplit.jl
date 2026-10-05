function prepare_regge_transverse_split(
    flat_data,
    simplices,
    flat_coordinates,
    boundary_edge,
    gamma,
)
    coordinates = flat_coordinate_dict(simplices, flat_coordinates)
    bulk_faces = EtaLengthDerivatives.get_bulk_faces_vertices(flat_data.geometry)
    bulk_edges, boundary_edges = EtaLengthDerivatives.get_bulk_edges(flat_data.geometry, bulk_faces)
    selected_boundary_edges = [sort(collect(boundary_edge))]
    perturb_edges = vcat(bulk_edges, selected_boundary_edges)
    T = eltype(first(values(coordinates)))

    dηdl = EtaLengthDerivatives.build_dηdl_matrix(
        bulk_faces,
        perturb_edges,
        coordinates,
        T,
        gamma,
    )
    transverse_basis = TransverseBasis.compute_transverse_basis(dηdl)

    return (
        bulk_faces=bulk_faces,
        bulk_edges=bulk_edges,
        boundary_edges=boundary_edges,
        selected_boundary_edges=selected_boundary_edges,
        perturb_edges=perturb_edges,
        dηdl=dηdl,
        transverse_basis=transverse_basis,
    )
end
