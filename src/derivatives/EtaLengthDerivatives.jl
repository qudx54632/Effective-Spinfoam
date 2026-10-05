module EtaLengthDerivatives

export build_dηdl_matrix

using LorentzianSimplexSolver

# ------------------------------------------------------------
# 1. Geometry helpers
# ------------------------------------------------------------

function get_bulk_faces_vertices(geom_base)
    η_h_all = geom_base.connectivity[1]["OrderBulkFaces"]

    return [
        geom_base.connectivity[1]["TetFaces"][v[1][1]][v[1][2]][v[1][3]]
        for v in η_h_all
    ]
end


function get_bulk_edges(geom_base, η_h_vertices)
    # edges from bulk faces
    edges_ηh = [
        sort([v[i], v[j]])
        for v in η_h_vertices
        for (i, j) in ((1,2), (1,3), (2,3))
    ]
    unique_edges_ηh = unique(edges_ηh)

    # boundary faces
    η_b_all = geom_base.connectivity[1]["OrderBDryFaces"]
    η_b_vertices = [
        geom_base.connectivity[1]["TetFaces"][v[1][1]][v[1][2]][v[1][3]]
        for v in η_b_all
    ]

    edges_bdry = [
        sort([v[i], v[j]])
        for v in η_b_vertices
        for (i, j) in ((1,2), (1,3), (2,3))
    ]
    unique_edges_bdry = unique(edges_bdry)

    unique_edges_bulk = sort(setdiff(unique_edges_ηh, unique_edges_bdry))

    return unique_edges_bulk, unique_edges_bdry
end


# ------------------------------------------------------------
# 2. Combinatorial helper
# ------------------------------------------------------------

@inline function edge_position(face::Vector{Int}, edge::Vector{Int})
    a, b, c = face
    u, v = edge

    (u==a && v==b || u==b && v==a) && return 1
    (u==a && v==c || u==c && v==a) && return 2
    (u==b && v==c || u==c && v==b) && return 3
    return 0
end


# ------------------------------------------------------------
# 3. Area / derivative formulas
# ------------------------------------------------------------

@inline function Δ(l1, l2, l3)
    return -l1^4 - l2^4 - l3^4 +
           2*(l1^2*l2^2 + l1^2*l3^2 + l2^2*l3^2)
end


@inline function dηdl(l1, l2, l3, gamma_value)
    Δval = Δ(l1, l2, l3)
    pref = 1 / (gamma_value * sqrt(Δval))

    return pref .* [
        l1*(l2^2 + l3^2 - l1^2),
        l2*(l1^2 + l3^2 - l2^2),
        l3*(l1^2 + l2^2 - l3^2)
    ]
end


# ------------------------------------------------------------
# 4. Length extraction
# ------------------------------------------------------------

function compute_face_edge_lengths(j_h_vertices, vertex_coords)
    j_h_coords = [
        [vertex_coords[i], vertex_coords[j], vertex_coords[k]]
        for (i, j, k) in map(Tuple, j_h_vertices)
    ]

    return [
        sqrt.([
            LorentzianSimplexSolver.Dihedral.minkowski_norm2(coords[1] - coords[2]),
            LorentzianSimplexSolver.Dihedral.minkowski_norm2(coords[1] - coords[3]),
            LorentzianSimplexSolver.Dihedral.minkowski_norm2(coords[2] - coords[3])
        ])
        for coords in j_h_coords
    ]
end


# ------------------------------------------------------------
# 5. Main driver
# ------------------------------------------------------------

function build_dηdl_matrix(
    η_h_vertices, perturb_edges, 
    vertex_coords,
    ::Type{T},
    gamma_value,
) where {T<:Real}
    nh = length(η_h_vertices)
    nb = length(perturb_edges)

    face_lengths = compute_face_edge_lengths(η_h_vertices, vertex_coords)
    dηdl_matrix = zeros(T, nh, nb)

    for h in 1:nh
        # dη_h/dl has three components, one for each edge of h.  Compute
        # the triangle formula once and place only the perturbed components.
        derivatives = dηdl(face_lengths[h]..., gamma_value)
        for b in 1:nb
            pos = edge_position(η_h_vertices[h], perturb_edges[b])
            pos == 0 && continue
            dηdl_matrix[h, b] = derivatives[pos]
        end
    end

    return dηdl_matrix
end

end # module
