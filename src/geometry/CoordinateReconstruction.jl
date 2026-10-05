function squared_length(a, b, coordinates)
    vector = coordinates[a] - coordinates[b]
    return LorentzianSimplexSolver.Dihedral.minkowski_norm2(vector)
end

function target_squared_length(a, b, coordinates, length_changes)
    a == b && return zero(eltype(coordinates[a]))

    edge = minmax(a, b)
    old_square = squared_length(a, b, coordinates)
    delta_l = get(length_changes, edge, zero(old_square))
    new_length = sqrt(abs(old_square)) + delta_l
    return sign(old_square) * new_length^2
end

function target_gram_matrix(simplex, coordinates, length_changes)
    T = eltype(coordinates[simplex[1]])
    gram = zeros(T, 4, 4)
    base = simplex[1]

    for i in 1:4, j in 1:4
        vi = simplex[i + 1]
        vj = simplex[j + 1]
        gram[i, j] = (
            target_squared_length(base, vi, coordinates, length_changes) +
            target_squared_length(base, vj, coordinates, length_changes) -
            target_squared_length(vi, vj, coordinates, length_changes)
        ) / 2
    end

    return gram
end


function reconstruct_simplex_coordinates(simplex, coordinates, length_changes)
    base_point = coordinates[simplex[1]]
    old_edge_vectors = hcat([
        coordinates[vertex] - base_point
        for vertex in simplex[2:5]
    ]...)

    T = eltype(base_point)
    metric = LorentzianSimplexSolver.SpinAlgebra.eta(T)
    old_gram = transpose(old_edge_vectors) * metric * old_edge_vectors
    new_gram = target_gram_matrix(simplex, coordinates, length_changes)

    # The general matrix square root can acquire roundoff-level imaginary parts.
    deformation = real.(sqrt(old_gram \ new_gram))
    new_edge_vectors = old_edge_vectors * deformation

    return vcat(
        [copy(base_point)],
        [base_point + new_edge_vectors[:, i] for i in 1:4],
    )
end


function reconstruct_local_coordinates(
    simplices,
    coordinates,
    perturb_edges,
    delta_lengths,
)
    length_changes = Dict(
        minmax(edge[1], edge[2]) => delta_l
        for (edge, delta_l) in zip(perturb_edges, delta_lengths)
    )

    return [
        reconstruct_simplex_coordinates(simplex, coordinates, length_changes)
        for simplex in simplices
    ]
end
