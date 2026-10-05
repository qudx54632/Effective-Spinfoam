module DihedralAngleDerivatives

using Symbolics
using Combinatorics
using LinearAlgebra
using LorentzianSimplexSolver

export compute_dθdl

edge_key(edge) = (min(edge[1], edge[2]), max(edge[1], edge[2]))

# Squared Cayley-Menger volumes for a 4-simplex, tetrahedron and triangle.
function V4_squared(edges)
    l12, l13, l14, l15, l23, l24, l25, l34, l35, l45 = edges
    return -det([
        0 l12 l13 l14 l15 1
        l12 0 l23 l24 l25 1
        l13 l23 0 l34 l35 1
        l14 l24 l34 0 l45 1
        l15 l25 l35 l45 0 1
        1 1 1 1 1 0
    ]) / (2^4 * factorial(4)^2)
end

function V3_squared(edges)
    l12, l13, l14, l23, l24, l34 = edges
    return det([
        0 l12 l13 l14 1
        l12 0 l23 l24 1
        l13 l23 0 l34 1
        l14 l24 l34 0 1
        1 1 1 1 0
    ]) / (2^3 * factorial(3)^2)
end

function V2_squared(edges)
    l12, l13, l23 = edges
    return -det([
        0 l12 l13 1
        l12 0 l23 1
        l13 l23 0 1
        1 1 1 0
    ]) / (2^2 * factorial(2)^2)
end

# The logarithm of this z gives the Lorentzian dihedral angle.  The final
# derivative is therefore -d(log z)/dl = -(dz/dl)/z.
function dihedral_z(simplex_edges, tet_a_edges, tet_b_edges, face_edges, opposite_edge)
    Vσ = V4_squared(simplex_edges)
    Vτa = V3_squared(tet_a_edges)
    Vτb = V3_squared(tet_b_edges)
    Vh = V2_squared(face_edges)

    @variables x
    varied_edges = copy(simplex_edges)
    varied_edges[opposite_edge] = x
    Vτab = substitute(
        Symbolics.derivative(V4_squared(varied_edges), x),
        x => simplex_edges[opposite_edge],
    )

    term1 = 4^2 * Vτab / Vh
    term2 = Vτa / Vh
    term3 = Vτb / Vh
    return (term1 + sqrt(term1^2 - term2 * term3)) /
           (sqrt(term2) * sqrt(term3))
end

function face_data(simplex, face, tets_in_simplex)
    tets = [tet for tet in tets_in_simplex if all(v -> v in tet, face)]
    length(tets) == 2 || error("Face $face must belong to two tetrahedra in simplex $simplex.")

    simplex_edges = [collect(edge) for edge in combinations(simplex, 2)]
    face_edges = [collect(edge) for edge in combinations(face, 2)]
    tetrahedron_edges = [
        [collect(edge) for edge in combinations(tet, 2)]
        for tet in tets
    ]

    opposite_edge = edge_key(setdiff(simplex, face))
    opposite_index = findfirst(edge -> edge_key(edge) == opposite_edge, simplex_edges)

    return (; simplex_edges, face_edges, tetrahedron_edges, opposite_index)
end

"""
Compute derivatives of deficit or boundary dihedral angles with respect to
the selected edge lengths. Each row corresponds to one triangular face.
"""
function compute_dθdl(
    simplices,
    face_vertices,
    perturb_edges,
    vertex_coords,
    tets_per_simplex,
    ::Type{T},
) where {T<:Real}
    nfaces = length(face_vertices)
    nedges = length(perturb_edges)
    derivatives = zeros(Complex{T}, nfaces, nedges)

    @variables lsq[1:nedges]
    length_symbols = collect(lsq)
    perturb_positions = Dict(edge_key(edge) => i for (i, edge) in enumerate(perturb_edges))
    numeric_lengths = Dict{Tuple{Int,Int},T}()

    function numeric_length_squared(edge)
        key = edge_key(edge)
        return get!(numeric_lengths, key) do
            LorentzianSimplexSolver.Dihedral.minkowski_norm2(
                vertex_coords[key[1]] - vertex_coords[key[2]],
            )
        end
    end

    function symbolic_or_numeric_length(edge)
        position = get(perturb_positions, edge_key(edge), 0)
        return position == 0 ? numeric_length_squared(edge) : length_symbols[position]
    end

    length_values = Dict(
        length_symbols[i] => numeric_length_squared(edge)
        for (i, edge) in enumerate(perturb_edges)
    )
    numeric_length_values = [length_values[symbol] for symbol in length_symbols]

    for h in eachindex(face_vertices), simplex_index in eachindex(simplices)
        simplex = simplices[simplex_index]
        face = face_vertices[h]
        all(v -> v in simplex, face) || continue

        data = face_data(simplex, face, tets_per_simplex[simplex_index])
        varied_edges = [
            get(perturb_positions, edge_key(edge), 0)
            for edge in data.simplex_edges
        ]
        filter!(i -> i != 0, varied_edges)
        isempty(varied_edges) && continue

        simplex_lengths = symbolic_or_numeric_length.(data.simplex_edges)
        tet_a_lengths = symbolic_or_numeric_length.(data.tetrahedron_edges[1])
        tet_b_lengths = symbolic_or_numeric_length.(data.tetrahedron_edges[2])
        face_lengths = symbolic_or_numeric_length.(data.face_edges)

        z = dihedral_z(
            simplex_lengths,
            tet_a_lengths,
            tet_b_lengths,
            face_lengths,
            data.opposite_index,
        )
        z_function = Symbolics.build_function(
            z,
            length_symbols;
            expression=Val(false),
        )
        z_value = Complex{T}(z_function(numeric_length_values))

        for b in unique(varied_edges)
            dz_dlsq = Symbolics.derivative(z, length_symbols[b])
            derivative_function = Symbolics.build_function(
                dz_dlsq,
                length_symbols;
                expression=Val(false),
            )
            dz_dlsq_value = Complex{T}(derivative_function(numeric_length_values))
            length_value = sqrt(length_values[length_symbols[b]])
            derivatives[h, b] -= 2 * length_value * dz_dlsq_value / z_value
        end
    end

    return derivatives
end

end # module
