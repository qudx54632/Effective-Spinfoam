module BoundaryAreaDerivatives

using LinearAlgebra
using SymEngine
using LorentzianSimplexSolver

export compute_dkb_dl

edge_key(edge) = (min(edge[1], edge[2]), max(edge[1], edge[2]))
as_basic(x) = Basic(string(x))

# Heron's formula in Cayley-Menger form, written in squared edge lengths.
function triangle_area_squared(l12sq, l13sq, l23sq)
    matrix = Basic[
        0 l12sq l13sq 1
        l12sq 0 l23sq 1
        l13sq l23sq 0 1
        1 1 1 0
    ]
    return -det(matrix) / 16
end

"""
Compute the first and second length derivatives of the boundary spins
`k_b = 2 A_b / gamma` for the selected boundary-edge perturbations.
"""
function compute_dkb_dl(
    geom,
    perturb_edges,
    boundary_edges_perturbed,
    vertices;
    gamma_symbol::Basic,
)
    connectivity = geom.connectivity[1]
    boundary_faces = connectivity["OrderBDryFaces"]
    triangles = [
        connectivity["TetFaces"][face[1][1]][face[1][2]][face[1][3]]
        for face in boundary_faces
    ]

    edge_positions = Dict(edge_key(edge) => i for (i, edge) in enumerate(perturb_edges))
    length_symbols = Dict(
        edge_key(edge) => Basic(Symbol("lb$(edge_key(edge)[1])_$(edge_key(edge)[2])"))
        for edge in boundary_edges_perturbed
    )

    for edge in keys(length_symbols)
        haskey(edge_positions, edge) || error("Perturbed boundary edge $edge is missing from perturb_edges.")
    end

    function edge_length_squared(i, j)
        key = edge_key((i, j))
        if haskey(length_symbols, key)
            return length_symbols[key]^2
        end
        return as_basic(LorentzianSimplexSolver.Volume.distance_sq(vertices[i], vertices[j]))
    end

    areas = [
        sqrt(triangle_area_squared(
            edge_length_squared(triangle[1], triangle[2]),
            edge_length_squared(triangle[1], triangle[3]),
            edge_length_squared(triangle[2], triangle[3]),
        ))
        for triangle in triangles
    ]

    length_values = Dict(
        symbol => as_basic(sqrt(LorentzianSimplexSolver.Volume.distance_sq(
            vertices[edge[1]],
            vertices[edge[2]],
        )))
        for (edge, symbol) in length_symbols
    )

    nfaces = length(triangles)
    nedges = length(perturb_edges)
    dkb_dl = fill(Basic(0), nfaces, nedges)
    d2kb_dldl = fill(Basic(0), nfaces, nedges, nedges)

    for h in eachindex(areas)
        first_derivatives = Dict(
            edge => SymEngine.diff(areas[h], symbol)
            for (edge, symbol) in length_symbols
        )

        for (edge_i, symbol_i) in length_symbols
            i = edge_positions[edge_i]
            dkb_dl[h, i] = subs(first_derivatives[edge_i], length_values) * 2 / gamma_symbol

            for (edge_j, symbol_j) in length_symbols
                j = edge_positions[edge_j]
                second_derivative = SymEngine.diff(first_derivatives[edge_i], symbol_j)
                d2kb_dldl[h, i, j] =
                    subs(second_derivative, length_values) * 2 / gamma_symbol
            end
        end
    end

    return dkb_dl, d2kb_dldl
end

end # module
