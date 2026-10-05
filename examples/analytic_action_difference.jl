using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")); io=devnull)

using EffectiveSpinfoam
using LorentzianSimplexSolver

ScalarT = Float64
configure_precision!(ScalarT)

simplices = [
    [1, 2, 3, 4, 6],
    [1, 2, 3, 5, 6],
    [1, 2, 4, 5, 6],
    [1, 2, 3, 4, 7],
    [1, 2, 3, 5, 7],
    [1, 2, 4, 5, 7],
]

flat_coordinates = [
    ScalarT.([0.0, 0.0, 0.0, 0.0]),
    ScalarT.([-0.068, -0.21988127663727278, -0.5316227766016838, -1.3316227766016839]),
    ScalarT.([0.0, 0.0, 0.0, -3.398088489694245]),
    ScalarT.([-0.24028114141347542, -0.6936319083813028, -0.9809436521275706, -1.6990442448471226]),
    ScalarT.([0.0, 0.0, -2.942830956382712, -1.6990442448471226]),
    ScalarT.([0.0, -2.7745276335252114, -0.9809436521275706, -1.6990442448471226]),
    ScalarT.([-2.4696884592430974, -3.893218630529324, -1.3565336794679874, -1.9090667752920147]),
]

boundary_edge = (1, 3)
# The order is all bulk edges followed by boundary_edge.
delta_lengths = ScalarT.([0.0, 1e-3])
gamma = ScalarT(1.0)
vertex_coordinates = Dict(i => flat_coordinates[i] for i in eachindex(flat_coordinates))

solver = build_solver_data(
    simplices,
    vertex_coordinates;
    bulk_sum_form=false,
    verbose=false,
    check=false,
)

data = build_effective_data(
    solver,
    simplices,
    vertex_coordinates;
    boundary_edges_perturbed=[sort(collect(boundary_edge))],
)

length(data.perturb_edges) == length(delta_lengths) ||
    error("Give one length perturbation for every edge in data.perturb_edges.")

terms = compute_effective_terms(data, gamma)
linear_difference = sum(terms.linear_regge .* delta_lengths)
quadratic_regge_difference = transpose(delta_lengths) *
    terms.quadratic_regge * delta_lengths
non_regge_difference = transpose(delta_lengths) *
    terms.non_regge_correction * delta_lengths
quadratic_difference = quadratic_regge_difference + non_regge_difference
analytic_action_difference = linear_difference + quadratic_difference

println("perturbation edges = $(data.perturb_edges)")
println("delta_lengths      = $(delta_lengths)")
println("linear difference  = $(linear_difference)")
println("quadratic Regge difference = $(quadratic_regge_difference)")
println("non-Regge difference = $(non_regge_difference)")
println("total quadratic difference = $(quadratic_difference)")
println("analytic action difference = $(analytic_action_difference)")
