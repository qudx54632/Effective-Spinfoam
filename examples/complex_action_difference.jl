using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")); io=devnull)

using EffectiveSpinfoam
using LorentzianSimplexSolver

# Input
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
# Order: all bulk edges, followed by boundary_edge.
# Here the order is [(1, 2), (1, 3)].
delta_lengths = ScalarT.([0.0, 1e-3])
gamma = ScalarT(1.0)

# Complex critical point with the perturbed lengths held fixed
data = prepare_action_difference(
    simplices,
    flat_coordinates,
    boundary_edge,
    delta_lengths;
    bulk_sum_form=true,
    gamma=gamma,
    verbose=false,
    check=false,
)

result = solve_action_difference(
    data,
    gamma;
    tol=ScalarT(5e-14),
    maxiter=30,
    verbose=true,
)

println("flat action       = $(result.flat_action)")
println("complex action    = $(result.complex_action)")
println("action difference = $(result.action_difference)")
println("max |dS|          = $(maximum(abs.(result.residual)))")
