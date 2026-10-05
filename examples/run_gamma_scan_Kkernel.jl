# Scan gamma values and plot the hinge-space correction kernel K.

using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")); io=devnull)

using LinearAlgebra
using GenericLinearAlgebra
using EffectiveSpinfoam
using LorentzianSimplexSolver
using Plots
using LaTeXStrings
using Polynomials

ScalarT = Float64
LorentzianSimplexSolver.configure_precision!(ScalarT; precision=64)

gamma_values = ScalarT(10.0) .^ range(ScalarT(-6), ScalarT(0), length=50)
figure_file = normpath(joinpath(@__DIR__, "..", "results", "gamma_scan_15move_refinement.png"))
figure_title = L"\mathrm{1-5\,\,move\,\,refinement}"
mkpath(dirname(figure_file))

function plot_gamma_scan(
    figure_file,
    gamma_values,
    scaled_norms,
    figure_title;
    nfit=20,
)
    nfit = min(nfit, length(gamma_values))
    power_law = fit(
        log10.(gamma_values[1:nfit]),
        log10.(scaled_norms[1:nfit]),
        1,
    )
    intercept, slope = coeffs(power_law)
    fit_curve = 10 .^ (intercept .+ slope .* log10.(gamma_values))

    figure = plot(
        gamma_values,
        scaled_norms;
        marker=:circle,
        xscale=:log10,
        yscale=:log10,
        xlabel=L"\gamma",
        ylabel=L"\gamma^2\|K\|",
        title=figure_title,
        label="numerical result",
        legend=:bottomright,
    )
    plot!(
        figure,
        gamma_values,
        fit_curve;
        lw=2,
        linestyle=:dash,
        label="power-law fit (slope = $(round(slope, digits=6)))",
    )
    savefig(figure, figure_file)
    return slope
end

simplices = [[2,3,4,6,7],[1,3,4,6,7],[1,2,4,6,7],[1,2,3,6,7],[1,2,3,4,7],[2,3,5,6,8],[1,3,5,6,8],[1,2,5,6,8],[1,2,3,6,8],[1,2,3,5,8],[2,4,5,6,9],[1,4,5,6,9],[1,2,5,6,9],[1,2,4,6,9],[1,2,4,5,9],[3,4,5,6,10],[1,4,5,6,10],[1,3,5,6,10],[1,3,4,6,10],[1,3,4,5,10],[3,4,5,6,11],[2,4,5,6,11],[2,3,5,6,11],[2,3,4,6,11],[2,3,4,5,11]]

coords_lines = [
"0, 0, 0, 0",
"0, -2.7745276335252114, -0.9809436521275706, -1.6990442448471226",
"0, 0, 0, -3.398088489694245",
"-0.24028114141347542, -0.6936319083813028, -0.9809436521275706, -1.6990442448471226",
"0, 0, -2.942830956382712, -1.6990442448471226",
"-0.068, -0.27, -0.5, -1.3",
"-0.06165622828269508, -0.7476319083813029, -0.49237746085102824, -1.6192353958776982",
"-0.013600000000000001, -0.6089055267050423, -0.8847549217020566, -1.6192353958776982",
"-0.05471634704156723, -0.6634799624836852, -1.2905133626305172, -1.3266576479993824",
"-0.05996263164968173, -0.18743250119993968, -0.8604521717798855, -1.5747576871100133",
"-0.0423034122998675, -0.5046526423930288, -1.0289336294868097, -2.25080216236223"
]

vertex_coords = build_vertex_coordinates(
    simplices,
    coords_lines,
    ScalarT,
)

println("[1/3] Building gamma-independent effective setup.")
solver = build_solver_data(
    simplices,
    vertex_coords;
    bulk_sum_form=false,
)
data = build_effective_data(
    solver,
    simplices,
    vertex_coords,
)

println("[2/3] Scanning $(length(gamma_values)) gamma values.")
K_kernel_norm_list = Vector{ScalarT}(undef, length(gamma_values))
scaled_K_kernel_norm_list = Vector{ScalarT}(undef, length(gamma_values))

for (i, gamma_value) in enumerate(gamma_values)
    println("[$i/$(length(gamma_values))] gamma = $gamma_value")
    result = compute_effective_terms(data, gamma_value)

    K_kernel_norm_list[i] = opnorm(result.K_kernel)
    scaled_K_kernel_norm_list[i] = gamma_value^2 * K_kernel_norm_list[i]

    println("    ||K|| = $(K_kernel_norm_list[i])")
    println("    gamma^2 ||K|| = $(scaled_K_kernel_norm_list[i])")
end

println("[3/3] Plotting gamma scan.")
slope = plot_gamma_scan(
    figure_file,
    gamma_values,
    scaled_K_kernel_norm_list,
    figure_title,
)
println("Saved plot to $figure_file.")
println("Small-gamma fitted slope = $(round(slope, digits=6)).")
