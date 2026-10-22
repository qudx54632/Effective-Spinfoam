# Scan gamma values and plot the hypercube hinge-space correction kernel K.

using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")); io=devnull)

using LinearAlgebra
using GenericLinearAlgebra
using EffectiveSpinfoam
using LorentzianSimplexSolver
using Plots
using LaTeXStrings
using Polynomials
using Printf

ScalarT = Float64
LorentzianSimplexSolver.configure_precision!(ScalarT; precision=64)

gamma_values = ScalarT(10.0) .^ range(ScalarT(-6), ScalarT(0), length=50)
result_directory = normpath(joinpath(@__DIR__, "..", "results"))
figure_file = joinpath(result_directory, "gamma_scan_hypercube.pdf")
data_file = joinpath(result_directory, "gamma_scan_hypercube.txt")
figure_title = "Hypercube"
mkpath(result_directory)

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

    # Match the action-difference scan plots.
    show_title = true
    legend_position = :bottomright

    figure = plot(
        gamma_values,
        scaled_norms;
        color=:blue,
        marker=:circle,
        markersize=5,
        linewidth=1.6,
        xscale=:log10,
        yscale=:log10,
        xlabel=L"\gamma",
        ylabel=L"\gamma^2\|K\|",
        title=show_title ? figure_title : "",
        label="numerical result",
        legend=legend_position,
        size=(600, 400),
        left_margin=2 * Plots.PlotMeasures.mm,
        right_margin=4 * Plots.PlotMeasures.mm,
        bottom_margin=2 * Plots.PlotMeasures.mm,
        top_margin=4 * Plots.PlotMeasures.mm,
        titlefontsize=16,
        guidefontsize=15,
        tickfontsize=12,
        legendfontsize=9,
        legend_columns=1,
        legend_background_color=:white,
        legend_foreground_color=:gray30,
        grid=true,
        gridalpha=0.18,
    )
    plot!(
        figure,
        gamma_values,
        fit_curve;
        color=:orange,
        linewidth=1.5,
        linestyle=:dash,
        label=@sprintf("power-law fit (p = %.4f)", slope),
    )
    savefig(figure, figure_file)
    display(figure)
    return slope
end

simplices = [[1, 9, 13, 15, 16],[1, 9, 13, 14, 16],[1, 9, 11, 15, 16],[1, 9, 11, 12, 16],[1, 9, 10, 14, 16],[1, 9, 10, 12, 16],[1, 5, 13, 15, 16],[1, 5, 13, 14, 16],[1, 5, 7, 15, 16],[1, 5, 7, 8, 16],[1, 5, 6, 14, 16],[1, 5, 6, 8, 16],[1, 3, 11, 15, 16],[1, 3, 11, 12, 16],[1, 3, 7, 15, 16],[1, 3, 7, 8, 16],[1, 3, 4, 12, 16],[1, 3, 4, 8, 16],[1, 2, 10, 14, 16],[1, 2, 10, 12, 16],[1, 2, 6, 14, 16],[1, 2, 6, 8, 16],[1, 2, 4, 12, 16],[1, 2, 4, 8, 16]]

coords_lines = [
    "0, 0, 0, 0",
    "0, 0, 0, 1",
    "0, 0, 1, 0",
    "0, 0, 1, 1",
    "0, 1, 0, 0",
    "0, 1, 0, 1",
    "0, 1, 1, 0",
    "0, 1, 1, 1",
    "0.5, 1.5, 2.25, 3.0",
    "0.5, 1.5, 2.25, 4.0",
    "0.5, 1.5, 3.25, 3.0",
    "0.5, 1.5, 3.25, 4.0",
    "0.5, 2.5, 2.25, 3.0",
    "0.5, 2.5, 2.25, 4.0",
    "0.5, 2.5, 3.25, 3.0",
    "0.5, 2.5, 3.25, 4.0",
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
open(data_file, "w") do io
    println(io, "# geometry = hypercube")
    println(io, "# scalar type = Float64")
    println(io, "# fitted interval = $(first(gamma_values)) to $(gamma_values[20])")
    println(io, "# fitted slope = $slope")
    println(io, "gamma  opnorm_K  gamma2_opnorm_K")
    for i in eachindex(gamma_values)
        println(
            io,
            "$(gamma_values[i])  $(K_kernel_norm_list[i])  $(scaled_K_kernel_norm_list[i])",
        )
    end
end
println("Saved plot to $figure_file.")
println("Saved scan data to $data_file.")
@printf("Small-gamma fitted slope = %.4f.\n", slope)
