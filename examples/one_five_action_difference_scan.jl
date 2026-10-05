using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")); io=devnull)

using EffectiveSpinfoam
using LorentzianSimplexSolver
using LaTeXStrings
using Plots
using Polynomials
using Printf

ScalarT = BigFloat
configure_precision!(ScalarT; precision=256)

gamma = ScalarT("1.0")
delta_l_values = ScalarT(10) .^ range(ScalarT(-5), ScalarT(-2), length=13)
fit_indices = 5:9

function fit_one_five_power_law(delta_l_values, values, fit_indices)
    polynomial = fit(
        log10.(delta_l_values[fit_indices]),
        log10.(values[fit_indices]),
        1,
    )
    intercept, slope = coeffs(polynomial)
    fit_curve = 10 .^ (intercept .+ slope .* log10.(delta_l_values))
    return slope, fit_curve
end

simplices = [
    [1, 2, 3, 4, 6],
    [1, 2, 3, 5, 6],
    [1, 2, 4, 5, 6],
    [1, 3, 4, 5, 6],
    [2, 3, 4, 5, 6],
]

coordinate_lines = [
    "0, 0, 0, 0",
    "0, -2.7745276335252114, -0.9809436521275706, -1.6990442448471226",
    "0, 0, 0, -3.398088489694245",
    "-0.24028114141347542, -0.6936319083813028, -0.9809436521275706, -1.6990442448471226",
    "0, 0, -2.942830956382712, -1.6990442448471226",
    "-0.068, -0.27, -0.5, -1.3",
]

flat_coordinates = [
    parse.(ScalarT, strip.(split(line, ",")))
    for line in coordinate_lines
]

boundary_edge = (1, 2)

result_directory = normpath(joinpath(@__DIR__, "..", "results"))
figure_file = joinpath(result_directory, "one_five_action_difference_vs_delta_l.png")
data_file = joinpath(result_directory, "one_five_action_difference_vs_delta_l.txt")
mkpath(result_directory)

println("1-5 move action-difference scan")
println("gamma = $gamma")
println("arithmetic = BigFloat, 256 bits")
println("all bulk edges: delta_l = 0")
println("boundary edge $boundary_edge: delta_l = 1e-5, ..., 1e-2")
println()

# The analytic second-order coefficients depend only on the flat background.
vertex_coordinates = Dict(i => flat_coordinates[i] for i in eachindex(flat_coordinates))

analytic_solver = build_solver_data(
    simplices,
    vertex_coordinates;
    bulk_sum_form=false,
    verbose=false,
    check=false,
)

analytic_data = build_effective_data(
    analytic_solver,
    simplices,
    vertex_coordinates;
    boundary_edges_perturbed=[sort(collect(boundary_edge))],
)

analytic_terms = compute_effective_terms(analytic_data, gamma)
number_of_bulk_edges = length(analytic_data.bulk_edges)

println("bulk edges = $(analytic_data.bulk_edges)")
println("varied boundary edge = $(analytic_data.boundary_edges_perturbed[1])")
println()

number_of_points = length(delta_l_values)
numerical_differences = Vector{Complex{ScalarT}}(undef, number_of_points)
analytic_differences = Vector{Complex{ScalarT}}(undef, number_of_points)
remainders = Vector{Complex{ScalarT}}(undef, number_of_points)
errors = Vector{ScalarT}(undef, number_of_points)
residuals = Vector{ScalarT}(undef, number_of_points)

previous_solution = nothing

for (i, delta_l) in enumerate(delta_l_values)
    println("[$i/$number_of_points] boundary delta_l = $delta_l")

    delta_lengths = vcat(
        zeros(ScalarT, number_of_bulk_edges),
        ScalarT[delta_l],
    )

    complex_data = prepare_action_difference(
        simplices,
        flat_coordinates,
        boundary_edge,
        delta_lengths;
        bulk_sum_form=true,
        gamma=gamma,
        verbose=false,
        check=false,
    )

    complex_result = solve_action_difference(
        complex_data,
        gamma;
        tol=ScalarT("1e-30"),
        maxiter=30,
        verbose=false,
        initial=previous_solution,
    )

    complex_result.converged || error("Complex Newton solve did not converge at delta_l = $delta_l.")
    global previous_solution = complex_result.solution

    linear_difference = sum(analytic_terms.linear_regge .* delta_lengths)
    quadratic_regge_difference = transpose(delta_lengths) *
        analytic_terms.quadratic_regge * delta_lengths
    non_regge_difference = transpose(delta_lengths) *
        analytic_terms.non_regge_correction * delta_lengths

    numerical_differences[i] = complex_result.action_difference
    analytic_differences[i] = linear_difference +
        quadratic_regge_difference + non_regge_difference
    residuals[i] = maximum(abs.(complex_result.residual))
    remainders[i] = numerical_differences[i] - analytic_differences[i]
    errors[i] = abs(remainders[i])

    println("    numerical Delta S = $(numerical_differences[i])")
    println("    analytic  Delta S = $(analytic_differences[i])")
    println("    absolute error    = $(errors[i])")
    println("    error / delta_l^3 = $(errors[i] / delta_l^3)")
    println("    max |dS|          = $(residuals[i])")
end

open(data_file, "w") do io
    println(io, "# delta_l  Re_DeltaS_numerical  Im_DeltaS_numerical  Re_DeltaS_analytic  Im_DeltaS_analytic  Re_remainder  Im_remainder  absolute_remainder  max_EOM_residual")
    for i in eachindex(delta_l_values)
        println(
            io,
            join(
                (
                    delta_l_values[i],
                    real(numerical_differences[i]),
                    imag(numerical_differences[i]),
                    real(analytic_differences[i]),
                    imag(analytic_differences[i]),
                    real(remainders[i]),
                    imag(remainders[i]),
                    errors[i],
                    residuals[i],
                ),
                "  ",
            ),
        )
    end
end

absolute_slope, absolute_fit = fit_one_five_power_law(
    delta_l_values,
    errors,
    fit_indices,
)
slope_label = @sprintf("%.4f", Float64(absolute_slope))

figure = plot(
    delta_l_values,
    errors;
    marker=:circle,
    markersize=6,
    color=:blue,
    xscale=:log10,
    yscale=:log10,
    xlabel=L"\delta \ell",
    ylabel=L"|R|",
    label="numerical result",
    legend=:bottomright,
    title=L"\mathrm{1\!-\!5\ move}",
    size=(800, 520),
    left_margin=8 * Plots.PlotMeasures.mm,
    right_margin=4 * Plots.PlotMeasures.mm,
    bottom_margin=6 * Plots.PlotMeasures.mm,
    guidefontsize=13,
    tickfontsize=10,
    legendfontsize=10,
    dpi=300,
)
plot!(
    figure,
    delta_l_values,
    absolute_fit;
    color=:orange,
    linewidth=2,
    linestyle=:dash,
    label="fit, p = $slope_label",
)

savefig(figure, figure_file)
println()
println("Power-law fit uses delta_l = $(delta_l_values[first(fit_indices)]) to $(delta_l_values[last(fit_indices)]).")
println("slope of |R| = $(absolute_slope)")
println("Saved data to $data_file")
println("Saved figure to $figure_file")
