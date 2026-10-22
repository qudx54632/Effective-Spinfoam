using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")); io=devnull)

using EffectiveSpinfoam
using LorentzianSimplexSolver
using LinearAlgebra
using LaTeXStrings
using Plots
using Polynomials
using Printf

ScalarT = BigFloat
configure_precision!(ScalarT; precision=160)

gamma = ScalarT("1.0")
fit_delta_L_values = ScalarT(10) .^ range(ScalarT(-6), ScalarT(-4), length=11)
large_delta_L_values = ScalarT(10) .^ range(
    log10(ScalarT("2e-4")),
    log10(ScalarT("5e-4")),
    length=4,
)
delta_L_values = vcat(fit_delta_L_values, large_delta_L_values)
fit_indices = eachindex(fit_delta_L_values)

pi_T = ScalarT(pi)
theta_values = ScalarT[0, pi_T / 8, pi_T / 4, 3pi_T / 8, pi_T / 2]
theta_names = ["0", "\\pi/8", "\\pi/4", "3\\pi/8", "\\pi/2"]

# Relative perturbations of (1,6), (2,6), (3,6), (4,6), (5,6).
# Each choice produces one figure. Change these two vectors as needed.
bulk_weight_sets = [
    [1, 2, 3, 4, 5],
    [5, 4, 3, 2, 1],
]

function nonuniform_perturbation_direction(theta, bulk_direction)
    cutoff = sqrt(eps(typeof(theta)))
    bulk_amplitude = abs(cos(theta)) < cutoff ? zero(theta) : cos(theta)
    boundary_amplitude = abs(sin(theta)) < cutoff ? zero(theta) : sin(theta)
    return vcat(bulk_amplitude .* bulk_direction, boundary_amplitude)
end

function run_nonuniform_direction_scan()
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

    bulk_edges = [[1, 6], [2, 6], [3, 6], [4, 6], [5, 6]]
    boundary_edge = (1, 2)
    result_directory = normpath(joinpath(@__DIR__, "..", "results"))
    mkpath(result_directory)

    println("1-5 move: nonuniform bulk perturbations")
    println("gamma = $gamma, BigFloat, 160 bits")
    println("bulk edges = $bulk_edges, boundary edge = $(collect(boundary_edge))")
    println("delta_l_bulk_vector = delta_L cos(theta) u_bulk, with norm(u_bulk) = 1")
    println("delta_l_boundary = delta_L sin(theta)")
    println()

    # Compute the flat-background analytic coefficients once for both figures.
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
        boundary_edges_perturbed=[collect(boundary_edge)],
    )
    analytic_terms = compute_effective_terms(analytic_data, gamma)

    expected_edges = vcat(bulk_edges, [collect(boundary_edge)])
    analytic_data.perturb_edges == expected_edges ||
        error("The perturbation-edge order is not [five bulk edges, boundary edge].")

    number_of_directions = length(theta_values)
    number_of_points = length(delta_L_values)
    colors = [:blue, :red, :green, :purple, :orange]
    markers = [:circle, :square, :diamond, :utriangle, :dtriangle]

    for (profile_index, bulk_weights) in enumerate(bulk_weight_sets)
        # ||delta_l_bulk|| = delta_L cos(theta), so ||delta_lengths|| = delta_L.
        bulk_direction = normalize(ScalarT.(bulk_weights))
        file_stem = "one_five_action_difference_nonuniform_direction_scan_$(profile_index)"
        data_file = joinpath(result_directory, file_stem * ".txt")
        figure_file = joinpath(result_directory, file_stem * ".pdf")

        println("Bulk profile $profile_index: relative weights = $bulk_weights")

        numerical_differences = Matrix{Complex{ScalarT}}(
            undef,
            number_of_directions,
            number_of_points,
        )
        analytic_differences = similar(numerical_differences)
        remainders = similar(numerical_differences)
        errors = Matrix{ScalarT}(undef, number_of_directions, number_of_points)
        residuals = similar(errors)
        slopes = Vector{ScalarT}(undef, number_of_directions)

        for (direction_index, theta) in enumerate(theta_values)
            direction = nonuniform_perturbation_direction(theta, bulk_direction)
            previous_solution = nothing
            println("theta = $(theta_names[direction_index])")

            for (point_index, delta_L) in enumerate(delta_L_values)
                delta_lengths = delta_L .* direction
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

                complex_result.converged || error(
                    "Complex Newton solve did not converge for bulk profile $profile_index, " *
                    "theta = $(theta_names[direction_index]), delta_L = $delta_L.",
                )
                previous_solution = complex_result.solution

                linear_difference = sum(analytic_terms.linear_regge .* delta_lengths)
                quadratic_regge_difference = transpose(delta_lengths) *
                    analytic_terms.quadratic_regge * delta_lengths
                non_regge_difference = transpose(delta_lengths) *
                    analytic_terms.non_regge_correction * delta_lengths

                numerical_differences[direction_index, point_index] =
                    complex_result.action_difference
                analytic_differences[direction_index, point_index] =
                    linear_difference + quadratic_regge_difference + non_regge_difference
                remainders[direction_index, point_index] =
                    numerical_differences[direction_index, point_index] -
                    analytic_differences[direction_index, point_index]
                errors[direction_index, point_index] =
                    abs(remainders[direction_index, point_index])
                residuals[direction_index, point_index] =
                    maximum(abs.(complex_result.residual))

                println(
                    "    [$point_index/$number_of_points] delta_L = $delta_L, " *
                    "|R| = $(errors[direction_index, point_index]), " *
                    "max |dS| = $(residuals[direction_index, point_index])",
                )
            end

            polynomial = fit(
                log10.(delta_L_values[fit_indices]),
                log10.(errors[direction_index, fit_indices]),
                1,
            )
            slopes[direction_index] = coeffs(polynomial)[2]
            @printf("    fitted slope = %.4f\n\n", Float64(slopes[direction_index]))
        end

        open(data_file, "w") do io
            println(io, "# gamma = $gamma, BigFloat, 160 bits")
            println(io, "# bulk relative weights = $bulk_weights")
            println(io, "# normalized u_bulk = $bulk_direction")
            println(io, "# delta_l_bulk_vector = delta_L cos(theta) u_bulk")
            println(io, "# delta_l_boundary = delta_L sin(theta), norm(delta_lengths) = delta_L")
            println(io, "# edge order: (1,6) (2,6) (3,6) (4,6) (5,6) (1,2)")
            println(io, "# fitted slopes for theta = 0, pi/8, pi/4, 3pi/8, pi/2: $slopes")
            println(io, "# fit interval: 1e-6 <= delta_L <= 1e-4")
            println(io, "# theta  delta_L  delta_l_bulk_norm  delta_l_16  delta_l_26  delta_l_36  delta_l_46  delta_l_56  delta_l_12  Re_DeltaS_numerical  Im_DeltaS_numerical  Re_DeltaS_analytic  Im_DeltaS_analytic  Re_remainder  Im_remainder  absolute_remainder  max_EOM_residual")

            for (direction_index, theta) in enumerate(theta_values)
                direction = nonuniform_perturbation_direction(theta, bulk_direction)
                for (point_index, delta_L) in enumerate(delta_L_values)
                    delta_lengths = delta_L .* direction
                    values = (
                        theta,
                        delta_L,
                        norm(delta_lengths[1:length(bulk_edges)]),
                        delta_lengths...,
                        real(numerical_differences[direction_index, point_index]),
                        imag(numerical_differences[direction_index, point_index]),
                        real(analytic_differences[direction_index, point_index]),
                        imag(analytic_differences[direction_index, point_index]),
                        real(remainders[direction_index, point_index]),
                        imag(remainders[direction_index, point_index]),
                        errors[direction_index, point_index],
                        residuals[direction_index, point_index],
                    )
                    println(io, join(values, "  "))
                end
            end
        end

        figure_title = "1-5 move: non-uniform $profile_index"
        delta_L_plot = Float64.(delta_L_values)
        show_title = true
        legend_position = :bottomright

        figure = plot(
            xscale=:log10,
            yscale=:log10,
            xlabel=L"\delta L",
            ylabel=L"|R(\delta L,\theta)|",
            legend=legend_position,
            title=show_title ? figure_title : "",
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

        for direction_index in eachindex(theta_values)
            slope_label = @sprintf("%.4f", Float64(slopes[direction_index]))
            curve_label = latexstring(
                "\\theta=",
                theta_names[direction_index],
                ",\\ p=",
                slope_label,
            )
            plot!(
                figure,
                delta_L_plot,
                Float64.(errors[direction_index, :]);
                color=colors[direction_index],
                marker=markers[direction_index],
                markersize=4,
                linewidth=1.6,
                label=curve_label,
            )
        end

        reference_index = first(fit_indices)
        cubic_reference = errors[3, reference_index] .* (
            delta_L_values ./ delta_L_values[reference_index]
        ) .^ 3
        plot!(
            figure,
            delta_L_plot,
            Float64.(cubic_reference);
            color=:black,
            linewidth=1.5,
            linestyle=:dash,
            label=L"O(\delta L^3)",
        )

        savefig(figure, figure_file)
        display(figure)
        println("Saved data to $data_file")
        println("Saved figure to $figure_file")
        println()
    end
end

run_nonuniform_direction_scan()
