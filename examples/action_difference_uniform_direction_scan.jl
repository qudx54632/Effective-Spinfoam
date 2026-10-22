using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")); io=devnull)

using EffectiveSpinfoam
using LorentzianSimplexSolver
using LaTeXStrings
using Plots
using Polynomials
using Printf

function uniform_perturbation_direction(theta, number_of_bulk_edges, ::Type{T}) where {T}
    cutoff = sqrt(eps(T))
    bulk_amplitude = abs(cos(theta)) < cutoff ? zero(T) : cos(theta)
    boundary_amplitude = abs(sin(theta)) < cutoff ? zero(T) : sin(theta)
    bulk_components = fill(
        bulk_amplitude / sqrt(T(number_of_bulk_edges)),
        number_of_bulk_edges,
    )
    return vcat(bulk_components, boundary_amplitude)
end

function uniform_analytic_action_difference(terms, delta_lengths)
    linear = sum(terms.linear_regge .* delta_lengths)
    quadratic_regge = transpose(delta_lengths) *
        terms.quadratic_regge * delta_lengths
    non_regge = transpose(delta_lengths) *
        terms.non_regge_correction * delta_lengths
    return linear + quadratic_regge + non_regge
end

function uniform_power_law_slope(delta_L_values, values, fit_indices)
    polynomial = fit(
        log10.(delta_L_values[fit_indices]),
        log10.(values[fit_indices]),
        1,
    )
    return coeffs(polynomial)[2]
end

function save_uniform_scan_data(
    data_file,
    geometry,
    gamma,
    theta_values,
    delta_L_values,
    numerical_differences,
    analytic_differences,
    remainders,
    errors,
    residuals,
    slopes,
    fit_indices,
    ::Type{T},
) where {T}
    number_of_bulk_edges = length(geometry.bulk_edges)
    perturb_edges = vcat(geometry.bulk_edges, [collect(geometry.boundary_edge)])
    edge_columns = ["delta_l_$(edge[1])$(edge[2])" for edge in perturb_edges]

    open(data_file, "w") do io
        println(io, "# geometry = $(geometry.name)")
        println(io, "# gamma = $gamma, BigFloat, 160 bits")
        println(io, "# bulk edges = $(geometry.bulk_edges)")
        println(io, "# boundary edge = $(collect(geometry.boundary_edge))")
        println(io, "# delta_l_bulk_vector = delta_L cos(theta) u_bulk")
        println(io, "# u_bulk is uniform and has unit norm")
        println(io, "# delta_l_boundary = delta_L sin(theta)")
        println(io, "# fitted slopes = $slopes")
        println(
            io,
            "# fit interval: $(delta_L_values[first(fit_indices)]) <= delta_L <= " *
            "$(delta_L_values[last(fit_indices)])",
        )
        println(
            io,
            "# theta  delta_L  delta_l_bulk_norm  " *
            join(edge_columns, "  ") *
            "  Re_DeltaS_numerical  Im_DeltaS_numerical" *
            "  Re_DeltaS_analytic  Im_DeltaS_analytic" *
            "  Re_remainder  Im_remainder  absolute_remainder  max_EOM_residual",
        )

        for (direction_index, theta) in enumerate(theta_values)
            direction = uniform_perturbation_direction(
                theta,
                number_of_bulk_edges,
                T,
            )
            for (point_index, delta_L) in enumerate(delta_L_values)
                delta_lengths = delta_L .* direction
                values = (
                    theta,
                    delta_L,
                    abs(delta_L * cos(theta)),
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
end

function plot_uniform_scan(
    figure_file,
    figure_title,
    theta_names,
    delta_L_values,
    errors,
    slopes,
    fit_indices,
)
    colors = [:blue, :red, :green, :purple, :orange]
    markers = [:circle, :square, :diamond, :utriangle, :dtriangle]
    delta_L_plot = Float64.(delta_L_values)
    errors_plot = Float64.(errors)

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

    for direction_index in eachindex(theta_names)
        curve_label = latexstring(
            "\\theta=",
            theta_names[direction_index],
            ",\\ p=",
            @sprintf("%.4f", Float64(slopes[direction_index])),
        )
        plot!(
            figure,
            delta_L_plot,
            errors_plot[direction_index, :];
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
end

function run_uniform_scan(
    geometry,
    gamma,
    output_suffix,
    fit_delta_L_values,
    theta_values,
    theta_names,
    result_directory,
    ::Type{T},
) where {T}
    large_delta_L_values = T(10) .^ range(
        log10(T("2e-4")),
        log10(geometry.largest_delta_L),
        length=4,
    )
    delta_L_values = vcat(fit_delta_L_values, large_delta_L_values)
    fit_indices = eachindex(fit_delta_L_values)
    flat_coordinates = [
        parse.(T, strip.(split(line, ",")))
        for line in geometry.coordinate_lines
    ]
    number_of_bulk_edges = length(geometry.bulk_edges)
    number_of_directions = length(theta_values)
    number_of_points = length(delta_L_values)

    println("$(geometry.name) uniform action-difference direction scan")
    println("gamma = $gamma, BigFloat, 160 bits")
    println("bulk edges = $(geometry.bulk_edges)")
    println("boundary edge = $(collect(geometry.boundary_edge))")
    println("delta_l_bulk_vector = delta_L cos(theta) u_bulk")
    println("delta_l_boundary = delta_L sin(theta)")
    println()

    vertex_coordinates = Dict(
        i => flat_coordinates[i]
        for i in eachindex(flat_coordinates)
    )
    analytic_solver = build_solver_data(
        geometry.simplices,
        vertex_coordinates;
        bulk_sum_form=false,
        verbose=false,
        check=false,
    )
    analytic_data = build_effective_data(
        analytic_solver,
        geometry.simplices,
        vertex_coordinates;
        boundary_edges_perturbed=[collect(geometry.boundary_edge)],
    )
    analytic_terms = compute_effective_terms(analytic_data, gamma)

    expected_edges = vcat(
        geometry.bulk_edges,
        [collect(geometry.boundary_edge)],
    )
    analytic_data.perturb_edges == expected_edges ||
        error("The perturbation-edge order is not [bulk edges, boundary edge].")

    numerical_differences = Matrix{Complex{T}}(
        undef,
        number_of_directions,
        number_of_points,
    )
    analytic_differences = similar(numerical_differences)
    remainders = similar(numerical_differences)
    errors = Matrix{T}(undef, number_of_directions, number_of_points)
    residuals = similar(errors)
    slopes = Vector{T}(undef, number_of_directions)

    for (direction_index, theta) in enumerate(theta_values)
        direction = uniform_perturbation_direction(
            theta,
            number_of_bulk_edges,
            T,
        )
        previous_solution = nothing
        println("theta = $(theta_names[direction_index])")

        for (point_index, delta_L) in enumerate(delta_L_values)
            delta_lengths = delta_L .* direction
            complex_data = prepare_action_difference(
                geometry.simplices,
                flat_coordinates,
                geometry.boundary_edge,
                delta_lengths;
                bulk_sum_form=true,
                gamma=gamma,
                verbose=false,
                check=false,
            )
            complex_result = solve_action_difference(
                complex_data,
                gamma;
                tol=T("1e-30"),
                maxiter=30,
                verbose=false,
                initial=previous_solution,
            )
            complex_result.converged || error(
                "Complex Newton solve did not converge for $(geometry.name), " *
                "gamma = $gamma, theta = $(theta_names[direction_index]), " *
                "delta_L = $delta_L.",
            )
            previous_solution = complex_result.solution

            numerical_differences[direction_index, point_index] =
                complex_result.action_difference
            analytic_differences[direction_index, point_index] =
                uniform_analytic_action_difference(analytic_terms, delta_lengths)
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

        slopes[direction_index] = uniform_power_law_slope(
            delta_L_values,
            errors[direction_index, :],
            fit_indices,
        )
        @printf("    fitted slope = %.4f\n\n", Float64(slopes[direction_index]))
    end

    file_stem = geometry.file_stem * output_suffix
    data_file = joinpath(result_directory, file_stem * ".txt")
    figure_file = joinpath(result_directory, file_stem * ".pdf")
    save_uniform_scan_data(
        data_file,
        geometry,
        gamma,
        theta_values,
        delta_L_values,
        numerical_differences,
        analytic_differences,
        remainders,
        errors,
        residuals,
        slopes,
        fit_indices,
        T,
    )

    gamma_title = gamma == one(T) ? "" : L", $\gamma=0.01$"
    figure_title = LaTeXString(geometry.figure_title * gamma_title)
    plot_uniform_scan(
        figure_file,
        figure_title,
        theta_names,
        delta_L_values,
        errors,
        slopes,
        fit_indices,
    )

    println("Saved data to $data_file")
    println("Saved figure to $figure_file")
    println()
end

function run_uniform_direction_scans()
    ScalarT = BigFloat
    configure_precision!(ScalarT; precision=160)

    fit_delta_L_values = ScalarT(10) .^
        range(ScalarT(-6), ScalarT(-4), length=11)
    pi_T = ScalarT(pi)
    theta_values = ScalarT[0, pi_T / 8, pi_T / 4, 3pi_T / 8, pi_T / 2]
    theta_names = ["0", "\\pi/8", "\\pi/4", "3\\pi/8", "\\pi/2"]
    result_directory = normpath(joinpath(@__DIR__, "..", "results"))
    mkpath(result_directory)

    geometries = [
        (
            name="Double-Delta3",
            file_stem="double_delta3_action_difference_direction_scan",
            figure_title=L"Double-$\Delta_3$",
            simplices=[
                [1, 2, 3, 4, 6],
                [1, 2, 3, 5, 6],
                [1, 2, 4, 5, 6],
                [1, 2, 3, 4, 7],
                [1, 2, 3, 5, 7],
                [1, 2, 4, 5, 7],
            ],
            coordinate_lines=[
                "0, 0, 0, 0",
                "-0.068, -0.21988127663727278, -0.5316227766016838, -1.3316227766016839",
                "0, 0, 0, -3.398088489694245",
                "-0.24028114141347542, -0.6936319083813028, -0.9809436521275706, -1.6990442448471226",
                "0, 0, -2.942830956382712, -1.6990442448471226",
                "0, -2.7745276335252114, -0.9809436521275706, -1.6990442448471226",
                "-2.4696884592430974, -3.893218630529324, -1.3565336794679874, -1.9090667752920147",
            ],
            bulk_edges=[[1, 2]],
            boundary_edge=(1, 3),
            largest_delta_L=ScalarT("1e-3"),
        ),
        (
            name="1-5 move",
            file_stem="one_five_action_difference_direction_scan",
            figure_title="1-5 move",
            simplices=[
                [1, 2, 3, 4, 6],
                [1, 2, 3, 5, 6],
                [1, 2, 4, 5, 6],
                [1, 3, 4, 5, 6],
                [2, 3, 4, 5, 6],
            ],
            coordinate_lines=[
                "0, 0, 0, 0",
                "0, -2.7745276335252114, -0.9809436521275706, -1.6990442448471226",
                "0, 0, 0, -3.398088489694245",
                "-0.24028114141347542, -0.6936319083813028, -0.9809436521275706, -1.6990442448471226",
                "0, 0, -2.942830956382712, -1.6990442448471226",
                "-0.068, -0.27, -0.5, -1.3",
            ],
            bulk_edges=[[1, 6], [2, 6], [3, 6], [4, 6], [5, 6]],
            boundary_edge=(1, 2),
            largest_delta_L=ScalarT("5e-4"),
        ),
    ]

    gamma_scans = [
        (gamma=ScalarT("1.0"), output_suffix=""),
        (gamma=ScalarT("0.01"), output_suffix="_gamma_001"),
    ]

    for gamma_scan in gamma_scans
        for geometry in geometries
            run_uniform_scan(
                geometry,
                gamma_scan.gamma,
                gamma_scan.output_suffix,
                fit_delta_L_values,
                theta_values,
                theta_names,
                result_directory,
                ScalarT,
            )
        end
    end
end

run_uniform_direction_scans()
