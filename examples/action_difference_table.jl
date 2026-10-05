using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")); io=devnull)

using EffectiveSpinfoam
using LorentzianSimplexSolver

ScalarT = Float64
configure_precision!(ScalarT)

gamma = ScalarT(1.0)
delta_l = ScalarT(1e-5)
result_file = normpath(joinpath(@__DIR__, "..", "results", "action_difference_table.txt"))
mkpath(dirname(result_file))

function analytic_action_difference(terms, delta_lengths)
    linear = sum(terms.linear_regge .* delta_lengths)
    quadratic_regge = transpose(delta_lengths) *
        terms.quadratic_regge * delta_lengths
    non_regge = transpose(delta_lengths) *
        terms.non_regge_correction * delta_lengths
    return linear + quadratic_regge + non_regge
end

function save_action_difference_row(
    result_file,
    geometry_name,
    perturbation_name,
    bulk_delta,
    boundary_delta,
    numerical_difference,
    analytic_difference,
    residual,
)
    remainder = numerical_difference - analytic_difference
    relative_difference = abs(remainder) / abs(numerical_difference)

    open(result_file, "a") do io
        println(
            io,
            join(
                (
                    geometry_name,
                    perturbation_name,
                    bulk_delta,
                    boundary_delta,
                    real(numerical_difference),
                    imag(numerical_difference),
                    real(analytic_difference),
                    imag(analytic_difference),
                    abs(remainder),
                    relative_difference,
                    residual,
                ),
                "  ",
            ),
        )
    end

    println("    numerical Delta S = $numerical_difference")
    println("    analytic  Delta S = $analytic_difference")
    println("    |R|               = $(abs(remainder))")
    println("    relative difference = $relative_difference")
    println("    max |dS|          = $residual")
end

function run_action_difference_cases(
    result_file,
    geometry_name,
    simplices,
    flat_coordinates,
    boundary_edge,
    delta_l,
    gamma,
)
    vertex_coordinates = Dict(
        i => flat_coordinates[i]
        for i in eachindex(flat_coordinates)
    )

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
    bulk_delta_lengths = zeros(ScalarT, number_of_bulk_edges)
    bulk_delta_lengths .= delta_l

    cases = [
        (
            name="bulk_only",
            bulk_delta=delta_l,
            boundary_delta=zero(delta_l),
            delta_lengths=vcat(bulk_delta_lengths, zero(delta_l)),
        ),
        (
            name="boundary_only",
            bulk_delta=zero(delta_l),
            boundary_delta=delta_l,
            delta_lengths=vcat(zeros(ScalarT, number_of_bulk_edges), delta_l),
        ),
        (
            name="bulk_and_boundary",
            bulk_delta=delta_l,
            boundary_delta=delta_l,
            delta_lengths=vcat(bulk_delta_lengths, delta_l),
        ),
    ]

    println("$geometry_name")
    println("    bulk edges = $(analytic_data.bulk_edges)")
    println("    varied bulk edges = $(analytic_data.bulk_edges)")
    println("    boundary edge = $(analytic_data.boundary_edges_perturbed[1])")

    for (i, case) in enumerate(cases)
        println("[$i/3] $(case.name)")

        complex_data = prepare_action_difference(
            simplices,
            flat_coordinates,
            boundary_edge,
            case.delta_lengths;
            bulk_sum_form=true,
            gamma=gamma,
            verbose=false,
            check=false,
        )
        complex_result = solve_action_difference(
            complex_data,
            gamma;
            tol=ScalarT(5e-14),
            maxiter=30,
            verbose=false,
        )
        complex_result.converged ||
            error("Complex Newton solve did not converge for $geometry_name, $(case.name).")

        analytic_difference = analytic_action_difference(
            analytic_terms,
            case.delta_lengths,
        )
        residual = maximum(abs.(complex_result.residual))

        save_action_difference_row(
            result_file,
            geometry_name,
            case.name,
            case.bulk_delta,
            case.boundary_delta,
            complex_result.action_difference,
            analytic_difference,
            residual,
        )
    end

    println()
end

open(result_file, "w") do io
    println(io, "# gamma = $gamma")
    println(io, "# Every nonzero length perturbation equals delta_l = $delta_l")
    println(io, "# Every bulk edge receives the same bulk_delta in bulk_only and bulk_and_boundary cases.")
    println(io, "# geometry  perturbation  bulk_delta  boundary_delta  Re_DeltaS_numerical  Im_DeltaS_numerical  Re_DeltaS_analytic  Im_DeltaS_analytic  abs_remainder  relative_difference  max_EOM_residual")
end

double_delta3_simplices = [
    [1, 2, 3, 4, 6],
    [1, 2, 3, 5, 6],
    [1, 2, 4, 5, 6],
    [1, 2, 3, 4, 7],
    [1, 2, 3, 5, 7],
    [1, 2, 4, 5, 7],
]

double_delta3_coordinates = [
    ScalarT.([0.0, 0.0, 0.0, 0.0]),
    ScalarT.([-0.068, -0.21988127663727278, -0.5316227766016838, -1.3316227766016839]),
    ScalarT.([0.0, 0.0, 0.0, -3.398088489694245]),
    ScalarT.([-0.24028114141347542, -0.6936319083813028, -0.9809436521275706, -1.6990442448471226]),
    ScalarT.([0.0, 0.0, -2.942830956382712, -1.6990442448471226]),
    ScalarT.([0.0, -2.7745276335252114, -0.9809436521275706, -1.6990442448471226]),
    ScalarT.([-2.4696884592430974, -3.893218630529324, -1.3565336794679874, -1.9090667752920147]),
]

one_five_simplices = [
    [1, 2, 3, 4, 6],
    [1, 2, 3, 5, 6],
    [1, 2, 4, 5, 6],
    [1, 3, 4, 5, 6],
    [2, 3, 4, 5, 6],
]

one_five_coordinates = [
    ScalarT.([0.0, 0.0, 0.0, 0.0]),
    ScalarT.([0.0, -2.7745276335252114, -0.9809436521275706, -1.6990442448471226]),
    ScalarT.([0.0, 0.0, 0.0, -3.398088489694245]),
    ScalarT.([-0.24028114141347542, -0.6936319083813028, -0.9809436521275706, -1.6990442448471226]),
    ScalarT.([0.0, 0.0, -2.942830956382712, -1.6990442448471226]),
    ScalarT.([-0.068, -0.27, -0.5, -1.3]),
]

println("Action-difference comparison at delta_l = $delta_l, gamma = $gamma")
println()

run_action_difference_cases(
    result_file,
    "double_delta3",
    double_delta3_simplices,
    double_delta3_coordinates,
    (1, 3),
    delta_l,
    gamma,
)

run_action_difference_cases(
    result_file,
    "one_five",
    one_five_simplices,
    one_five_coordinates,
    (1, 2),
    delta_l,
    gamma,
)

println("Saved comparison table to $result_file")
