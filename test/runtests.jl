using Test
using EffectiveSpinfoam
using LorentzianSimplexSolver

@testset verbose=true "EffectiveSpinfoam analytic workflow" begin
    ScalarT = Float64
    configure_precision!(ScalarT)

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

    vertex_coordinates = build_vertex_coordinates(
        simplices,
        coordinate_lines,
        ScalarT,
    )
    solver = build_solver_data(
        simplices,
        vertex_coordinates;
        bulk_sum_form=false,
    )
    data = build_effective_data(
        solver,
        simplices,
        vertex_coordinates,
    )
    result = compute_effective_terms(data, ScalarT(1.0))

    number_of_bulk_faces = length(data.eta_symbols)
    number_of_bulk_edges = length(data.bulk_edges)

    @test solver.bulk_sum_form == false
    @test size(result.K_kernel) == (number_of_bulk_faces, number_of_bulk_faces)
    @test size(result.dηdl_bulk) == (number_of_bulk_faces, number_of_bulk_edges)
    @test size(result.transverse_basis, 2) == number_of_bulk_faces - number_of_bulk_edges
    @test isapprox(
        transpose(result.dηdl_bulk) * result.transverse_basis,
        zeros(number_of_bulk_edges, number_of_bulk_faces - number_of_bulk_edges);
        atol=1e-12,
        rtol=0,
    )
    @test all(isfinite, result.K_kernel)
    @test all(isfinite, result.linear_regge)
    @test all(isfinite, result.quadratic_regge)
    @test all(isfinite, result.non_regge_correction)
    @test result.spinfoam_quadratic ≈
        result.quadratic_regge + result.non_regge_correction
end
