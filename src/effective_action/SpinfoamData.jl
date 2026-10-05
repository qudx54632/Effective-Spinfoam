function flat_coordinate_dict(simplices, coordinates)
    coordinates isa AbstractDict && return coordinates
    vertices = sort(unique(Iterators.flatten(simplices)))
    return Dict(vertex => coordinates[vertex] for vertex in vertices)
end

function prepare_geometry_data(
    simplices,
    coordinates;
    curved::Bool=false,
    verbose::Bool=false,
    check::Bool=false,
)
    input_coordinates = curved ?
        coordinates : flat_coordinate_dict(simplices, coordinates)
    geometry = curved ?
        construct_curved_geometry(input_coordinates; verbose=verbose) :
        construct_geometry(simplices, input_coordinates; verbose=verbose)

    prepare_global_geometry!(
        geometry,
        simplices;
        check=check,
        verbose=verbose,
    )
    regge = compute_regge_action(geometry, simplices, input_coordinates)

    return (
        geometry=geometry,
        regge=regge,
        coordinates=input_coordinates,
    )
end

"""Construct geometry and the exact action used by the complex action-difference workflow."""
function prepare_action_data(
    simplices,
    coordinates;
    bulk_sum_form::Bool=true,
    curved::Bool=false,
    gamma=1.0,
    verbose::Bool=false,
    check::Bool=false,
)
    bulk_sum_form || error(
        "The complex critical-point workflow uses bulk_sum_form=true (the exact bulk action).",
    )

    geometry_data = prepare_geometry_data(
        simplices,
        coordinates;
        curved=curved,
        verbose=verbose,
        check=check,
    )
    exact_spinfoam = compute_spinfoam_action(
        geometry_data.geometry,
        geometry_data.regge;
        gamma=gamma,
        bulk_sum_form=bulk_sum_form,
    )

    return merge(
        geometry_data,
        (exact_spinfoam=exact_spinfoam, bulk_sum_form=bulk_sum_form),
    )
end
