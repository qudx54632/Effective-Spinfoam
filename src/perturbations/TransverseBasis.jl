module TransverseBasis

using LinearAlgebra
using LorentzianSimplexSolver

export compute_transverse_basis

"""
    compute_transverse_basis(dηdl)

Return an orthonormal basis transverse to the Regge submanifold.
The columns `e` satisfy `transpose(dηdl) * e = 0`.
"""
function compute_transverse_basis(dηdl::Matrix{T}) where {T<:Real}
    tol = T(LorentzianSimplexSolver.PrecisionUtils.get_tolerance())
    return nullspace(transpose(dηdl); atol=tol, rtol=tol)
end

end # module
