
"""
    projection(Phi::AbstractMatrix, m::Int, n::Int)

Compute the nearest Kronecker product (NKP) projection of Phi onto B ⊗ A.

# Arguments
- Phi: The estimated coefficient matrix (size n_1^2 × n_2^2)
- n1, n2: The dimensions of matrices A and B, where A is n1 × n1 and B is n2 × n2

# Returns
- A_hat, B_hat: Projection estimators of A and B
- phi_est: Estimated phi calculated as B ⊗ A
"""
function projection(phi::AbstractMatrix{T}, dims::Tuple) where T
    n = size(phi, 1)
    n1, n2 = dims
    @assert n == n1 * n2 "Size mismatch: n1 * n2 ≠ n"
    tensor_phi = reshape(phi, (n1, n2, n1, n2))
    R = reshape(permutedims(tensor_phi, (1, 3, 2, 4)), n1 * n1, n2 * n2)
    F = svd(R)
    A = reshape(F.U[:, 1] * sqrt(F.S[1]), n1, n1)
    B = reshape(F.V[:, 1] * sqrt(F.S[1]), n2, n2)
    phi_est = kron(B, A)

    norm_A = norm(A)
    A = A / norm_A
    B = B * norm_A

    return (; A, B, phi_est)
end

function projection(phi::Vector{<:AbstractMatrix{T}}, dims::Tuple) where T
    n1, n2 = dims
    n, m = size(phi[1])
    k = length(phi)
    @assert n == n1 * n2 "Size mismatch: n1 * n2 ≠ size(phi,1)"
    @assert n == m "Each slice must be square (n×n)"

    As  = Vector{Array{T,2}}(undef, k)
    Bs  = Vector{Array{T,2}}(undef, k)
    Phis = Vector{Array{T,2}}(undef, k)

    for i in 1:k
        res = projection(phi[i], dims)
        As[i]   = res.A
        Bs[i]   = res.B
        Phis[i] = res.phi_est
    end

    return (; A = As, B = Bs, phi_est = Phis)
end

