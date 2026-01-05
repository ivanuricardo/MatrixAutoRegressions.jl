
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
    n1, n2 = dims
    tensor_phi = reshape(phi, (n1, n2, n1, n2))
    R = reshape(permutedims(tensor_phi, (1, 3, 2, 4)), n1 * n1, n2 * n2)
    F = svd(R)
    A = reshape(F.U[:, 1] * sqrt(F.S[1]), n1, n1)
    B = reshape(F.V[:, 1] * sqrt(F.S[1]), n2, n2)
    phi_est = kron(B, A)

    A, B = normalize_slices(A, B)

    return (; A, B, phi_est)
end

function projection(phivec::Vector{<:AbstractMatrix{T}}, dims::Tuple{Int,Int}) where T
    p = length(phivec)
    n1, n2 = dims
    As = Vector{Matrix{T}}(undef, p)
    Bs = Vector{Matrix{T}}(undef, p)
    Phis = Vector{Matrix{T}}(undef, p)

    for (i, phi) in enumerate(phivec)
        @assert size(phi,1) == size(phi,2) "All φ must be square matrices"
        res = projection(phi, dims)
        As[i] = res.A
        Bs[i] = res.B
        Phis[i] = res.phi_est
    end

    return (; A = As, B = Bs, phi_est = Phis)
end

function permutation_matrix(dims; dense::Bool=false)
    n1, n2 = dims
    n = n1 * n2
    N = n * n
    idx = reshape(1:N, n, n)

    tensor_idx = reshape(idx, (n1, n2, n1, n2))
    permuted = permutedims(tensor_idx, (1, 3, 2, 4))
    R_idx = reshape(permuted, n1 * n1, n2 * n2)
    perm = vec(R_idx)
    P_sparse = sparse(1:N, perm, ones(N), N, N)

    return dense ? Matrix(P_sparse) : P_sparse
end

