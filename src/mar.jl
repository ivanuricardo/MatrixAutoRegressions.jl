
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
function projection(phi::AbstractMatrix{T}, n1::Int, n2::Int) where T
    n = size(phi, 1)
    @assert n == n1 * n2 " Size mismatch: n1 * n2 ≠ n"
    tensor_phi = reshape(phi, (n1, n2, n1, n2))
    R = reshape(permutedims(tensor_phi, (1, 3, 2, 4)), n1 * n1, n2 * n2)
    F = svd(R)
    A = reshape(F.U[:, 1] * sqrt(F.S[1]), n1, n1)
    B = reshape(F.V[:, 1] * sqrt(F.S[1]), n2, n2)
    phi_est = kron(B, A)
    return (; A, B, phi_est)
end

"""
    als(A_init, B_init, resp, pred; maxiter=100, tol=1e-6)

Alternating Least Squares estimation for the MAR(1) model:

    y_t = (B ⊗ A) y_{t-1} + e_t

# Arguments
- `A_init`, `B_init`: initial guesses for A and B.
- `resp`: response matrix (N × T).
- `pred`: predictor matrix (N × T).

# Keyword arguments
- `maxiter`: maximum number of ALS iterations.
- `tol`: convergence tolerance (Frobenius norm change).

# Returns
- `(A, B)`: estimated matrices.
"""
function als(A_init, B_init, resp, pred; maxiter=100, tol=1e-6)
    A = copy(A_init)
    B = copy(B_init)
    n1, n2 = size(A, 1), size(B, 1)
    obs = size(resp, 3)

    for i in 1:maxiter
        X_given_A = kron(I(n2), A) * pred
        B = resp * X_given_A' * inv(X_given_A * X_given_A')




        X_given_B = kron(B, I(n1)) * pred
        if i == maxiter
            @warn "Reached maximum number of iterations"
            return A, B
        end
    end
end

function ls_objective(data, A, B; p=1)
    obs = size(data, 3)
    resp = data[:, :, 2:end]
    pred = data[:, :, 1:end-1]

    ssr = 0
    for i in 1:(obs-p)
        residual = resp[:, :, i] - A * pred[:, :, i] * B'
        ssr += sum(abs2, residual)
    end
    
    return ssr
end




















