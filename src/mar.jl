
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

function update_B(resp::AbstractArray{T}, pred::AbstractArray{T}, A::AbstractMatrix{T}) where T
    n2 = size(resp, 2)
    B_num = zeros(n2, n2)
    B_den = zeros(n2, n2)
    obs = size(resp, 3)

    for t in 1:obs
        B_num += resp[:, :, t]' * A * pred[:, :, t]
        B_den += resp[:, :, t]' * A'A * pred[:, :, t]
    end

    return B_num / B_den
end

function update_A(resp::AbstractArray{T}, pred::AbstractArray{T}, B::AbstractMatrix{T}) where T
    n1 = size(resp, 1)
    A_num = zeros(n1, n1)
    A_den = zeros(n1, n1)
    obs = size(resp, 3)

    for t in 1:obs
        A_num += resp[:, :, t] * B * pred[:, :, t]'
        A_den += pred[:, :, t] * B'B * pred[:, :, t]'
    end

    return A_num / A_den
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
    obj = ls_objective(resp, pred, A, B)

    track_a = fill(NaN, maxiter)
    track_b = fill(NaN, maxiter)
    track_obj = fill(NaN, maxiter)

    for i in 1:maxiter
        A_old = copy(A)
        B_old = copy(B)
        obj_old = copy(obj)
        B = update_B(resp, pred, A)
        A = update_A(resp, pred, B)

        obj = ls_objective(resp, pred, A, B)
        norm_A = norm(A)
        A = A / norm_A
        B = B * norm_A

        track_a[i] = norm(A - A_old)
        track_b[i] = norm(B - B_old)
        track_obj[i] = abs(obj - obj_old)

        if track_a[i] < tol || track_b[i] < tol || track_obj[i] < tol
            break
        end

        if i == maxiter
            @warn "Reached maximum number of iterations"
            return (; A, B, track_a, track_b, track_obj)
        end
    end

    track_a = track_a[.!isnan.(track_a)]
    track_b = track_b[.!isnan.(track_b)]
    track_obj = track_obj[.!isnan.(track_obj)]
    obj = ls_objective(resp, pred, A, B)

    return (; A, B, track_a, track_b, track_obj, obj)
end

function ls_objective(data::AbstractArray{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}; p=1) where T
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

function ls_objective(resp::AbstractArray{T}, pred::AbstractArray{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}; p=1) where T
    obs = size(resp, 3)
    ssr = 0
    for i in 1:(obs-p)
        residual = resp[:, :, i] - A * pred[:, :, i] * B'
        ssr += sum(abs2, residual)
    end
    
    return ssr
    
end




















