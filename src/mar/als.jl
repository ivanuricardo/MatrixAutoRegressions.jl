
function update_A(resp::AbstractArray{T},
                  pred::AbstractArray{T},
                  B::AbstractVecOrMat{T};
                  Sigma2=I) where T
    n1 = size(resp, 1)
    A_num = zeros(T, n1, n1)
    A_den = zeros(T, n1, n1)
    obs = size(resp, 3)

    for t in 1:obs
        A_num += (resp[:, :, t] / Sigma2) * B * pred[:, :, t]'
        A_den += (pred[:, :, t] * B' / Sigma2) * B * pred[:, :, t]'
    end

    return A_num / A_den
end

function update_B(resp::AbstractArray{T},
                  pred::AbstractArray{T},
                  A::AbstractVecOrMat{T};
                  Sigma1=I) where T
    n2 = size(resp, 2)
    B_num = zeros(T, n2, n2)
    B_den = zeros(T, n2, n2)
    obs = size(resp, 3)

    for t in 1:obs
        B_num += (resp[:, :, t]' / Sigma1) * A * pred[:, :, t]
        B_den += (pred[:, :, t]' * A' / Sigma1) * A * pred[:, :, t]
    end

    return B_num / B_den
end

function residual_given_idx(
    data::AbstractArray{T}, 
    A::Vector{<:AbstractMatrix}, 
    B::Vector{<:AbstractMatrix},
    given_idx::Int,
) where {T}

    p = length(A)
    @assert length(B) == p "length(A) and length(B) must match"
    @assert 1 <= given_idx <= p "given_idx out of bounds"

    m, n, obs = size(data)
    obs_eff = obs - p

    resp = data[:, :, (p+1):end]
    residuals = copy(resp)

    @inbounds for i in 1:p
        if i == given_idx
            continue
        end
        Ai, Bi = A[i], B[i]
        pred = data[:, :, (p+1-i):(end-i)]

        @inbounds for t in 1:obs_eff
            residuals[:,:,t] .-= Ai * pred[:,:,t] * Bi'
        end
    end

    return residuals
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
function als(
    resp::AbstractArray{T},
    pred::AbstractArray{T},
    A::AbstractVecOrMat,
    B::AbstractVecOrMat{T};
    maxiter::Int=100,
    tol::Real=1e-6,
    Sigma1=I,
    Sigma2=I,
    ) where T

    n1, n2 = size(A, 1), size(B, 1)
    obs = size(resp, 3)
    obj = ls_objective(resp, pred, A, B)

    track_obj = fill(NaN, maxiter)

    num_iter = 0
    for i in 1:maxiter
        num_iter += 1
        obj_old = copy(obj)
        B = update_B(resp, pred, A; Sigma1)
        A = update_A(resp, pred, B; Sigma2)

        obj = ls_objective(resp, pred, A, B)
        norm_A = norm(A)
        A = A / norm_A
        B = B * norm_A

        track_obj[i] = abs(obj - obj_old)

        if track_obj[i] < tol
            track_obj = track_obj[.!isnan.(track_obj)]
            return (; A, B, track_obj, obj, num_iter)
        end

        if i == maxiter
            @warn "Reached maximum number of iterations"
            return (; A, B, track_obj, obj, num_iter)
        end
    end
end

function als(
    data::AbstractArray{T},
    A::Vector{<:AbstractMatrix},
    B::Vector{<:AbstractMatrix};
    maxiter::Int=100,
    tol::Real=1e-6,
    Sigma1=I,
    Sigma2=I,
    ) where T

    n1, n2 = size(A[1], 1), size(B[1], 1)
    p = length(A)
    obj = ls_objective(data, A, B; p)

    track_obj = fill(NaN, maxiter)

    num_iter = 0
    for i in 1:maxiter
        num_iter += 1
        obj_old = copy(obj)

        for j in 1:p
            resp = residual_given_idx(data, A, B, j)
            pred = data[:, :, (p+1-j):(end-j)]
            results = als(resp, pred, A[j], B[j]; maxiter, tol, Sigma1, Sigma2)
            A[j] = results.A
            B[j] = results.B
        end

        obj = ls_objective(data, A, B; p)
        track_obj[i] = abs(obj - obj_old)

        if track_obj[i] < tol
            track_obj = track_obj[.!isnan.(track_obj)]
            return (; A, B, track_obj, obj, num_iter)
        end

        if i == maxiter
            @warn "Reached maximum number of iterations"
            return (; A, B, track_obj, obj, num_iter)
        end

    end

end

