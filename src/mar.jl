
mutable struct MAR <: AbstractARModel
    A::Union{Nothing, Matrix{Float64}}
    B::Union{Nothing, Matrix{Float64}}
    p::Int
    Sigma1::Union{Nothing, Matrix{Float64}}
    Sigma2::Union{Nothing, Matrix{Float64}}
    dims::Tuple{Int,Int}
    obs::Int
    method::Symbol
    resp::AbstractArray
    pred::AbstractArray
    maxiter::Int
    tol::Float64
    iters::Union{Nothing, Int}
end

function MAR(data::AbstractArray;
    p::Int=1,
    method::Symbol=:ls,
    A::Union{Nothing, AbstractVecOrMat}=nothing,
    B::Union{Nothing, AbstractVecOrMat}=nothing,
    maxiter::Int=100,
    tol::Real=1e-6,
    )

    # Demean over time
    demeaned_data = data .- mean(data, dims=3)
    resp = demeaned_data[:, :, 2:end]
    pred = demeaned_data[:, :, 1:end-1]

    dims = size(data)[1:2]
    obs = size(data, 3)
    iters = nothing

    return MAR(A, B, p, nothing, nothing, dims, obs, method, resp, pred, maxiter, tol, iters)
end

function fit!(model::MAR)
    ols_est = ols(model.resp, model.pred; model.p)
    proj_est = projection(ols_est, model.dims)

    if model.method == :proj
        model.A = proj_est.A
        model.B = proj_est.B

    elseif model.method == :ls

        A0 = isnothing(model.A) ? proj_est.A : copy(model.A)
        B0 = isnothing(model.B) ? proj_est.B : copy(model.B)

        results = als(A0, B0, model.resp, model.pred;
            maxiter=model.maxiter, tol=model.tol)

        model.A = results.A
        model.B = results.B
        model.iters = results.num_iter

    elseif model.method == :mle

        A0 = isnothing(model.A) ? proj_est.A : copy(model.A)
        B0 = isnothing(model.B) ? proj_est.B : copy(model.B)
        Sigma1_init = Matrix{Float64}(I(model.dims[1]))
        Sigma2_init = Matrix{Float64}(I(model.dims[2]))

        results = mle(A0, B0, Sigma1_init, Sigma2_init, model.resp, model.pred;
            maxiter=model.maxiter, tol=model.tol)

        model.A = results.A
        model.B = results.B
        model.Sigma1 = results.Sigma1
        model.Sigma2 = results.Sigma2
        model.iters = results.num_iter

    else
        throw(ArgumentError("Unknown method: $(model.method)"))
    end

    return model
end

_update_fac(S::UniformScaling) = S
_update_fac(S::AbstractMatrix) = factorize(S)

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

function update_B(resp::AbstractArray{T},
                  pred::AbstractArray{T},
                  A::AbstractVecOrMat{T};
                  Sigma1=I) where T
    n2 = size(resp, 2)
    B_num = zeros(T, n2, n2)
    B_den = zeros(T, n2, n2)
    obs = size(resp, 3)

    fac = _update_fac(Sigma1)

    for t in 1:obs
        B_num += (resp[:, :, t]' / fac) * A * pred[:, :, t]
        B_den += (pred[:, :, t]' * A' / fac) * A * pred[:, :, t]
    end

    return B_num / B_den
end

function update_A(resp::AbstractArray{T},
                  pred::AbstractArray{T},
                  B::AbstractVecOrMat{T};
                  Sigma2=I) where T
    n1 = size(resp, 1)
    A_num = zeros(T, n1, n1)
    A_den = zeros(T, n1, n1)
    obs = size(resp, 3)

    fac = _update_fac(Sigma2)

    for t in 1:obs
        A_num += (resp[:, :, t] / fac) * B * pred[:, :, t]'
        A_den += (pred[:, :, t] * B' / fac) * B * pred[:, :, t]'
    end

    return A_num / A_den
end

function update_Sigma1(resp::AbstractArray{T},
    pred::AbstractArray{T},
    A::AbstractVecOrMat{T},
    B::AbstractVecOrMat{T},
    Sigma2::AbstractMatrix{T}) where T

    n1, n2, obs = size(resp)
    Sigma1 = zeros(T, n1, n1)
    fac = _update_fac(Sigma2)

    for t in 1:obs
        residual = resp[:, :, t] - A * pred[:, :, t] * B'
        Sigma1 += (residual / fac) * residual'
    end

    return Sigma1 / (n2 * obs)
end

function update_Sigma2(resp::AbstractArray{T},
    pred::AbstractArray{T},
    A::AbstractVecOrMat{T},
    B::AbstractVecOrMat{T},
    Sigma1::AbstractMatrix{T}) where T
    
    n1, n2, obs = size(resp)
    Sigma2 = zeros(T, n2, n2)
    fac = _update_fac(Sigma1)

    for t in 1:obs
        residual = resp[:, :, t] - A * pred[:, :, t] * B'
        Sigma2 += (residual' / fac) * residual
    end

    return Sigma2 / (n1 * obs)
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
function als(A_init::AbstractArray{T},
    B_init::AbstractArray{T},
    data::AbstractArray{T},
    maxiter::Int=100,
    tol::Real=1e-6,
    p::Int=1
    ) where T

    A = copy(A_init)
    B = copy(B_init)
    n1, n2 = size(A, 1), size(B, 1)
    obs = size(resp, 3)
    obj = ls_objective(resp, pred, A, B)

    track_obj = fill(NaN, maxiter)

    num_iter = 0
    for i in 1:maxiter
        num_iter += 1
        A_old = copy(A)
        B_old = copy(B)
        obj_old = copy(obj)
        B = update_B(resp, pred, A)
        A = update_A(resp, pred, B)

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

function mle(A_init::AbstractVecOrMat{T},
    B_init::AbstractVecOrMat{T},
    Sigma1_init::AbstractMatrix{T},
    Sigma2_init::AbstractMatrix{T},
    resp::AbstractArray{T},
    pred::AbstractArray{T};
    maxiter::Int=100,
    tol::Real=1e-6,
    ) where T
    A = copy(A_init)
    B = copy(B_init)
    Sigma1 = copy(Sigma1_init)
    Sigma2 = copy(Sigma2_init)

    n1, n2 = size(A, 1), size(B, 1)
    obs = size(resp, 3)
    obj = ls_objective(resp, pred, A, B)

    track_obj = fill(NaN, maxiter)

    num_iter = 0
    for i in 1:maxiter
        num_iter += 1
        A_old = copy(A)
        B_old = copy(B)
        Sigma1_old = copy(Sigma1)
        Sigma2_old = copy(Sigma2)
        obj_old = copy(obj)

        B = update_B(resp, pred, A)
        A = update_A(resp, pred, B)
        Sigma1 = update_Sigma1(resp, pred, A, B, Sigma2)
        Sigma2 = update_Sigma2(resp, pred, A, B, Sigma1)

        obj = mle_objective(resp, pred, A, B, Sigma1, Sigma2)
        norm_A = norm(A)
        A = A / norm_A
        B = B * norm_A

        norm_Sigma1 = norm(Sigma1)
        norm_Sigma2 = norm(Sigma2)
        Sigma1 = Sigma1 / norm_Sigma1
        Sigma2 = Sigma2 * norm_Sigma2

        track_obj[i] = abs(obj - obj_old)

        if track_obj[i] < tol

            track_obj = track_obj[.!isnan.(track_obj)]
            return (; A, B, Sigma1, Sigma2, track_obj, obj, num_iter)
        end

        if i == maxiter
            @warn "Reached maximum number of iterations"
            return (; A, B, Sigma1, Sigma2, track_obj, obj, num_iter)
        end
    end
end

function ls_objective(data::AbstractArray{T}, A::AbstractArray{T}, B::AbstractArray{T}; p=1) where T
    n1, n2, obs = size(data)

    ssr = 0
    for i in (p+1):obs
        pred = zeros(n1, n2)
        for j in 1:p
            pred .+= A[:, :, j] * data[:, :, i-j] * B[:, :, j]'
        end
        residual = data[:, :, i] - pred
        ssr += sum(abs2, residual)
    end
    
    return ssr
end

function ls_objective(model::MAR)
    require_fitted(model)
    return ls_objective(model.resp, model.pred, model.A, model.B; model.p)
end

function mle_objective(data::AbstractArray{T}, A::AbstractArray{T}, B::AbstractArray{T}, Sigma1::AbstractMatrix{T}, Sigma2::AbstractMatrix{T}; p=1) where T
    n1, n2, obs = size(data)

    ssr = 0
    fac1 = _update_fac(Sigma1)
    fac2 = _update_fac(Sigma2)

    for i in (p+1):obs
        pred = zeros(n1, n2)
        for j in 1:p
            pred .+= A[:, :, j] * data[:, :, i-j] * B[:, :, j]'
        end

        residual = data[:, :, i] - pred
        ssr += tr((fac1 \ residual) / fac2 * residual')
    end
    eff_obs = obs - p
    first_cov = -n2 * eff_obs * logdet(fac1)
    second_cov = -n1 * eff_obs * logdet(fac2)

    return first_cov + second_cov - ssr
    
end

function mle_objective(model::MAR)
    require_fitted(model)
    if model.Sigma1 == nothing || model.Sigma2 == nothing
        model.Sigma1 = I(model.dims[1])
        model.Sigma2 = I(model.dims[2])
    end
    return mle_objective(model.resp, model.pred, model.A, model.B, model.Sigma1, model.Sigma2; model.p)
end

