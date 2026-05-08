
function update_Sigma1(
    data::AbstractArray{T}, 
    A::Vector{<:AbstractMatrix{T}}, 
    B::Vector{<:AbstractMatrix{T}}, 
    Sigma2::AbstractMatrix,
) where T
    p = length(A)
    @assert length(B) == p "length(A) and length(B) must match"
    n1, n2, obs = size(data)
    obs_eff = obs - p
    Sigma1 = zeros(T, n1, n1)
    Sigma2_chol = cholesky(Symmetric(Sigma2))

    pred = zeros(T, n1, n2)
    residual = zeros(T, n1, n2)
    tmp_AX = zeros(T, n1, n2)
    tmp_resS2 = zeros(T, n1, n2)

    @inbounds for t in 1:obs_eff
        resp = data[:, :, t + p]

        fill!(pred, zero(T))
        @inbounds for i in 1:p
            X_t = data[:, :, t + p - i]
            mul!(tmp_AX, A[i], X_t)
            mul!(pred, tmp_AX, B[i]', one(T), one(T))
        end

        residual .= resp .- pred

        tmp_resS2 .= residual
        rdiv!(tmp_resS2, Sigma2_chol)
        mul!(Sigma1, tmp_resS2, residual', one(T), one(T))
    end

    return Sigma1 / (n2 * obs_eff)
end

function update_Sigma2(
    data::AbstractArray{T}, 
    A::Vector{<:AbstractMatrix{T}}, 
    B::Vector{<:AbstractMatrix{T}}, 
    Sigma1::AbstractMatrix,
) where T
    p = length(A)
    n1, n2, obs = size(data)
    obs_eff = obs - p
    Sigma2 = zeros(T, n2, n2)
    Sigma1_chol = cholesky(Symmetric(Sigma1))

    pred = zeros(T, n1, n2)
    residual = zeros(T, n1, n2)
    tmp_AX = zeros(T, n1, n2)
    tmp_resS1 = zeros(T, n2, n1)

    @inbounds for t in 1:obs_eff
        resp = data[:, :, t + p]

        fill!(pred, zero(T))
        @inbounds for i in 1:p
            X_t = data[:, :, t + p - i]
            mul!(tmp_AX, A[i], X_t)
            mul!(pred, tmp_AX, B[i]', one(T), one(T))
        end

        residual .= resp .- pred

        tmp_resS1 .= residual'
        rdiv!(tmp_resS1, Sigma1_chol)
        mul!(Sigma2, tmp_resS1, residual, one(T), one(T))
    end

    return Sigma2 / (n1 * obs_eff)
end

function mle(
    data::AbstractArray{T},
    A::Vector{<:AbstractMatrix},
    B::Vector{<:AbstractMatrix},
    Sigma1::AbstractMatrix,
    Sigma2::AbstractMatrix;
    maxiter::Int=500,
    tol::Real=1e-6,
    ) where T

    n1, n2 = size(A[1], 1), size(B[1], 1)
    p = length(A)
    resp = data[:, :, (p+1):end]
    pred = structure_lagged_data(data; p)

    resp = resp .- mean(resp, dims=3)
    pred = pred .- mean(pred, dims=3)
    obj = mle_objective(data, A, B, Sigma1, Sigma2)

    Astack = hcat(A...)
    Bstack = hcat(B...)

    track_obj = fill(NaN, maxiter)

    num_iter = 0
    for i in 1:maxiter
        num_iter += 1
        obj_old = copy(obj)

        Sigma2_chol = cholesky(Symmetric(Sigma2))
        Sigma1_chol = cholesky(Symmetric(Sigma1))
        Astack = update_A(resp, pred, Bstack; Sigma2=Sigma2_chol)
        Bstack = update_B(resp, pred, Astack; Sigma1=Sigma1_chol)

        Astack, Bstack = normalize_slices(Astack, Bstack)

        A = [@view Astack[:, (k-1)*n1+1 : k*n1] for k in 1:p]
        B = [@view Bstack[:, (k-1)*n2+1 : k*n2] for k in 1:p]

        Sigma1 = update_Sigma1(data, A, B, Sigma2)
        Sigma2 = update_Sigma2(data, A, B, Sigma1)

        Sigma1, Sigma2 = normalize_slices(Sigma1, Sigma2)

        Sigma1 = Symmetric((Sigma1 + Sigma1')/2)
        Sigma2 = Symmetric((Sigma2 + Sigma2')/2)

        obj = mle_objective(data, A, B, Sigma1, Sigma2)
        track_obj[i] = abs(obj - obj_old)

        if track_obj[i] < tol
            A = [@view Astack[:, (k-1)*n1+1 : k*n1] for k in 1:p]
            B = [@view Bstack[:, (k-1)*n2+1 : k*n2] for k in 1:p]
            track_obj = track_obj[.!isnan.(track_obj)]
            return (; A, B, Sigma1, Sigma2, track_obj, obj, num_iter)
        end

        if i == maxiter
            A = [@view Astack[:, (k-1)*n1+1 : k*n1] for k in 1:p]
            B = [@view Bstack[:, (k-1)*n2+1 : k*n2] for k in 1:p]
            @warn "Reached maximum number of iterations"
            return (; A, B, Sigma1, Sigma2, track_obj, obj, num_iter)
        end
    end
end

function flipflop_covariance(X::AbstractArray;
    maxiter::Int=500,
    tol::Real=1e-6,
    sigma1::Union{Nothing, Matrix{Float64}}=nothing,
    sigma2::Union{Nothing, Matrix{Float64}}=nothing,
    )

    n1, n2, obs = size(X)

    # Initial values
    sigma1 = sigma1 == nothing ? I(n1) : copy(sigma1)
    sigma2 = sigma2 == nothing ? I(n2) : copy(sigma2)

    iters = 0
    for iter in 1:maxiter
        iters = iter
        sigma1_old = sigma1
        sigma2_old = sigma2

        # === Update Σ2 given Σ1 ===
        sigma1inv = I / sigma1
        S2 = zeros(n2, n2)

        for t in 1:obs
            Xt = view(X, :, :, t)
            S2 .+= Xt' * sigma1inv * Xt
        end

        sigma2 = S2 / (obs * n1)

        # === Update Σ1 given Σ2 ===
        sigma2inv =  I / sigma2
        S1 = zeros(n1, n1)

        for t in 1:obs
            Xt = view(X, :, :, t)
            S1 .+= Xt * sigma2inv * Xt'
        end

        sigma1 = S1 / (obs * n2)

        # normalize (identification)
        scale = norm(sigma1)
        sigma2 .*= scale
        sigma1 ./= scale

        # === convergence check ===
        err = max(norm(sigma1 - sigma1_old) / norm(sigma1_old),
                  norm(sigma2 - sigma2_old) / norm(sigma2_old))

        if err < tol
            break
        end
    end

    return (;sigma1, sigma2, iters)
end
