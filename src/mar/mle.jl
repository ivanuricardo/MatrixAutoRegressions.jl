
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

    @inbounds for t in 1:obs_eff
        resp = data[:, :, t + p]

        pred = zero(resp)
        @inbounds for i in 1:p
            pred .+= A[i] * data[:, :, t + p - i] * B[i]'
        end

        residual = resp - pred
        Sigma1 += (residual / Sigma2) * residual'
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

    @inbounds for t in 1:obs_eff
        resp = data[:, :, t + p]

        pred = zero(resp)
        @inbounds for i in 1:p
            pred .+= A[i] * data[:, :, t + p - i] * B[i]'
        end

        residual = resp - pred
        Sigma2 += (residual' / Sigma1) * residual
    end

    return Sigma2 / (n1 * obs_eff)
end

function mle(
    data::AbstractArray{T},
    A::Vector{<:AbstractMatrix},
    B::Vector{<:AbstractMatrix},
    Sigma1::AbstractMatrix,
    Sigma2::AbstractMatrix;
    maxiter::Int=100,
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

        Astack = update_A(resp, pred, Bstack; Sigma2=Sigma2)
        Bstack = update_B(resp, pred, Astack; Sigma1=Sigma1)

        Astack, Bstack = normalize_slices(Astack, Bstack)

        Sigma1 = update_Sigma1(data, A, B, Sigma2)
        Sigma2 = update_Sigma2(data, A, B, Sigma1)

        obj = mle_objective(data, A, B, Sigma1, Sigma2)

        Sigma1, Sigma2 = normalize_slices(Sigma1, Sigma2)

        Sigma1 = Symmetric((Sigma1 + Sigma1')/2)
        Sigma2 = Symmetric((Sigma2 + Sigma2')/2)

        track_obj[i] = abs(obj - obj_old)

        if track_obj[i] < tol
            A = [@view Astack[:, (i-1)*n1+1 : i*n1] for i in 1:p]
            B = [@view Bstack[:, (i-1)*n2+1 : i*n2] for i in 1:p]
            track_obj = track_obj[.!isnan.(track_obj)]
            return (; A, B, Sigma1, Sigma2, track_obj, obj, num_iter)
        end

        if i == maxiter
            A = [@view Astack[:, (i-1)*n1+1 : i*n1] for i in 1:p]
            B = [@view Bstack[:, (i-1)*n2+1 : i*n2] for i in 1:p]
            @warn "Reached maximum number of iterations"
            return (; A, B, Sigma1, Sigma2, track_obj, obj, num_iter)
        end
    end
end

function flipflop_covariance(X::AbstractArray;
    maxiter::Int=100,
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
