
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
    obj = mle_objective(data, A, B, Sigma1, Sigma2)

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

        Sigma1 = update_Sigma1(data, A, B, Sigma2)
        Sigma2 = update_Sigma2(data, A, B, Sigma1)

        obj = mle_objective(data, A, B, Sigma1, Sigma2)
        norm_A = norm(A)
        A = A / norm_A
        B = B * norm_A

        norm_Sigma1 = norm(Sigma1)
        Sigma1 = Sigma1 / norm_Sigma1
        Sigma2 = Sigma2 * norm_Sigma1

        Sigma1 = Symmetric((Sigma1 + Sigma1')/2)
        Sigma2 = Symmetric((Sigma2 + Sigma2')/2)

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
