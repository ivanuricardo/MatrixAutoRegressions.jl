
function update_Sigma1(
    data::AbstractArray{T}, 
    A::Vector{<:AbstractMatrix{T}}, 
    B::Vector{<:AbstractMatrix{T}}, 
    Sigma2::AbstractMatrix{T}
) where T

    p = length(A)
    @assert length(B) == p "length(A) and length(B) must match"

    n1, n2, obs = size(data)
    obs_eff = obs - p
    Sigma1 = zeros(T, n1, n1)

    fac = _update_fac(Sigma2)

    @inbounds for t in 1:obs_eff
        resp = data[:, :, t + p]

        fit = zero(resp)
        @inbounds for i in 1:p
            fit .+= A[i] * data[:, :, t + p - i] * B[i]'
        end

        residual = resp - fit
        Sigma1 += (residual / fac) * residual'
    end

    return Sigma1 / (n2 * obs_eff)
end

function update_Sigma2(
    data::AbstractArray{T}, 
    A::Vector{<:AbstractMatrix{T}}, 
    B::Vector{<:AbstractMatrix{T}}, 
    Sigma1::AbstractMatrix{T}
) where T

    p = length(A)

    n1, n2, obs = size(data)
    obs_eff = obs - p
    Sigma2 = zeros(T, n2, n2)

    fac = _update_fac(Sigma1)

    @inbounds for t in 1:obs_eff
        resp = data[:, :, t + p]

        fit = zero(resp)
        @inbounds for i in 1:p
            fit .+= A[i] * data[:, :, t + p - i] * B[i]'
        end

        residual = resp - fit
        Sigma2 += (residual' / fac) * residual
    end

    return Sigma2 / (n1 * obs_eff)
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

        B = update_B(resp, pred, A; Sigma1)
        A = update_A(resp, pred, B; Sigma2)
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
