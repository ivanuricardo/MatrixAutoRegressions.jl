
mutable struct MAR <: AbstractARModel
    A::Union{Nothing, Vector{<:AbstractMatrix}}
    B::Union{Nothing, Vector{<:AbstractMatrix}}
    p::Int
    Sigma1::Union{Nothing, Matrix{Float64}}
    Sigma2::Union{Nothing, Matrix{Float64}}
    dims::Tuple{Int,Int}
    obs::Int
    method::Symbol
    data::AbstractArray
    maxiter::Int
    tol::Float64
    iters::Union{Nothing, Int}
end

function MAR(data::AbstractArray;
    p::Int=1,
    method::Symbol=:ls,
    A::Union{Nothing, Vector{<:AbstractMatrix}}=nothing,
    B::Union{Nothing, Vector{<:AbstractMatrix}}=nothing,
    maxiter::Int=100,
    tol::Real=1e-6,
    )

    # Demean over time
    demeaned_data = data .- mean(data, dims=3)

    dims = size(data)[1:2]
    obs = size(data, 3)
    iters = nothing

    return MAR(A, B, p, nothing, nothing, dims, obs, method, demeaned_data, maxiter, tol, iters)
end

function fit!(model::MAR)
    ols_est = estimate_var(model.data; model.p)
    proj_est = projection(ols_est, model.dims)

    if model.method == :proj
        model.A = proj_est.A
        model.B = proj_est.B

    elseif model.method == :ls

        A0 = isnothing(model.A) ? proj_est.A : copy(model.A)
        B0 = isnothing(model.B) ? proj_est.B : copy(model.B)

        results = als(model.data, A0, B0; maxiter=model.maxiter, tol=model.tol, p=model.p)

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

function ls_objective(resp::AbstractArray{T}, pred::AbstractArray{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where T
    obs = size(resp, 3)
    ssr = 0
    for i in 1:obs
        residual = resp[:, :, i] - A * pred[:, :, i] * B'
        ssr += sum(abs2, residual)
    end
    return ssr
end

function ls_objective(data::AbstractArray{T}, A::Vector{<:AbstractMatrix}, B::Vector{<:AbstractMatrix}; p=1) where T
    n1, n2, obs = size(data)

    ssr = 0
    for i in (p+1):obs
        pred = zeros(n1, n2)
        for j in 1:p
            pred .+= A[j] * data[:, :, i-j] * B[j]'
        end
        residual = data[:, :, i] - pred
        ssr += sum(abs2, residual)
    end
    
    return ssr
end

function ls_objective(model::MAR)
    require_fitted(model)
    return ls_objective(model.data, model.A, model.B; model.p)
end

function mle_objective(data::AbstractArray{T}, A::Vector{<:AbstractMatrix}, B::Vector{<:AbstractMatrix}, Sigma1::AbstractMatrix{T}, Sigma2::AbstractMatrix{T}; p=1) where T
    n1, n2, obs = size(data)

    ssr = 0
    fac1 = _update_fac(Sigma1)
    fac2 = _update_fac(Sigma2)

    for i in (p+1):obs
        pred = zeros(n1, n2)
        for j in 1:p
            pred .+= A[j] * data[:, :, i-j] * B[j]'
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
    return mle_objective(model.data, model.A, model.B, model.Sigma1, model.Sigma2; model.p)
end

