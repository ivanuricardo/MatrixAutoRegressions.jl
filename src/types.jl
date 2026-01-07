abstract type AbstractARModel end

mutable struct VAR <: AbstractARModel
    C::Union{Nothing, Vector{<:AbstractMatrix}}
    n::Int
    p::Int
    Sigma::Union{Nothing, Matrix{Float64}}
    obs::Int
    method::Symbol
    data::AbstractArray
    residuals::Union{Nothing, Matrix{Float64}}
end

"""
# MAR

Multivariate AutoRegressive (MAR) model container.

# Fields
- `A::Vector{<:AbstractMatrix}` — Left coefficient matrices for each lag (length `p`). Each `A[i]` is `dims[1] × dims[1]`.
- `B::Vector{<:AbstractMatrix}` — Right coefficient matrices for each lag (length `p`). Each `B[i]` is `dims[2] × dims[2]`.
- `C::Vector{<:AbstractMatrix}` — Restricted VAR coefficient given by B ⊗ A. Each `C[i]` is `dims[1] dims[2] × dims[1] dims[2]`.
- `p::Int` — Lag order.
- `Sigma::Union{Nothing, Matrix{Float64}}` — Estimated innovation covariance (full `dims[1]*dims[2]` representation or the appropriately shaped covariance).
- `Sigma1::Union{Nothing, Matrix{Float64}}` — Rowwise covariance (if computed).
- `Sigma2::Union{Nothing, Matrix{Float64}}` — Columnwise covariance (if computed).
- `dims::Tuple{Int,Int}` — Dimensions of the panel / matrix time series `(nrow, ncol)`.
- `obs::Int` — Effective number of observations (time axis length minus `p`).
- `method::Symbol` — Estimation method used, e.g. `:mle`, `:als`, or `:proj`.
- `data::AbstractArray` — Original data array with shape `(nrow, ncol, T)`.
- `maxiter::Int` — Maximum iterations used when fitting.
- `tol::Float64` — Convergence tolerance used when fitting.
- `iters::Union{Nothing, Int}` — Number of iterations actually performed (set after fitting).
- `residuals::Union{Nothing, AbstractArray{Float64}}` — Residual array with same `dims` over `obs` if computed.

# Notes
This struct is a lightweight container only. Use `fit!` (see below) to populate estimated parameters and residuals.
"""
mutable struct MAR <: AbstractARModel
    A::Vector{<:AbstractMatrix}
    B::Vector{<:AbstractMatrix}
    C::Vector{<:AbstractMatrix}
    p::Int
    Sigma::Union{Nothing, Matrix{Float64}}
    Sigma1::Union{Nothing, Matrix{Float64}}
    Sigma2::Union{Nothing, Matrix{Float64}}
    dims::Tuple{Int,Int}
    obs::Int
    method::Symbol
    data::AbstractArray
    maxiter::Int
    tol::Float64
    iters::Union{Nothing, Int}
    residuals::Union{Nothing, AbstractArray{Float64}}
end

function residuals(model::AbstractARModel)
    return calculate_residuals(model)
end

function loglikelihood(model::VAR)
    require_fitted(model)
    E = residuals(model)
    n, obs = size(E)

    const_term = - (obs * n / 2) * log(2π)

    Sigma = Symmetric(model.Sigma)
    ch = cholesky(Sigma)
    # omit dividing by 2 because its a cholesky factor
    logdetsigma = obs * sum(log.(abs.(diag(ch.L))))
    sol = ch \ E
    quad = sum(sol .* E)
    return const_term - logdetsigma - 0.5 * quad
end

function loglikelihood(model::MAR)
    require_fitted(model)
    if model.method != :mle
        error("Method must be Maximum Likelihood!")
    end
    if model.p == 0
        E = model.residuals
    else
        E = residuals(model)
    end

    n1, n2, obs = size(E)

    const_term = - (obs * n1 * n2) / 2 * log(2π)

    sigma1 = Symmetric(model.Sigma1)
    sigma2 = Symmetric(model.Sigma2)

    ch1 = cholesky(sigma1)
    ch2 = cholesky(sigma2)

    logdet1 = 2 * sum(log, diag(ch1.L))
    logdet2 = 2 * sum(log, diag(ch2.L))

    term_logdet = (obs * n2) / 2 * logdet1 + (obs * n1) / 2 * logdet2

    sum_tr = 0.0
    for t in 1:obs
        Et = @view E[:, :, t]
        U = ch1.L \ Et
        W = (ch2.L \ U')'
        sum_tr += sum(abs2, W)
    end
    term_quad = 0.5 * sum_tr
    return const_term - term_logdet - term_quad

end

function aic(model::AbstractARModel)
    obs = model.obs
    k = number_parameters(model)
    if model.method == :mle
        return 2 * k - 2 * loglikelihood(model)
    end
    ch = cholesky(model.Sigma)
    logdetterm = 2 * sum(log, diag(ch.L))
    return 2 * k + obs * logdetterm
end

function bic(model::AbstractARModel)
    obs = model.obs
    k = number_parameters(model)
    if model.method == :mle
        return k * log(obs) - 2 * loglikelihood(model)
    end
    ch = cholesky(model.Sigma)
    logdetterm = 2 * sum(log, diag(ch.L))
    return k * log(obs) + obs * logdetterm
end

function hqc(model::AbstractARModel)
    k = number_parameters(model)
    obs = model.obs
    if model.method == :mle
        return k * 2 * log(log(obs)) - 2 * loglikelihood(model)
    end
    ch = cholesky(model.Sigma)
    logdetterm = 2 * sum(log, diag(ch.L))
    return k * log(log(obs)) + obs * logdetterm
end

function ic(model::AbstractARModel; ic_type::Symbol=:bic)
    if ic_type == :bic
        return bic(model)
    elseif ic_type == :aic
        return aic(model)
    elseif ic_type == :hqc
        return hqc(model)
    else
        error("No valid IC type given!")
    end
end


