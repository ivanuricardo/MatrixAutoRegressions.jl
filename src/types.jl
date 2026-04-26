abstract type AbstractARModel end

abstract type BiasCorrection end

mutable struct VAR <: AbstractARModel
    C::Union{Nothing, Vector{<:AbstractMatrix}}
    n::Int
    p::Int
    Sigma::Union{Nothing, Matrix{Float64}}
    obs::Int
    method::Symbol
    data::AbstractArray
    residuals::Union{Nothing, Matrix{Float64}}
    lambda::Float64
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

struct Analytical <: BiasCorrection end

struct Bootstrap <: BiasCorrection
    n_boot::Int
    ci_level::Float64
    seed::Union{Nothing, Int}
end
