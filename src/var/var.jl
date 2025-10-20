
mutable struct VAR <: AbstractARModel
    A::AbstractMatrix
    p::Int
    Sigma::Union{Nothing, Matrix{Float64}}
    dims::Tuple{Int,Int}
    obs::Int
    method::Symbol
    data::AbstractArray
    maxiter::Int
    tol::Float64
    iters::Union{Nothing, Int}
end
