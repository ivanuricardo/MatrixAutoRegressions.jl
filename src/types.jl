abstract type AbstractARModel end

struct MAR <: AbstractARModel
    p::Int
    A::Vector{Matrix{Float64}}
    B::Vector{Matrix{Float64}}
    Sigma1::Matrix{Float64}
    Sigma2::Matrix{Float64}
    dims::Tuple{Int,Int}
    obs::Int
end
