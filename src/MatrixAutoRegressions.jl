module MatrixAutoRegressions

using LinearAlgebra
using StatsBase
using Statistics
using Distributions
using Random

include("./types.jl")
include("./mar.jl")
export MAR
export fit!
export projection
export als
export mle
export ls_objective
export mle_objective
export update_A
export update_B

include("./utils.jl")
export vectorize
export matricize
export make_companion
export isstable
export ols
export require_fitted
export normalize_slices

include("./simulation.jl")
export generate_mar_coefs
export simulate_mar

end
