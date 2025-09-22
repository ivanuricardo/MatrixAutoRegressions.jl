module MatrixAutoRegressions

using LinearAlgebra
using StatsBase
using Statistics
using Distributions
using Random

include("./types.jl")
include("./mar/mar.jl")
export MAR
export fit!
export ls_objective
export mle_objective

include("./mar/als.jl")
export _update_fac
export update_A
export update_B
export residual_given_idx
export als

include("./mar/mle.jl")
export update_Sigma1
export update_Sigma2
export mle

include("./mar/proj.jl")
export projection

include("./utils.jl")
export vectorize
export calculate_residuals
export matricize
export make_companion
export isstable
export estimate_ols
export require_fitted
export normalize_slices
export mar_eigvals
export estimate_var

include("./simulation.jl")
export generate_mar_coefs
export simulate_mar

end
