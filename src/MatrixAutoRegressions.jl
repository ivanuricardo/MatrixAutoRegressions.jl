module MatrixAutoRegressions

using LinearAlgebra
using StatsBase
using Statistics
using Distributions
using Random
using SparseArrays

include("./types.jl")
include("./mar/mar.jl")
export MAR
export fit!
export predict
export train_test_split
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
export permutation_matrix

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
export calculate_residuals
export residuals

include("./simulation.jl")
export generate_mar_coefs
export simulate_mar

include("./mar/irfs.jl")
export irf
export local_projection

include("./mar/inference.jl")
export variance_als
export variance_ols
export variance_mle
export variance_proj
export asymptotic_variance

end
