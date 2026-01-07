module MatrixAutoRegressions

using LinearAlgebra
using StatsBase
using Statistics
using Distributions
using Random
using SparseArrays

import Distributions: loglikelihood
import Base: show

include("./types.jl")
export AbstractARModel
export VAR
export MAR
export residuals
export aic
export bic
export hqc
export ic

include("./var/var.jl")
include("./var/irfs.jl")

include("./var/ols.jl")
export estimate_var
export estimate_var_cov

include("./var/inference.jl")
export avar_sigma
export vech
export revech
export revech_lower
export avar_cholesky
export vech_selection_mat

include("./mar/mar.jl")
export fit!
export predict
export train_test_split
export ls_objective
export coef
export mle_objective

include("./mar/als.jl")
export _update_fac
export update_A
export update_B
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
export make_companion_var
export fit_and_select!
export isstable
export estimate_ols
export require_fitted
export normalize_slices
export mar_eigvals
export var_eigvals
export calculate_residuals
export commutation_matrix
export large_commutation_matrix
export vectorize_kronecker
export number_parameters
export Q_matrix
export specification_test
export define_c

include("./simulation.jl")
export generate_mar_coefs
export generate_var_coefs
export simulate_mar
export simulate_var

include("./mar/irfs.jl")
export irf_ma
export reduced_form_irf
export orthogonalized_irf
export lp_irf
export make_g
export all_irf_variances
export irf

include("./mar/inference.jl")
export structure_lagged_data
export selection_matrix
export stack_coefs
export variance_als
export variance_mle
export variance_proj
export asymptotic_variance
export irf_variance
export construct_gamma
export construct_WW
export std_errors
export get_c_stderr

end
