module MatrixAutoRegressions

using LinearAlgebra
using StatsBase
using Statistics
using Distributions

include("./types.jl")
include("./utils.jl")
include("./mar.jl")
export projection
export als


end
