module MatrixAutoRegressions

using LinearAlgebra
using StatsBase
using Statistics
using Distributions

include("./types.jl")
include("./mar.jl")
export projection
export als

include("./utils.jl")
export vectorize
export matricize

end
