using MatrixAutoRegressions
using Test, LinearAlgebra, Random, Distributions
Random.seed!(20250915)

println("Starting tests")
ti = time()

@testset "MatrixAutoRegressions basic test" begin
    @test 1 == 1
end
include("./mar/test-mar.jl")
include("./mar/test-inference.jl")
include("./mar/test-als.jl")
include("./mar/test-proj.jl")
include("./mar/test-mle.jl")
include("./mar/test-irf.jl")

include("./var/test-inference.jl")
include("./var/test-irfs.jl")
include("./var/test-var.jl")
include("./var/test-ols.jl")
include("./var/test-bias_correction.jl")

include("./test-simulation.jl")
include("./test-utils.jl")
include("./test-types.jl")

ti = time() - ti
println("\nTest took total time of:")
println(round(ti / 60, digits=3), " minutes")
