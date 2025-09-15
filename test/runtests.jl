using MatrixAutoRegressions
using Test, LinearAlgebra


println("Starting tests")
ti = time()

@testset "MatrixAutoRegressions basic test" begin
    @test 1 == 1
end
include("./test-mar.jl")

ti = time() - ti
println("\nTest took total time of:")
println(round(ti / 60, digits=3), " minutes")
