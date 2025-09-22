
@testset "fitting method" begin
    obs = 100
    dgp = simulate_mar(obs)
    matdata = dgp.Y

    model = MAR(matdata, method = :proj)
    @test model.A == nothing
    @test model.B == nothing
    fit!(model)
    @test model.A isa AbstractVecOrMat
    @test model.B isa AbstractVecOrMat

    model = MAR(matdata, method = :ls)
    @test model.A == nothing
    @test model.B == nothing
    fit!(model)
    @test model.A isa AbstractVecOrMat
    @test model.B isa AbstractVecOrMat
    @test ls_objective(model) isa Real
    @test mle_objective(model) isa Real

    model = MAR(matdata, method = :mle)
    @test model.A == nothing
    @test model.B == nothing
    fit!(model)
    @test model.A isa AbstractVecOrMat
    @test model.B isa AbstractVecOrMat
    @test ls_objective(model) isa Real
    @test mle_objective(model) isa Real

end

