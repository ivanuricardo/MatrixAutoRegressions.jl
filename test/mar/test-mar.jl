
@testset "fitting method" begin
    obs = 100
    dgp = simulate_mar(obs)
    matdata = dgp.Y

    model_proj = MAR(matdata, method = :proj)
    @test model_proj.A == nothing
    @test model_proj.B == nothing
    fit!(model_proj)
    @test model_proj.A isa Vector{<:AbstractMatrix}
    @test model_proj.B isa Vector{<:AbstractMatrix}

    model_ls = MAR(matdata, method = :ls)
    @test model_ls.A == nothing
    @test model_ls.B == nothing
    fit!(model_ls)
    @test model_ls.A isa Vector{<:AbstractMatrix}
    @test model_ls.B isa Vector{<:AbstractMatrix}
    @test ls_objective(model_ls) isa Real
    @test mle_objective(model_ls) isa Real

    model_mle = MAR(matdata, method = :mle)
    @test model_mle.A == nothing
    @test model_mle.B == nothing
    fit!(model_mle)
    @test model_mle.A isa Vector{<:AbstractMatrix}
    @test model_mle.B isa Vector{<:AbstractMatrix}
    @test ls_objective(model_mle) isa Real
    @test mle_objective(model_mle) isa Real

end

