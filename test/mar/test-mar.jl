
@testset "fitting method" begin
    obs = 100
    dgp = simulate_mar(obs)
    matdata = dgp.Y

    model_proj = MAR(matdata, method = :proj)
    @test isempty(model_proj.A)
    @test isempty(model_proj.B)
    fit!(model_proj)
    @test model_proj.A isa Vector{<:AbstractMatrix}
    @test model_proj.B isa Vector{<:AbstractMatrix}

    model_ls = MAR(matdata, method = :als)
    @test isempty(model_ls.A)
    @test isempty(model_ls.B)
    fit!(model_ls)
    @test model_ls.A isa Vector{<:AbstractMatrix}
    @test model_ls.B isa Vector{<:AbstractMatrix}
    @test ls_objective(model_ls) isa Real
    @test mle_objective(model_ls) isa Real

    model_mle = MAR(matdata, method = :mle)
    @test isempty(model_mle.A)
    @test isempty(model_mle.B)
    fit!(model_mle)
    @test model_mle.A isa Vector{<:AbstractMatrix}
    @test model_mle.B isa Vector{<:AbstractMatrix}
    @test ls_objective(model_mle) isa Real
    @test mle_objective(model_mle) isa Real

end

@testset "Train Test Split" begin

    sim = simulate_mar(100; n1=3, n2=4, p=2)
    model = MAR(sim.Y, p=2)
    fit!(model)
    train_data, test_data = train_test_split(model; h=5)
    @test size(test_data, 3) == 5
    @test size(train_data, 3) == 95

end

@testset "forecasts" begin
    sim = simulate_mar(200; n1=3, n2=4, p=2, snr=1000)
    model = MAR(sim.Y, p=2)
    fit!(model)

    train_data, test_data = train_test_split(model; h=5)
    model.data = train_data

    Yhat = predict(model; h=5)  # 5-step-ahead forecast
    @test size(Yhat) == (3,4,5)

    @test norm(Yhat - test_data) < 0.5


end

