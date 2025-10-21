@testset "Variance OLS" begin
    
    obs = 100
    dgp = simulate_mar(obs)
    matdata = dgp.Y

    model = MAR(matdata, method = :proj)
    fit!(model)

    var_ols = variance_ols(model)


    @test size(var_ols, 1) == 12 * 12
end

@testset "Variance Projection" begin

    obs = 100
    dgp = simulate_mar(obs)
    matdata = dgp.Y

    model = MAR(matdata, method = :proj)
    fit!(model)

    var_ols = variance_ols(model)
    var_proj = variance_proj(model)


    alpha = vec(model.A[1])
    @test size(var_proj, 1) == 12 * 12
end

@testset "Variance ALS" begin
    dgp = simulate_mar(100)
    matdata = dgp.Y

    model = MAR(matdata, method = :als)
    fit!(model)

    var_ols = variance_ols(model)
    var_proj = variance_proj(model)
    var_als = variance_als(model)
    @test size(var_proj) == size(var_als) == size(var_ols)

end

@testset "Variance MLE" begin
    dgp = simulate_mar(10000)
    matdata = dgp.Y

    model_mle = MAR(matdata, method = :mle)
    fit!(model_mle)
    var_mle = asymptotic_variance(model_mle)

    model_als = MAR(matdata, method = :als)
    fit!(model_als)
    var_als = asymptotic_variance(model_als)
    sorted_ev = sort(eigvals(var_als.xi - var_mle.xi))

    # Test whether the variance is smaller for the MLE case
    @test all(sorted_ev[2:end] .> 0)

end
