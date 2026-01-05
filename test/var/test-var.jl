
@testset "Fitting the VAR" begin
    obs = 1000
    p = 2
    dgp = simulate_mar(obs; p=2)
    vecdata = vectorize(dgp.Y)

    true_coef1 = kron(dgp.B[1], dgp.A[1])
    true_coef2 = kron(dgp.B[2], dgp.A[2])
    var_model = VAR(vecdata; p)
    fit!(var_model)
    @test size(var_model.C[1]) == (12, 12)
    @test size(var_model.C[2]) == (12, 12)
    @test size(var_model.Sigma) == (12, 12)

end

@testset "VAR variances" begin
    obs = 1000
    p = 2
    n = 12
    dgp = simulate_mar(obs; p=2)
    vecdata = vectorize(dgp.Y)

    true_coef1 = kron(dgp.B[1], dgp.A[1])
    true_coef2 = kron(dgp.B[2], dgp.A[2])
    var_model = VAR(vecdata; p)
    fit!(var_model)

    avar = asymptotic_variance(var_model)
    @test size(avar) == (n*n*p, n*n*p)

end
