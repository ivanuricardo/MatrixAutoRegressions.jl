
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

@testset "Ridge-VAR number of parameters" begin
    obs = 1000
    p = 2
    n = 12
    dgp = simulate_mar(obs; p=2)
    vecdata = vectorize(dgp.Y)

    true_coef1 = kron(dgp.B[1], dgp.A[1])
    true_coef2 = kron(dgp.B[2], dgp.A[2])
    var_model = VAR(vecdata; p, lambda = 3)

    fit!(var_model)
    n = var_model.n
    num_cov = n * (n + 1) / 2
    estimated_numpar = number_parameters(var_model) - num_cov

    sv = svd(var_model.data).S
    den = sv.^2 .+ var_model.lambda
    s = sum(sv.^2 ./ den)

    x = var_model.data'
    trace_formula = tr(x * inv(x' * x + var_model.lambda * I) * x')

    @test isapprox(estimated_numpar, s)
    @test isapprox(estimated_numpar, trace_formula)

    second_var = VAR(vecdata; p, lambda = 0)
    fit!(second_var)
    @test bic(second_var) > bic(var_model)
end



















