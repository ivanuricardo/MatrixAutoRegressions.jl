
@testset "VAR estimation" begin

    obs = 100000
    p = 2
    dgp = simulate_mar(obs; p)
    matdata = dgp.Y
    vecdata = vectorize(matdata)
    true_coef1 = kron(dgp.B[1], dgp.A[1])
    true_coef2 = kron(dgp.B[2], dgp.A[2])

    est_coef1, Sigma_u1 = estimate_var(matdata; p)
    est_coef2, Sigma_u2 = estimate_var(vecdata; p)
    @test est_coef1[1] == est_coef2[1]
    @test est_coef1[2] == est_coef2[2]
    @test Sigma_u1 == Sigma_u2
    @test norm(est_coef1[1] - true_coef1) < 0.1
    @test norm(est_coef1[2] - true_coef2) < 0.1

    full_sigma = kron(dgp.Sigma2, dgp.Sigma1)
    @test norm(full_sigma - Sigma_u1) < 0.1

end

@testset "Standard errors" begin

    obs = 100
    p = 2
    dgp = simulate_mar(obs; p)
    vecdata = vectorize(dgp.Y)

    true_coef1 = kron(dgp.B[1], dgp.A[1])
    true_coef2 = kron(dgp.B[2], dgp.A[2])
    var_model = VAR(vecdata; p)
    fit!(var_model)
    se = std_errors(var_model)

    mar_model = MAR(dgp.Y; p=2)
    fit!(mar_model)
    mar_se = std_errors(mar_model)
    C_std = mar_se.C_stderr[1]
    @test norm(C_std) < norm(se)

end
