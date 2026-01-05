
@testset "mle objective" begin

    # If the snr goes up, the likelihood should go up
    obs = 100
    dgp1 = simulate_mar(obs; snr = 0.2)
    A_init1 = dgp1.A
    B_init1 = dgp1.B
    Sigma1_init1 = dgp1.Sigma1
    Sigma2_init1 = dgp1.Sigma2
    matdata1 = dgp1.Y

    ssr1 = mle_objective(matdata1, A_init1, B_init1, Sigma1_init1, Sigma2_init1)

    dgp2 = simulate_mar(obs; snr = 1)
    A_init2 = dgp2.A
    B_init2 = dgp2.B
    Sigma1_init2 = dgp1.Sigma1
    Sigma2_init2 = dgp1.Sigma2
    matdata2 = dgp2.Y

    ssr2 = mle_objective(matdata2, A_init2, B_init2, Sigma1_init2, Sigma2_init2)
    @test ssr1 < ssr2

end

@testset "mle algorithm behavior" begin
    obs = 100
    dgp = simulate_mar(obs)
    data = dgp.Y
    A_init = dgp.A
    B_init = dgp.B
    Sigma1_init = dgp.Sigma1
    Sigma2_init = dgp.Sigma2
    obj_true = mle_objective(data, A_init, B_init, Sigma1_init, Sigma2_init)

    results = mle(data, A_init, B_init, Sigma1_init, Sigma2_init)

    # Should be monotonically decreasing
    @test all(diff(results.track_obj) .<= 0)
    @test isapprox(norm(results.A), 1)

end

@testset "mle algorithm correctness" begin
    obs = 1000
    dgp = simulate_mar(obs)
    data = dgp.Y
    obj_true = mle_objective(data, dgp.A, dgp.B, dgp.Sigma1, dgp.Sigma2)

    results = mle(data, dgp.A, dgp.B, dgp.Sigma1, dgp.Sigma2)
    results.obj

    @test norm(results.A[1] - dgp.A[1]) < 0.1
    @test norm(results.B[1] - dgp.B[1]) < 0.1

    kron_est = kron(results.B[1], results.A[1])
    kron_true = kron(dgp.B[1], dgp.A[1])
    @test norm(kron_est - kron_true) < 0.5

    @test norm(results.Sigma1 - dgp.Sigma1) < 0.5
    @test norm(results.Sigma2 - dgp.Sigma2) < 0.5

    obj_est = mle_objective(data, results.A, results.B, results.Sigma1, results.Sigma2)
    @test obj_true < obj_est

end

@testset "mle algorithm projection initialization" begin
    obs = 1000
    dgp = simulate_mar(obs)
    data = dgp.Y
    obj_true = mle_objective(data, dgp.A, dgp.B, dgp.Sigma1, dgp.Sigma2)

    ols_est, cov_est = estimate_var(data)
    proj_est = projection(ols_est, (3,4))
    proj_cov = projection([cov_est], (3,4))

    results = mle(data, proj_est.A, proj_est.B, proj_cov.A[1], proj_cov.B[1])
    results.obj

    @test norm(results.A[1] - dgp.A[1]) < 0.1
    @test norm(results.B[1] - dgp.B[1]) < 0.1

    kron_est = kron(results.B[1], results.A[1])
    kron_true = kron(dgp.B[1], dgp.A[1])
    @test norm(kron_est - kron_true) < 0.1

    @test norm(results.Sigma1 - dgp.Sigma1) < 0.5
    @test norm(results.Sigma2 - dgp.Sigma2) < 0.5

    obj_est = mle_objective(data, results.A, results.B, results.Sigma1, results.Sigma2)
    @test obj_true < obj_est

end

@testset "mle algorithm correctness p = 2" begin
    obs = 10000
    dgp = simulate_mar(obs; snr = 1000, p=2)
    data = dgp.Y
    A_init = dgp.A
    B_init = dgp.B
    Sigma1_init = I(3)
    Sigma2_init = I(4)
    obj_true = mle_objective(data, A_init, B_init, Sigma1_init, Sigma2_init)

    results = mle(data, A_init, B_init, Sigma1_init, Sigma2_init)

    @test norm(results.A[1] - A_init[1]) < 1.0
    @test norm(results.B[1] - B_init[1]) < 1.0

    @test norm(results.A[2] - A_init[2]) < 1.0
    @test norm(results.B[2] - B_init[2]) < 1.0

    kron_est = kron(results.B[1], results.A[1])
    kron_true = kron(B_init[1], A_init[1])
    @test norm(kron_est - kron_true) < 0.1

    @test norm(results.Sigma1 - dgp.Sigma1) < 0.1
    @test norm(results.Sigma2 - dgp.Sigma2) < 0.1

    obj_est = mle_objective(data, results.A, results.B, results.Sigma1, results.Sigma2)
    @test obj_true < obj_est

end
