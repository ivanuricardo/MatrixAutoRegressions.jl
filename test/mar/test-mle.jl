
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
    matdata = dgp.Y
    A_init = dgp.A
    B_init = dgp.B
    Sigma1_init = dgp.Sigma1
    Sigma2_init = dgp.Sigma2
    resp = matdata[:, :, 2:end]
    pred = matdata[:, :, 1:end-1]
    obj_true = mle_objective(resp, pred, A_init, B_init, Sigma1_init, Sigma2_init)

    results = mle(A_init, B_init, Sigma1_init, Sigma2_init, resp, pred)

    # Should be monotonically decreasing
    @test all(diff(results.track_obj) .<= 0)
    @test isapprox(norm(results.A), 1)

end

@testset "mle algorithm correctness" begin
    obs = 1000
    dgp = simulate_mar(obs; snr = 1000)
    matdata = dgp.Y
    A_init = dgp.A
    B_init = dgp.B
    Sigma1_init = dgp.Sigma1
    Sigma2_init = dgp.Sigma2
    resp = matdata[:, :, 2:end]
    pred = matdata[:, :, 1:end-1]
    obj_true = mle_objective(resp, pred, A_init, B_init, Sigma1_init, Sigma2_init)

    results = mle(A_init, B_init, Sigma1_init, Sigma2_init, resp, pred)

    @test norm(results.A - A_init) < 0.1
    @test norm(results.B - B_init) < 0.1

    kron_est = kron(results.B, results.A)
    kron_true = kron(B_init, A_init)
    @test norm(kron_est - kron_true) < 0.1

end
