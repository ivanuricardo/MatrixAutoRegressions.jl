@testset "Nearest Kronecker Product" begin
    n1 = 4
    n2 = 3
    A = randn(n1,n1)
    B = randn(n2,n2)

    phi = kron(B, A)

    est = projection(phi, n1, n2)
    true_a = A / norm(A)
    est_a = est.A / norm(est.A)
    @test isapprox(true_a, est_a; atol=1e-8) || isapprox(true_a, -est_a; atol=1e-8)

    true_b = B / norm(B)
    est_b = est.B / norm(est.B)
    @test isapprox(true_b, est_b; atol=1e-8) || isapprox(true_b, -est_b; atol=1e-8)

    true_product = kron(B, A)
    est_product = kron(est.B, est.A)
    @test isapprox(est_product, true_product, atol=1e-08)

end

@testset "ls objective" begin

    # If the snr goes up, the ssr should go down
    obs = 100
    results1 = simulate_mar(obs; snr = 0.2)
    A_init1 = results1.A
    B_init1 = results1.B
    matdata1 = results1.Y

    ssr1 = ls_objective(matdata1, A_init1, B_init1)

    results2 = simulate_mar(obs; snr = 1)
    A_init2 = results2.A
    B_init2 = results2.B
    matdata2 = results2.Y

    ssr2 = ls_objective(matdata2, A_init2, B_init2)
    @test ssr1 > ssr2

    @test ssr1 > 0
    @test ssr2 > 0

end

@testset "als algorithm behavior" begin
    obs = 100
    dgp = simulate_mar(obs)
    matdata = dgp.Y
    A_init = dgp.A
    B_init = dgp.B
    resp = matdata[:, :, 2:end]
    pred = matdata[:, :, 1:end-1]
    obj_true = ls_objective(resp, pred, A_init, B_init)

    results = als(A_init, B_init, resp, pred)

    # Should be monotonically decreasing
    @test all(diff(results.track_obj) .<= 0)
    @test isapprox(norm(results.A), 1)

end

@testset "als algorithm correctness" begin
    obs = 1000
    dgp = simulate_mar(obs; snr = 1000)
    matdata = dgp.Y
    A_init = dgp.A
    B_init = dgp.B
    resp = matdata[:, :, 2:end]
    pred = matdata[:, :, 1:end-1]
    obj_true = ls_objective(resp, pred, A_init, B_init)

    results = als(A_init, B_init, resp, pred)

    @test norm(results.A - A_init) < 0.1
    @test norm(results.B - B_init) < 0.1

    kron_est = kron(results.B, results.A)
    kron_true = kron(B_init, A_init)
    @test norm(kron_est - kron_true) < 0.1

end

@testset "update_A/B scalar case" begin
    resp = reshape([2.0, 4.0], 1, 1, 2)   # size (1,1,2)
    pred = reshape([1.0, 2.0], 1, 1, 2)
    A = [1.0]
    B = [1.0]

    B_new = update_B(resp, pred, A)
    A_new = update_A(resp, pred, B)

    # In scalar case, they reduce to weighted least squares ratios
    expected_B = (2*1*1 + 4*1*2) / (1*1*1 + 2*1*2)   # hand-computed
    expected_A = (2*1*1 + 4*1*2) / (1*1*1 + 2*1*2)

    @test isapprox(B_new[1], expected_B; atol=1e-12)
    @test isapprox(A_new[1], expected_A; atol=1e-12)
end

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
