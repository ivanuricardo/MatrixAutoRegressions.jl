@testset "Nearest Kronecker Product" begin
    n1 = 4
    n2 = 3
    A1 = randn(n1,n1)
    B1 = randn(n2,n2)

    phi = [kron(B1, A1)]

    est = projection(phi, (n1, n2))
    true_a1 = A1 / norm(A1)
    @test isapprox(true_a1, est.A[1]; atol=1e-8) || isapprox(true_a1, -est.A[1]; atol=1e-8)

    true_b1 = B1 * norm(A1)
    @test isapprox(true_b1, est.B[1]; atol=1e-8) || isapprox(true_b1, -est.B[1]; atol=1e-8)

    true_product = kron(B1, A1)
    est_product = kron(est.B[1], est.A[1])
    @test isapprox(est_product, true_product, atol=1e-08)

    # Adding another coef as a lag
    A2 = randn(n1, n1)
    B2 = randn(n2, n2)

    phi2 = kron(B2, A2)
    push!(phi, phi2)

    est2 = projection(phi, (n1, n2))
    true_a2 = A2 / norm(A2)
    @test isapprox(true_a1, est2.A[1]; atol=1e-8) || isapprox(true_a1, -est2.A[1]; atol=1e-8)
    @test isapprox(true_a2, est2.A[2]; atol=1e-8) || isapprox(true_a2, -est2.A[2]; atol=1e-8)

    true_b2 = B2 * norm(A2)
    @test isapprox(true_b1, est2.B[1]; atol=1e-8) || isapprox(true_b1, -est2.B[1]; atol=1e-8)
    @test isapprox(true_b2, est2.B[2]; atol=1e-8) || isapprox(true_b2, -est2.B[2]; atol=1e-8)

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

@testset "proj algorithm behavior" begin
    obs = 10000
    dgp = simulate_mar(obs)
    matdata = dgp.Y
    A_init = dgp.A
    B_init = dgp.B

    model = MAR(matdata; method = :proj)
    fit!(model)

    @test norm(model.A[1] - A_init[1]) < 0.1
    @test norm(model.B[1] - B_init[1]) < 0.1

    dgp2 = simulate_mar(obs; p=2)
    matdata2 = dgp2.Y
    A2 = dgp2.A
    B2 = dgp2.B

    model2 = MAR(matdata2; method = :proj, p = 2)
    fit!(model2)

    kron1 = kron(B2[1], A2[1])
    est_kron1 = kron(model2.B[1], model2.A[1])
    @test norm(kron1 - est_kron1) < 0.1

    kron2 = kron(B2[2], A2[2])
    est_kron2 = kron(model2.B[2], model2.A[2])
    @test norm(kron2 - est_kron2) < 0.1

end

@testset "als algorithm behavior" begin
    obs = 100
    dgp = simulate_mar(obs)
    matdata = dgp.Y
    A_init = dgp.A
    B_init = dgp.B
    obj_true = ls_objective(dgp.Y, A_init, B_init)

    resp = matdata[:, :, 2:end]
    pred = matdata[:, :, 1:end]
    results = als(A_init[1], B_init[1], resp, pred)

    # Should be monotonically decreasing
    @test all(diff(results.track_obj) .<= 0)
    @test isapprox(norm(results.A), 1)
    @test norm(results.A - A_init[1]) < 0.1
    obj_est = ls_objective(dgp.Y, [results.A], [results.B])
    @test obj_est < obj_true

end

@testset "als with multiple lags" begin
    obs = 100
    dgp = simulate_mar(obs)
    matdata = dgp.Y
    A_init = dgp.A
    B_init = dgp.B
    obj_true = ls_objective(dgp.Y, A_init, B_init)

    results = als(A_init, B_init, data)
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







