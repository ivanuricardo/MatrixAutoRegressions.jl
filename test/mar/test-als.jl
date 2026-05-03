
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

@testset "ls objective p = 2" begin
    # If the snr goes up, the ssr should go down
    obs = 100
    results1 = simulate_mar(obs; snr = 0.2, p = 2)
    A_init1 = results1.A
    B_init1 = results1.B
    matdata1 = results1.Y

    ssr1 = ls_objective(matdata1, A_init1, B_init1; p = 2)

    results2 = simulate_mar(obs; snr = 1, p = 2)
    A_init2 = results2.A
    B_init2 = results2.B
    matdata2 = results2.Y

    ssr2 = ls_objective(matdata2, A_init2, B_init2; p = 2)
    @test ssr1 > ssr2

    @test ssr1 > 0
    @test ssr2 > 0
end

@testset "als algorithm behavior" begin
    obs = 100
    dgp = simulate_mar(obs)
    data = dgp.Y
    A_init = dgp.A
    B_init = dgp.B
    obj_true = ls_objective(dgp.Y, A_init, B_init)

    results = als(data, A_init, B_init)

    # Should be monotonically decreasing
    @test all(diff(results.track_obj) .<= 0)
    @test isapprox(norm(results.A), 1)
    @test norm(results.A[1] - A_init[1]) < 0.1
    obj_est = ls_objective(dgp.Y, results.A, results.B)
    @test obj_est < obj_true

end

@testset "als with multiple lags" begin
    obs = 1000
    p = 2
    maxiter = 100

    dgp = simulate_mar(obs; p)
    data = dgp.Y
    A = dgp.A
    B = dgp.B
    A_init = [randn(3,3), randn(3,3)]
    B_init = [randn(4,4), randn(4,4)]
    obj_true = ls_objective(dgp.Y, A, B)

    results = als(data, A_init, B_init)
    @test norm(abs.(results.A[1]) - abs.(A[1])) < 0.5
    @test norm(abs.(results.B[1]) - abs.(B[1])) < 0.5
    @test isapprox(norm(results.A[1]), 1)
    @test isapprox(norm(results.A[2]), 1)

    model = MAR(data; method = :als, p = 2)
    fit!(model)

    Astack, Bstack = stack_coefs(model)
    @test isapprox(norm(Astack), sqrt(p))

    model = MAR(data; method = :als, p = 3)
    fit!(model)

    Astack, Bstack = stack_coefs(model)
    @test isapprox(norm(Astack), sqrt(3))

    @test norm(abs.(results.A[2]) - abs.(A[2])) < 0.5
    @test norm(abs.(results.B[2]) - abs.(B[2])) < 0.5

    kron_est = kron(results.B[1], results.A[1])
    kron_true = kron(B[1], A[1])
    @test norm(kron_est - kron_true) < 0.5
    obj_est = ls_objective(dgp.Y, results.A, results.B; p=2)

    @test obj_est < obj_true

    # Checking max iter
    results = als(data, A_init, B_init; maxiter=2, tol=1e-30)

end

@testset "als algorithm correctness" begin
    obs = 1000
    dgp = simulate_mar(obs; snr = 1000)
    data = dgp.Y
    A_init = dgp.A
    B_init = dgp.B
    obj_true = ls_objective(data, A_init, B_init)

    results = als(data, A_init, B_init)

    @test norm(abs.(results.A[1]) - abs.(A_init[1])) < 0.5
    @test norm(abs.(results.B[1]) - abs.(B_init[1])) < 0.5

    kron_est = kron(results.B[1], results.A[1])
    kron_true = kron(B_init[1], A_init[1])
    @test norm(kron_est - kron_true) < 0.1

    A_init2 = [randn(3,3)]
    B_init2 = [randn(4,4)]

    results2 = als(data, A_init2, B_init2)

    @test norm(abs.(results2.A[1]) - abs.(A_init[1])) < 0.5
    @test norm(abs.(results2.B[1]) - abs.(B_init[1])) < 0.5

    kron_est = kron(results2.B[1], results2.A[1])
    kron_true = kron(B_init[1], A_init[1])
    @test norm(kron_est - kron_true) < 0.1

end

@testset "als algorithm correctness lag = 2" begin
    obs = 1000
    p = 2
    dgp = simulate_mar(obs; snr=1000, p)
    matdata = dgp.Y
    A_init = dgp.A
    B_init = dgp.B
    obj_true = ls_objective(matdata, dgp.A, dgp.B)

    results = als(matdata, A_init, B_init)

    @test norm(abs.(results.A[1]) - abs.(A_init[1])) < 0.5
    @test norm(abs.(results.B[1]) - abs.(B_init[1])) < 0.5

    @test norm(abs.(results.A[2]) - abs.(A_init[2])) < 0.5
    @test norm(abs.(results.B[2]) - abs.(B_init[2])) < 0.5

    kron_est = kron(results.B[1], results.A[1])
    kron_true = kron(B_init[1], A_init[1])
    @test norm(kron_est - kron_true) < 0.5

    obj_est = ls_objective(matdata, results.A, results.B; p=2)
    @test obj_est < obj_true

    # Trying with projection estimator as init
    ols_est, _ = estimate_var(matdata; p)
    proj_est = projection(ols_est, (3, 4))
    A0 = proj_est.A
    B0 = proj_est.B
    results_different_init = als(matdata, A_init, B_init)

    @test norm(abs.(results.A[1]) - abs.(A_init[1])) < 0.5
    @test norm(abs.(results.B[1]) - abs.(B_init[1])) < 0.5

    @test norm(abs.(results.A[2]) - abs.(A_init[2])) < 0.5
    @test norm(abs.(results.B[2]) - abs.(B_init[2])) < 0.5

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

@testset "Structure lagged data" begin
       
    dgp = simulate_mar(100)
    matdata = dgp.Y

    model = MAR(matdata, method = :als)
    fit!(model)
    data = structure_lagged_data(model)

    @test data == model.data[:, :, 1:end-1]

    dgp = simulate_mar(100; p=2)
    matdata = dgp.Y

    model = MAR(matdata; method = :als, p=2)
    fit!(model)
    data = structure_lagged_data(model)

    n1, n2 = model.dims
    @test size(data) == (2*n1, 2*n2, 100-2)

end

@testset "Reaching maximum iterations" begin

    dgp = simulate_mar(100)
    matdata = dgp.Y

    model = MAR(matdata, method = :als, maxiter=2, tol=1e-12)
    @test_warn "Reached maximum number of iterations" fit!(model)
end

