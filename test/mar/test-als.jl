
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
    obs = 1000
    p = 2
    maxiter = 100

    dgp = simulate_mar(obs; p)
    data = dgp.Y
    A = dgp.A
    B = dgp.B
    A_init = [randn(3,3), randn(3,3)]
    B_init = [randn(4,4), randn(4,4)]
    obj_true = ls_objective(dgp.Y, A, B; p=2)

    results = als(A_init, B_init, data; p=2)
    @test norm(abs.(results.A[1]) - abs.(A[1])) < 0.1
    @test norm(abs.(results.B[1]) - abs.(B[1])) < 0.1

    @test norm(abs.(results.A[2]) - abs.(A[2])) < 0.1
    @test norm(abs.(results.B[2]) - abs.(B[2])) < 0.1

    kron_est = kron(results.B[1], results.A[1])
    kron_true = kron(B[1], A[1])
    @test norm(kron_est - kron_true) < 0.5
    obj_est = ls_objective(dgp.Y, results.A, results.B; p=2)

    @test obj_est < obj_true

end

@testset "als algorithm correctness" begin
    obs = 1000
    dgp = simulate_mar(obs; snr = 1000)
    matdata = dgp.Y
    A_init = dgp.A
    B_init = dgp.B
    resp = matdata[:, :, 2:end]
    pred = matdata[:, :, 1:end-1]
    obj_true = ls_objective(resp, pred, A_init[1], B_init[1])

    results = als(A_init[1], B_init[1], resp, pred)

    @test norm(abs.(results.A) - abs.(A_init[1])) < 0.5
    @test norm(abs.(results.B) - abs.(B_init[1])) < 0.5

    kron_est = kron(results.B, results.A)
    kron_true = kron(B_init[1], A_init[1])
    @test norm(kron_est - kron_true) < 0.5

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

@testset "residuals given index" begin

    # simple case: identity transforms
    m, n, obs, p = 2, 2, 6, 2
    data = randn(m, n, obs)

    # A[i] = I, B[i] = I
    A = [Matrix(I, m, m) for _ in 1:p]
    B = [Matrix(I, n, n) for _ in 1:p]

    # case 1: skip index 1 → should subtract only lag 2 contribution
    res1 = residual_given_idx(data, A, B, 1)
    @test size(res1) == (m, n, obs - p)

    # sanity check: if A = 0 matrices, residuals = response
    A0 = [zeros(m, m) for _ in 1:p]
    B0 = [zeros(n, n) for _ in 1:p]
    res0 = residual_given_idx(data, A0, B0, 1)
    @test res0 ≈ data[:,:,p+1:end]

end
