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

@testset "als algorithm" begin
    obs = 100
    dgp = simulate_mar(obs)
    matdata = dgp.Y
    A_init = dgp.A
    B_init = dgp.B
    resp = matdata[:, :, 2:end]
    pred = matdata[:, :, 1:end-1]
    maxiter = 100
    tol = 1e-06
    obj_true = ls_objective(resp, pred, A_init, B_init)

    results = als(A_init, B_init, resp, pred)
    results.A
    @test all(diff(results.track_obj) .<= 0)



end













