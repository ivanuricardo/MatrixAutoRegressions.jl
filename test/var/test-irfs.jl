
@testset "VAR IRF and inference" begin
    obs = 1000
    p = 2
    n = 12
    hmax = 14
    dgp = simulate_mar(obs; p=2)
    vecdata = vectorize(dgp.Y)

    true_coef1 = kron(dgp.B[1], dgp.A[1])
    true_coef2 = kron(dgp.B[2], dgp.A[2])
    var_model = VAR(vecdata; p)
    fit!(var_model)

    irfs = reduced_form_irf(var_model; hmax)
    @test size(irfs) == (12, 15)

    theta = irf_ma(var_model; hmax)
    g_var = make_g(theta, var_model.C, hmax)

    @test size(g_var) == (144, 288)

    all_variances = all_irf_variances(var_model, theta; hmax)
    @test length(all_variances) == hmax+1

    shock_idx = 4
    selected_variances = irf_variance(var_model, theta; hmax, shock_idx)
    @test size(selected_variances) == (n, hmax+1)

    full_irfs = irf(var_model; hmax, shock_idx)
    @test size(full_irfs.irfs) == size(full_irfs.irf_se)

end

function make_spd(n; scale::Float64=1.0)
    A = randn(n,n)
    Σ = A*A' + scale*n*I # well-conditioned SPD
    return Σ
end

@testset "Integration tests: get_cholesky_innovation_matrix and irf_ma" begin
    n = 3
    p = 1

    A = [0.2 0.0 0.1;
         0.0 0.1 0.0;
         0.0 0.0 0.3]

    C = [A]

    dgp = simulate_var(100; C=C)

    model = VAR(dgp.Y)  # adjust as needed
    fit!(model)

    B = MatrixAutoRegressions.get_cholesky_innovation_matrix(model)
    @test size(B) == (n, n)
    @test isapprox(B * B', model.Sigma; atol=1e-10, rtol=1e-10)
    @test all(diag(B) .> 0.0)

    theta_reduced = irf_ma(model; hmax=1, ident=:reduced)
    theta_chol = irf_ma(model; hmax=1, ident=:cholesky)

    # For every horizon h, theta_chol[h] should equal theta_reduced[h] * B
    for h in 1:length(theta_reduced)
        @test isapprox(theta_chol[h], theta_reduced[h] * B; atol=1e-12, rtol=1e-12)
    end
end

@testset "Property test: IRF response to structural shock (sanity)" begin

    n = 2
    p = 1
    A = [0.5 0.0; 0.0 0.2]
    C = [A]

    dgp = simulate_var(100; C=C)

    model = VAR(dgp.Y)  # adjust as needed
    fit!(model)
    sigma = model.Sigma

    theta = irf_ma(model; hmax=2, ident=:reduced)
    theta_chol = irf_ma(model; hmax=2, ident=:cholesky)
    B = MatrixAutoRegressions.get_cholesky_innovation_matrix(model)

    shock_idx = 1
    irf_reduced = reduced_form_irf(model; hmax=2, shock_idx=shock_idx, theta=theta, ident=:reduced)
    irf_chol    = reduced_form_irf(model; hmax=2, shock_idx=shock_idx, theta=theta_chol, ident=:cholesky)

    e = zeros(n); e[shock_idx] = 1.0
    for h in 0:2
        manual = theta[h+1] * B * e
        @test isapprox(manual, irf_chol[:, h+1]; atol=1e-12, rtol=1e-12)
        @test isapprox(irf_chol[:, h+1], theta_chol[h+1] * e; atol=1e-12, rtol=1e-12)
    end
end
