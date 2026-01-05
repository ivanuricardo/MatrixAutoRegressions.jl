
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

