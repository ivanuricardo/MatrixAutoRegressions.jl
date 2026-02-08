
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

@testset "Pope-Killian bias" begin

    sims = 10000
    p = 2
    n = 4
    obs = 50

    dgp = simulate_var(obs; p, n)
    Y = dgp.Y
    true_coef = dgp.C
    true_Sigma = dgp.Sigma
    abs.(var_eigvals(true_coef))

    average_before1 = fill(NaN, n, n, sims)
    average_after1 = fill(NaN, n, n, sims)
    average_before2 = fill(NaN, n, n, sims)
    average_after2 = fill(NaN, n, n, sims)
    for i in 1:sims
        dgp = simulate_var(obs; p, n, C=true_coef, Sigma=true_Sigma)
        Y = dgp.Y

        var_model = VAR(Y; p)
        fit!(var_model)
        A_hat = var_model.C

        bias_correction!(var_model)
        A_corrected = var_model.C

        # e.g., only the first lag
        average_before1[:, :, i] = A_hat[1]
        average_after1[:, :, i] = A_corrected[1]

        average_before2[:, :, i] = A_hat[2]
        average_after2[:, :, i] = A_corrected[2]
    end

    magnitude_before1 = norm(mean(average_before1, dims = 3) .- true_coef[1])
    magnitude_after1 = norm(mean(average_after1, dims = 3) .- true_coef[1])
    @test magnitude_before1 > magnitude_after1

    magnitude_before2 = norm(mean(average_before2, dims = 3) .- true_coef[2])
    magnitude_after2 = norm(mean(average_after2, dims = 3) .- true_coef[2])
    @test magnitude_before2 > magnitude_after2

end

