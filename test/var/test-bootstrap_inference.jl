
@testset "enforce_stationarity" begin
    n = 2
    p = 1

    @testset "already stationary — no shrinkage" begin
        C_hat = [0.5 * I(n)]
        bias_mats = [0.1 * I(n)]
        C_corrected = enforce_stationarity(C_hat, bias_mats; p)
        # bias is small, corrected should equal C_hat - bias exactly (δ = 1)
        @test C_corrected[1] ≈ C_hat[1] - bias_mats[1]
    end

    @testset "correction would be nonstationary — shrinks δ" begin
        C_hat = [0.8 * I(n)]
        # a bias so large that C_hat - bias has roots outside unit circle
        bias_mats = [-0.5 * I(n)]  # corrected would be 1.3*I
        C_corrected = enforce_stationarity(C_hat, bias_mats; p)
        # result must be stationary
        A_big = hcat(C_corrected...)
        comp = make_companion(A_big)
        @test maximum(abs.(eigvals(comp))) < 1.0
        # but correction was shrunk, so it's not C_hat - bias_mats
        @test !(C_corrected[1] ≈ C_hat[1] - bias_mats[1])
    end

    @testset "uncorrectable — returns original" begin
        # C_hat itself is already on the boundary
        C_hat = [1.0 * I(n)]
        bias_mats = [zeros(n, n)]
        C_corrected = enforce_stationarity(C_hat, bias_mats; p)
        @test C_corrected[1] ≈ C_hat[1]
    end

    @testset "multilag p=2" begin
        p2 = 2
        C_hat = [0.5 * I(n), 0.2 * I(n)]
        bias_mats = [0.05 * I(n), 0.02 * I(n)]
        C_corrected = enforce_stationarity(C_hat, bias_mats; p=p2)
        A_big = hcat(C_corrected...)
        comp = make_companion(A_big)
        @test maximum(abs.(eigvals(comp))) < 1.0
        @test length(C_corrected) == p2
    end
end

@testset "irf_bootstrap" begin
    p = 1
    n = 2
    obs = 100

    dgp = simulate_var(obs; p, n)
    true_coef = dgp.C
    true_Sigma = dgp.Sigma

    model = VAR(dgp.Y; p)
    fit!(model)

    @testset "CI ordering: lower ≤ point ≤ upper" begin
        result = irf_bootstrap(model, Analytical();
                               boot_runs=200, hmax=10, shock_idx=1,
                               ident=:cholesky, alpha=0.05)
        # Not guaranteed pointwise, but median of bootstrap
        # should sit between bounds
        median_boot = mapslices(x -> quantile(x, 0.5), 
                                result.irf_store; dims=3)[:,:,1]
        @test all(result.ci_lower .<= median_boot)
        @test all(median_boot .<= result.ci_upper)
    end

    @testset "wider alpha leads to narrower intervals" begin
        wide = irf_bootstrap(model, Analytical();
                             boot_runs=200, hmax=10, shock_idx=1,
                             ident=:cholesky, alpha=0.10)
        narrow = irf_bootstrap(model, Analytical();
                               boot_runs=200, hmax=10, shock_idx=1,
                               ident=:cholesky, alpha=0.50)
        width_wide = wide.ci_upper - wide.ci_lower
        width_narrow = narrow.ci_upper - narrow.ci_lower
        @test all(width_narrow .<= width_wide .+ 1e-10)
    end

    @testset "shortcut=true vs false both run" begin
        r1 = irf_bootstrap(model, Analytical();
                           boot_runs=50, hmax=5, shortcut=true)
        r2 = irf_bootstrap(model, Analytical();
                           boot_runs=50, hmax=5, shortcut=false)
        @test size(r1.irfs) == size(r2.irfs)
        # point IRFs should be identical (same first-stage correction)
        @test r1.irfs ≈ r2.irfs
    end

    @testset "works with Bootstrap bias method" begin
        result = irf_bootstrap(model, Bootstrap(bias_runs=50);
                               boot_runs=50, hmax=5, shock_idx=1,
                               ident=:cholesky)
        @test size(result.irfs) == (n, 6)
        @test all(result.ci_lower .<= result.ci_upper)
    end

    @testset "bootstrap CIs are stationary" begin
        result = irf_bootstrap(model, Analytical();
                               boot_runs=100, hmax=5)
        # all stored IRFs should be finite (no explosive paths)
        @test all(isfinite, result.irf_store)
    end
end

@testset "irf_bootstrap coverage" begin
    sims = 500
    p = 1
    n = 2
    obs = 100
    hmax = 10
    alpha = 0.10

    dgp = simulate_var(obs; p, n)
    true_coef = dgp.C
    true_Sigma = dgp.Sigma

    # compute true IRFs
    true_model = VAR(dgp.Y; p)
    fit!(true_model)
    true_model.C = true_coef
    true_model.Sigma = true_Sigma
    true_irfs = reduced_form_irf(true_model; hmax=hmax, shock_idx=1)

    coverage = zeros(n, hmax + 1)
    for s in 1:sims
        dgp_s = simulate_var(obs; p, n, C=true_coef, Sigma=true_Sigma)
        model = VAR(dgp_s.Y; p)
        fit!(model)
        result = irf_bootstrap(model, Bootstrap(bias_runs=100);
                               boot_runs=100, hmax=hmax, shock_idx=1, alpha=alpha)
        for i in 1:n, h in 1:(hmax+1)
            if result.ci_lower[i,h] <= true_irfs[i,h] <= result.ci_upper[i,h]
                coverage[i,h] += 1
            end
        end
    end
    coverage ./= sims
    nominal = 1 - alpha
    # average coverage should be in a reasonable range
    avg_coverage = mean(coverage)
    @test avg_coverage > nominal - 0.10
end
