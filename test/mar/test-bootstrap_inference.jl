
@testset "irf_bootstrap coverage" begin
    sims = 10
    p = 1
    n1 = 2
    n2 = 3
    n = n1 * n2
    obs = 100
    hmax = 10
    alpha = 0.10

    dgp = simulate_mar(obs; p, n1, n2)
    trueA = dgp.A
    trueB = dgp.B
    true_coef = dgp.C
    true_Sigma1, true_Sigma2 = dgp.Sigma1, dgp.Sigma2

    # compute true IRFs
    true_model = MAR(dgp.Y; p)
    fit!(true_model)
    true_model.C = true_coef
    true_model.Sigma1, true_model.Sigma2 = true_Sigma1, true_Sigma2
    true_irfs = reduced_form_irf(true_model; hmax=hmax, shock_idx=[1, 1])

    coverage = zeros(n, hmax + 1)
    for s in 1:sims
        dgp_s = simulate_mar(obs; p, n1, n2, A=trueA, B=trueB,
                             Sigma1=true_Sigma1, Sigma2=true_Sigma2)
        model = MAR(dgp_s.Y; p)
        fit!(model)
        result = irf_bootstrap(model, Bootstrap(bias_runs=1000);
                        boot_runs=2000, hmax=hmax, shock_idx=[1, 1], alpha=alpha)
        for i in 1:n, h in 1:(hmax+1)
            if result.ci_lower[i,h] <= true_irfs[i,h] <= result.ci_upper[i,h]
                coverage[i,h] += 1
            end
        end
    end
    coverage ./= sims
    nominal = 1 - alpha
    # average coverage should be in a reasonable range
    avg_coverage = mean(coverage[:, 2:end])
    @test avg_coverage > nominal - 0.10
end
