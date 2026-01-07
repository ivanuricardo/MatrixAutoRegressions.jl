@testset "Log likelihoods" begin
    obs = 100
    dgp = simulate_mar(obs)

    vecdata = vectorize(dgp.Y)
    matdata = dgp.Y

    var_model = VAR(vecdata)
    fit!(var_model)

    mar_model = MAR(matdata)
    fit!(mar_model)

    ll_var = loglikelihood(var_model)
    ll_mar = loglikelihood(mar_model)

    @test ll_var > ll_mar  # more parameters -> higher likelihood

    # Verify loglikelihood(VAR) equals the sum of logpdf values from Distributions.MvNormal constructed with model.Sigma and the residual columns
    E = residuals(var_model)
    sigma = Symmetric(var_model.Sigma)
    mvn = MvNormal(zeros(size(E,1)), sigma)

    ll_mvn = sum(logpdf.(Ref(mvn), eachcol(E)))
    @test isapprox(loglikelihood(var_model), ll_mvn; atol=1e-8, rtol=0)

    # Can verify the same with matrix normal errors
    E = (mar_model.p == 0) ? mar_model.residuals : residuals(mar_model)
    n1, n2, obs = size(E)
    sigma1 = Symmetric(mar_model.Sigma1)
    sigma2 = Symmetric(mar_model.Sigma2)

    sigma_kron = kron(sigma2, sigma1)
    mvn = MvNormal(zeros(n1*n2), sigma_kron)

    ll_mvn = 0.0
    for t in 1:obs
        ll_mvn += logpdf(mvn, vec(E[:, :, t]))
    end

    @test isapprox(loglikelihood(mar_model), ll_mvn; atol=1e-8, rtol=0)
end

@testset "MAR method must be :mle" begin

    obs = 100
    dgp = simulate_mar(obs)
    matdata = dgp.Y

    mar_model = MAR(matdata)
    fit!(mar_model)

    bad = deepcopy(mar_model)
    bad.method = :ols
    @test_throws ErrorException loglikelihood(bad)
end

@testset "MAR p==0 branch" begin

    obs = 100
    dgp = simulate_mar(obs)
    matdata = dgp.Y

    mar_model = MAR(matdata)
    fit!(mar_model)

    m0 = deepcopy(mar_model)
    m0.p = 0
    @test isapprox(loglikelihood(m0), begin
        E = m0.residuals
        Σ_kron = kron(Symmetric(m0.Sigma2), Symmetric(m0.Sigma1))
        mvn = MvNormal(zeros(size(Σ_kron,1)), Σ_kron)
        sum(logpdf.(Ref(mvn), (vec(E[:,:,t]) for t in 1:size(E,3))))
    end; atol=1e-8, rtol=0)
end

@testset "MAR quadratic term matches kron-inv" begin
    # Just a little sanity check for the thing inside my loglikelihood loop
    n1 = 2
    n2 = 3
    E1 = randn(n1,n2)
    A = randn(n1,n1)
    sigma1 = Symmetric(A*A')
    B = randn(n2,n2)
    sigma2 = Symmetric(B*B')

    sigma_kron = kron(sigma2, sigma1)
    q_manual = 0.5 * vec(E1)' * (sigma_kron \ vec(E1))

    ch1 = cholesky(sigma1)
    ch2 = cholesky(sigma2)
    U = ch1.L \ E1
    W = (ch2.L \ U')'
    q_impl = 0.5 * sum(abs2, W)

    @test isapprox(q_impl, q_manual; atol=1e-10)
end
