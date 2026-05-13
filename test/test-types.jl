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

# === Helper: small dummy model types to isolate IC logic ===
mutable struct DummyMLE <: AbstractARModel
    obs::Int
    method::Symbol
    Sigma::Matrix{Float64}
    k::Int
    ll::Float64
end

mutable struct DummyNonMLE <: AbstractARModel
    obs::Int
    method::Symbol
    Sigma::Matrix{Float64}
    k::Int
end

# Also need some helpers here
MatrixAutoRegressions.number_parameters(m::DummyMLE) = m.k
MatrixAutoRegressions.number_parameters(m::DummyNonMLE) = m.k

MatrixAutoRegressions.loglikelihood(m::DummyMLE) = m.ll

@testset "Information criteria: basic formula checks" begin
    # --- MLE branch matches explicit formulas ---
    m = DummyMLE(100, :mle, zeros(1,1), 10, -123.456)
    expected_aic = 2 * m.k - 2 * loglikelihood(m)
    expected_bic = m.k * log(m.obs) - 2 * loglikelihood(m)
    expected_hqc = m.k * 2 * log(log(m.obs)) - 2 * loglikelihood(m)
    @test aic(m) == expected_aic
    @test bic(m) == expected_bic
    @test hqc(m) == expected_hqc

    # --- non-MLE branch uses cholesky(logdet) formula ---
    # Use identity covariance for clarity: logdetterm == 0 so AIC == 2k
    sigma = Matrix{Float64}(I, 4, 4)
    mn = DummyNonMLE(50, :als, sigma, 7)
    ch = cholesky(mn.Sigma)
    logdetterm = 2 * sum(log, diag(ch.L))
    expected_aic_nonmle = 2 * mn.k + mn.obs * logdetterm
    expected_bic_nonmle = mn.k * log(mn.obs) + mn.obs * logdetterm
    expected_hqc_nonmle = 2 * mn.k * log(log(mn.obs)) + mn.obs * logdetterm
    @test aic(mn) == expected_aic_nonmle
    @test bic(mn) == expected_bic_nonmle
    @test hqc(mn) == expected_hqc_nonmle

    # sanity for identity Sigma (logdetterm==0)
    @test isapprox(logdetterm, 0.0; atol=1e-14)
    @test aic(mn) == 2 * mn.k
    @test bic(mn) == mn.k * log(mn.obs)
    @test hqc(mn) == 2 * mn.k * log(log(mn.obs))
end

@testset "IC dispatcher and invalid type" begin
    m = DummyMLE(20, :mle, zeros(1,1), 3, -10.0)

    @test ic(m; ic_type=:aic) == aic(m)
    @test ic(m; ic_type=:bic) == bic(m)
    @test ic(m; ic_type=:hqc) == hqc(m)
    @test_throws ErrorException ic(m; ic_type=:unknown)
end

@testset "Monotonicity in parameter count (penalty increases predictably)" begin
    # Two MLE models with identical loglik but different k
    base_ll = -200.0
    m1 = DummyMLE(200, :mle, zeros(1,1), 5, base_ll)
    m2 = DummyMLE(200, :mle, zeros(1,1), 8, base_ll)

    # AIC difference should be 2*(k2-k1)
    @test isapprox(aic(m2) - aic(m1), 2 * (m2.k - m1.k))

    # BIC difference should be log(obs) * (k2-k1)
    @test isapprox(bic(m2) - bic(m1), log(m1.obs) * (m2.k - m1.k))

    # HQC difference should be 2*log(log(obs))*(k2-k1)
    @test isapprox(hqc(m2) - hqc(m1), 2 * log(log(m1.obs)) * (m2.k - m1.k))

    # Same checks for non-MLE branch: penalty terms independent of loglik
    Σ = Matrix{Float64}(I, 2, 2)
    n1 = DummyNonMLE(120, :als, Σ, 4)
    n2 = DummyNonMLE(120, :als, Σ, 6)

    @test isapprox(aic(n2) - aic(n1), 2 * (n2.k - n1.k))
    @test isapprox(bic(n2) - bic(n1), log(n1.obs) * (n2.k - n1.k))
    @test isapprox(hqc(n2) - hqc(n1), log(log(n1.obs)) * (n2.k - n1.k))
end

@testset "Numeric stability / edge checks" begin
    # Large obs value: ensure no unexpected NaN for HQC log(log(obs))
    large_obs = 10_000
    m = DummyMLE(large_obs, :mle, zeros(1,1), 2, 1.0)
    @test !isnan(hqc(m)) && isfinite(hqc(m))

    # Small obs but >1 so log(log(obs)) defined
    small_obs = 3
    m_small = DummyMLE(small_obs, :mle, zeros(1,1), 2, 1.0)
    @test isfinite(hqc(m_small))
end
