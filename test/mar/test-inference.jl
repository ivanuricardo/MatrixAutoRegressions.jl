
@testset "Rearranged coef leads to rearranged covariance" begin

    n1 = 3
    n2 = 4
    obs = 100
    dgp = simulate_mar(obs)
    matdata = dgp.Y
    vecdata = vectorize(matdata)

    var_model = VAR(vecdata)
    fit!(var_model)

    tensor_ols = reshape(var_model.C[1], (n1, n2, n1, n2))
    reshaped_ols = reshape(permutedims(tensor_ols, (1, 3, 2, 4)), n1 * n1, n2 * n2)
    var_model.C[1]
    # From reshaped to var model
    # (1,1) -> (1,1), (2,1) -> (2,1), (3,1) -> (3,1), (4,1) -> (1,2),
    # (5,1) -> (2,2), (1,2) -> (4,1), (8, 10) -> (5,12)

    av = asymptotic_variance(var_model)
    p = permutation_matrix([n1, n2])
    reshaped_cov = p * av * p'
    reshaped_se = sqrt.(diag(reshaped_cov))

    vector_coef = vec(var_model.C[1])
    se = std_errors(var_model)
    reshaped_se_mat = reshape(reshaped_se, n1 * n1, n2*n2)
    se[1]
end

function ensure_symbol_method!(model, meth::Symbol)
    # Some constructors sometimes store methods as strings; force the symbol used by asymptotic_variance
    if hasproperty(model, :method) && model.method != meth
        try
            model.method = meth
        catch
            # ignore if immutable; the subsequent explicit direct-call tests still exercise functions
        end
    end
end

@testset "Variance Projection" begin

    obs = 100
    dgp = simulate_mar(obs)
    matdata = dgp.Y

    model_proj = MAR(matdata; method = :proj)
    fit!(model)

    var_proj = variance_proj(model_proj)

    @test size(var_proj, 1) == 12 * 12

    model = MAR(matdata; method = :proj, p = 2)
    fit!(model)

    var_proj = variance_proj(model)

    @test size(var_proj, 1) == n1^2 * n2^2 * p^2
end

@testset "Variance ALS" begin
    dgp = simulate_mar(100; p = 3)
    matdata = dgp.Y

    model = MAR(matdata, method = :als, p = 3)
    fit!(model)

    var_proj = variance_proj(model)
    var_als = variance_als(model)
    @test size(var_proj) == size(var_als.cov_full) == size(var_ols)

    model = MAR(matdata; method = :als)
    fit!(model)

    var_proj = variance_proj(model)
    var_als = variance_als(model)
    @test size(var_proj) == size(var_als.cov_full) == size(var_ols)

end

@testset "Variance MLE" begin
    obs = 100
    dgp = simulate_mar(obs; p=2)
    matdata = dgp.Y

    model_mle = MAR(matdata; method = :mle)
    fit!(model_mle)
    var_mle = asymptotic_variance(model_mle)
    @test !isnothing(var_mle.cov_full)

    model_als = MAR(matdata, method = :als)
    fit!(model_als)
    var_als = asymptotic_variance(model_als)
    @test !isnothing(var_als.cov_full)

    model_proj = MAR(matdata, method = :proj)
    fit!(model_proj)
    var_proj = asymptotic_variance(model_proj)
    @test !isnothing(var_proj.cov_full)

    @test size(var_mle.cov_full) == size(var_als.cov_full) == size(var_proj.cov_full)

    mle_cov_full = var_mle.cov_full
    als_cov_full = var_als.cov_full
    ev_mle = abs.(eigvals(mle_cov_full))
    ev_als = abs.(eigvals(als_cov_full))

    # Testing whether there are 24 valid parameters, the rest should be zeros
    @test length(ev_mle[ev_mle .> 1e-10]) == 24
    @test length(ev_als[ev_als .> 1e-10]) == 24

    dgp = simulate_mar(obs; p=2)
    matdata = dgp.Y

    model_mle = MAR(matdata; method = :mle, p = 2)
    fit!(model_mle)
    var_mle = asymptotic_variance(model_mle)

    model_als = MAR(matdata; method = :als, p = 2)
    fit!(model_als)
    var_als = asymptotic_variance(model_als)

    mle_cov_full = var_mle.cov_full
    als_cov_full = var_als.cov_full
    ev_mle = abs.(eigvals(mle_cov_full))
    ev_als = abs.(eigvals(als_cov_full))

    # Each lag has 24 parameters
    @test length(ev_mle[ev_mle .> 1e-10]) == 24 * 2
    @test length(ev_als[ev_als .> 1e-10]) == 24 * 2

end

@testset "Lagged Data structure" begin
    obs = 100
    n1 = 3
    n2 = 4
    p1 = 2
    dgp1 = simulate_mar(obs; p=p1)
    matdata1 = dgp1.Y
    model1 = MAR(matdata1; method = :proj, p=p1)
    X1 = structure_lagged_data(model1)

    @test size(X1) == (n1*p1, n2*p1, obs-p1)

    p2 = 3
    dgp2 = simulate_mar(obs; p=p2)
    matdata2 = dgp2.Y
    model2 = MAR(matdata2; method = :proj, p=p2)
    X2 = structure_lagged_data(model2)

    @test size(X2) == (n1*p2, n2*p2, obs-p2)

    p3 = 4
    dgp3 = simulate_mar(obs; p=p3)
    matdata3 = dgp3.Y
    model3 = MAR(matdata3; method = :proj, p=p3)
    X3 = structure_lagged_data(model3)

    @test size(X3) == (n1*p3, n2*p3, obs-p3)

end

@testset "Selection Matrix" begin
    obs = 100
    dgp = simulate_mar(obs; p=3)
    matdata = dgp.Y

    model = MAR(matdata; method = :proj, p=3)
    fit!(model)

    Astack, Bstack = stack_coefs(model)

    sel1 = selection_matrix(1, model)
    sel2 = selection_matrix(2, model)
    sel3 = selection_matrix(3, model)

    full_kron = vec(kron(Bstack, Astack))
    vec1 = vec(kron(model.B[1], model.A[1]))
    vec2 = vec(kron(model.B[2], model.A[2]))
    vec3 = vec(kron(model.B[3], model.A[3]))

    @test isapprox(norm(sel1 * full_kron - vec1), 0.0)
    @test isapprox(norm(sel2 * full_kron - vec2), 0.0)
    @test isapprox(norm(sel3 * full_kron - vec3), 0.0)

end

@testset "Stack Matrices" begin

    obs = 100
    dgp = simulate_mar(obs)
    matdata = dgp.Y

    model = MAR(matdata; method = :proj, p=3)
    fit!(model)

    Astack, Bstack = stack_coefs(model)
    @test size(Astack) == (3,9)
    @test size(Bstack) == (4,12)
    @test isapprox(norm(Astack), 1)

    model = MAR(matdata; method = :proj, p=1)
    fit!(model)

    Astack, Bstack = stack_coefs(model)
    @test size(Astack) == (3,3)
    @test size(Bstack) == (4,4)
    @test isapprox(norm(Astack), 1)

    model = MAR(matdata; method = :proj, p=2)
    fit!(model)

    Astack, Bstack = stack_coefs(model)
    @test size(Astack) == (3,6)
    @test size(Bstack) == (4,8)
    @test isapprox(norm(Astack), 1)

end

@testset "Constructing gamma" begin
    obs = 100
    dgp = simulate_mar(obs)
    matdata = dgp.Y

    model = MAR(matdata; method = :proj, p = 2)
    fit!(model)
    gamma = construct_gamma(model)

    @test vec(model.A[1]) == gamma[1:9, 1]
    @test vec(model.A[2]) == gamma[10:18, 2]
end

@testset "Standard Errors" begin
    obs = 100
    dgp = simulate_mar(obs; p=3)
    matdata = dgp.Y

    model = MAR(matdata; method = :als, p=3)
    fit!(model)
    A_stderr, B_stderr, C_stderr = std_errors(model)

    C = kron(model.B[1], model.A[1])
    @test size(A_stderr[1]) == size(model.A[1])
    @test size(B_stderr[1]) == size(model.B[1])
    @test size(C_stderr[1]) == size(C)

    C = kron(model.B[2], model.A[2])
    @test size(A_stderr[2]) == size(model.A[2])
    @test size(B_stderr[2]) == size(model.B[2])
    @test size(C_stderr[2]) == size(C)
end

@testset "Check if second lag is insignificant" begin
    obs = 1000
    dgp = simulate_mar(obs; p=1)
    matdata = dgp.Y

    model2 = MAR(matdata; method = :mle, p=2)
    fit!(model2)

    crit = 1.96  # 95% normal-approx CI
    A_stderr, B_stderr, C_stderr = std_errors(model2)
    number_negative = count(<(0), vec(model2.C[2]) - crit .* vec(C_stderr[2]))
    number_positive = count(>(0), vec(model2.C[2]) + crit .* vec(C_stderr[2]))

    # I expect the lower bound of the CI to be negative and the upper bound of
    # the CI to be positive in ~90% of the cases

    @test number_negative > 130
    @test number_positive > 130
    est = vec(model2.C[2])
    se  = vec(C_stderr[2])

    lower = est .- crit .* se
    upper = est .+ crit .* se

    n_zero = sum((lower .< 0) .& (upper .> 0))   # number of CIs that include zero
    total  = length(est)
    prop_zero = n_zero / total

    # There should be a decent amount of zeros in the second lag.
    # I check how many fail to reject the null, should be close to 80
    # In reality, I should expect about 144 * 0.05 = 7.2 false positives
    @test prop_zero > 0.90

    # I can also use LR tests to see if the second lag does not help with 
    # prediction
    model1 = MAR(matdata; method = :mle, p = 1)
    fit!(model1)

    residuals1 = vectorize(residuals(model1))
    residuals2 = vectorize(residuals(model2))

    Sigma1 = residuals1 * residuals1' / size(residuals1, 1)
    Sigma2 = residuals2 * residuals2' / size(residuals2, 1)

    q = number_parameters(model2) - number_parameters(model1)

    LR = size(residuals2, 1) * (logdet(Sigma1) - logdet(Sigma2))
    pval = 1 - cdf(Chisq(q), LR)
    @test pval > 0.05

end

@testset "easy symmetric function" begin
    n = 12
    A = randn(n, n)
    sym_a = MatrixAutoRegressions.sym(A)
    @test issymmetric(sym_a)
end

@testset "get C standard error" begin
    obs = 1000
    dgp = simulate_mar(obs; p=1)
    matdata = dgp.Y

    model = MAR(matdata; method = :mle, p=1)
    fit!(model)

    c_stderr = get_c_stderr(model)
    all_std_errors = std_errors(model)
    @test all_std_errors.C_stderr[1] == c_stderr[1]

    var_model = VAR(vectorize(matdata))
    fit!(var_model)
    var_stderr = std_errors(var_model)
    @test norm(var_stderr[1]) > norm(c_stderr[1])
    
end
