@testset "vech selection" begin

    n = 3
    L = vech_selection_mat(n)

    A = reshape(1:9, n, n)
    vechA = vech(A)

    correct_a = [
        A[1,1];
        A[2,1];
        A[3,1];
        A[2,2];
        A[3,2];
        A[3,3];
    ]
    @test vechA == correct_a
end

@testset "Asymptotic variances of sigma and cholesky" begin

    obs = 1000
    p = 2
    n = 4
    dgp = simulate_var(obs; p=2, n)
    vecdata = dgp.Y

    var_model = VAR(vecdata; p)
    fit!(var_model)

    var_asymptotic_variance = avar_sigma(var_model)
    ses = sqrt.(diag(var_asymptotic_variance))
    
    vech_sigma = vech(var_model.Sigma)

    cholesky_asymptotic_variance = avar_cholesky(var_model)
    cholesky_ses = sqrt.(diag(cholesky_asymptotic_variance))

    @test length(vech_sigma) == length(ses) == n * (n+1) / 2

    # Quick revech test
    u = revech(vech_sigma)
    @test u == var_model.Sigma

    lower_u = revech_lower(vech_sigma)
    @test istril(lower_u)
    @test istriu(u - lower_u, 1) # From the first superdiagonal

end
