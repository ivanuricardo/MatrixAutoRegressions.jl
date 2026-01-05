
@testset "Impulse Responses" begin
    # simulate / fit
    obs = 200
    dgp = simulate_mar(obs; p=2)
    matdata = dgp.Y
    model = MAR(matdata; method = :mle, p=2)
    fit!(model)

    C = model.C
    p = length(C)
    n = size(C[1],1)
    hmax = 8

    # irf_ma tests
    theta = irf_ma(model; hmax)
    @test length(theta) == hmax+1
    @test all(size(theta[h+1]) == (n,n) for h in 0:hmax)
    I_n = Matrix{eltype(C[1])}(I, n, n)
    @test theta[1] == I_n  # Θ_0 == I

    for h in 1:hmax
        M = zeros(eltype(C[1]), n, n)
        for j in 1:min(p,h)
            M .+= C[j] * theta[h+1-j]
        end
        @test theta[h+1] == M
    end

    # reduced-form IRF tests
    shock_idx = [1,1]                  # same convention you used
    vec_shock_idx = shock_idx[1] + (shock_idx[2]-1) * 3
    rf = reduced_form_irf(model; hmax, shock_idx)
    @test size(rf) == (n, hmax+1)

    e = zeros(eltype(C[1]), n)
    e[vec_shock_idx] = one(eltype(C[1]))
    for h in 0:hmax
        @test rf[:, h+1] == theta[h+1] * e
    end

    # orthogonalized IRF tests
    Sigma = kron(model.Sigma2, model.Sigma1)
    orf = orthogonalized_irf(model, Sigma, hmax, shock)
    @test size(orf) == (n, hmax+1)

    L = cholesky(Symmetric(Sigma)).L
    b = L[:, vec_shock_idx]

    for h in 0:hmax
        @test orf[:, h+1] == theta[h+1] * b
    end
end

@testset "Making Jacobian G" begin
    obs = 200
    dgp = simulate_mar(obs; p=3)
    matdata = dgp.Y
    model = MAR(matdata; method = :mle, p=3)
    fit!(model)

    C = model.C
    p = length(C)
    n = size(C[1],1)
    hmax = 8

    # irf_ma tests
    theta = irf_ma(model; hmax)

    closed_G1 = make_g(theta, C, 1)
    closed_G2 = make_g(theta, C, 2)
    closed_G3 = make_g(theta, C, 3)
    closed_G4 = make_g(theta, C, 4)
    closed_G5 = make_g(theta, C, 5)

    @test size(closed_G1) == (n^2, n^2 * p)
    @test size(closed_G2) == (n^2, n^2 * p)
    @test size(closed_G3) == (n^2, n^2 * p)
    @test size(closed_G4) == (n^2, n^2 * p)
    @test size(closed_G5) == (n^2, n^2 * p)

    @test closed_G1 ≈ kron(hcat(I(n), zeros(n, n*(p-1))), I(n))

    # For h = 2
    # Note that I shift the index for theta because theta[1] should be theta[0]
    block = hcat(theta[2]', theta[1]', zeros(size(theta[1])))
    kron_prod = kron(block, I(n))
    G2 = kron_prod + kron(I(n), C[1]) * closed_G1
    @test closed_G2 ≈ G2

    # For h = 3
    block = hcat(theta[3]', theta[2]', theta[1]')
    kron_prod = kron(block, I(n))
    G3 = kron_prod + kron(I(n), C[1]) * G2 + kron(I(n), C[2]) * closed_G1
    @test closed_G3 ≈ G3

    # For h = 4
    block = hcat(theta[4]', theta[3]', theta[2]')
    kron_prod = kron(block, I(n))
    G4 = kron_prod + kron(I(n), C[1]) * G3 + kron(I(n), C[2]) * G2 + kron(I(n), C[3]) * closed_G1
    @test closed_G4 ≈ G4

    # For h = 5
    block = hcat(theta[5]', theta[4]', theta[3]')
    kron_prod = kron(block, I(n))
    G5 = kron_prod + kron(I(n), C[1]) * G4 + kron(I(n), C[2]) * G3 + kron(I(n), C[3]) * G2
    @test closed_G5 ≈ G5
end

@testset "IRF variances" begin

    obs = 100
    dgp = simulate_mar(obs; p=2)
    matdata = dgp.Y

    model = MAR(matdata; method = :mle, p=2)
    fit!(model)

    hmax = 14
    theta = irf_ma(model; hmax)
    irf_vars = all_irf_variances(model, theta; hmax)
    @test size(irf_vars[1]) == (3*4, 3*4)
    @test size(irf_vars[2]) == (3*4, 3*4)

    dgp = simulate_mar(obs; p=3)
    matdata = dgp.Y

    model = MAR(matdata; method = :mle, p=3)
    fit!(model)

    hmax = 14
    theta = irf_ma(model; hmax)
    irf_vars = all_irf_variances(model, theta; hmax)
    @test size(irf_vars[1]) == (3*4, 3*4)
    @test size(irf_vars[2]) == (3*4, 3*4)
    @test size(irf_vars[3]) == (3*4, 3*4)
    @test size(irf_vars[4]) == (3*4, 3*4)

    shock_idx = [2,3]
    selected_irfs = irf_variance(model, theta; hmax, shock_idx)

end

@testset "Full IRF test" begin
    obs = 100
    dgp = simulate_mar(obs; p=4)
    matdata = dgp.Y

    model = MAR(matdata; method = :mle, p=4)
    fit!(model)
    hmax = 14
    shock_idx = [2,1]

    mar_irfs = irf(model; hmax, shock_idx)
    @test size(mar_irfs.irfs) == size(mar_irfs.irf_se)

end

@testset "Local Projection" begin

# This is all given
    hmax = 10
    obs = 100
    dgp = simulate_mar(obs)
    data = dgp.Y
    model = MAR(data, method = :als)

################################################################################


end
