@testset "Nearest Kronecker Product" begin
    n1 = 4
    n2 = 3
    A = randn(n1,n1)
    B = randn(n2,n2)

    phi = kron(B, A)

    est = projection(phi, n1, n2)
    true_a = A / norm(A)
    est_a = est.A / norm(est.A)
    @test isapprox(true_a, est_a; atol=1e-8) || isapprox(true_a, -est_a; atol=1e-8)

    true_b = B / norm(B)
    est_b = est.B / norm(est.B)
    @test isapprox(true_b, est_b; atol=1e-8) || isapprox(true_b, -est_b; atol=1e-8)

    true_product = kron(B, A)
    est_product = kron(est.B, est.A)
    @test isapprox(est_product, true_product, atol=1e-08)

end

