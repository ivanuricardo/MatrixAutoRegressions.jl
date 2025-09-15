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




function als(A_init, B_init, resp, pred; maxiter=100, tol=1e-6)
    A = copy(A_init)
    B = copy(B_init)
    n1, n2 = size(A, 1), size(B, 1)
    n, obs = size(resp)

    for i in 1:maxiter
        X_given_A = kron(I(n2), A) * pred
        B = resp * X_given_A' * inv(X_given_A * X_given_A')




        X_given_B = kron(B, I(n1)) * pred
        if i == maxiter
            @warn "Reached maximum number of iterations"
            return A, B
        end
    end
end

