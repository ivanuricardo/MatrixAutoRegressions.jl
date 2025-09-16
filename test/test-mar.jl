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




@testset "ls objective" begin

    # If the snr goes up, the ssr should go down
    obs = 100
    results1 = simulate_mar(obs; snr = 0.2)
    A_init1 = results1.A
    B_init1 = results1.B
    matdata1 = results1.Y

    ssr1 = ls_objective(matdata1, A_init1, B_init1)

    results2 = simulate_mar(obs; snr = 1)
    A_init2 = results2.A
    B_init2 = results2.B
    matdata2 = results2.Y

    ssr2 = ls_objective(matdata2, A_init2, B_init2)
    @test ssr1 > ssr2

end

obs = 100
results = simulate_mar(obs)
A_init = results.A
B_init = results.B
matdata = results.Y
resp = matdata[:, :, 2:end]
pred = matdata[:, :, 1:end-1]
maxiter = 1000
tol = 1e-06


A = copy(A_init)
B = copy(B_init)
n1, n2 = size(A, 1), size(B, 1)
obs = size(resp, 3)

function update_B(resp::AbstractArray{T}, pred::AbstractArray{T}, A::AbstractMatrix{T}) where T
    n2 = size(resp, 2)
    B_num = zeros(n2, n2)
    B_den = zeros(n2, n2)
    obs = size(resp, 3)

    for t in 1:(obs-1)
        B_num += resp[:, :, t]' * A * pred[:, :, t]
        B_den += resp[:, :, t]' * A'A * pred[:, :, t]
    end

    return B_den / B_num
end

function update_A(resp::AbstractArray{T}, pred::AbstractArray{T}, B::AbstractMatrix{T}) where T
    n1 = size(resp, 1)
    A_num = zeros(n1, n1)
    A_den = zeros(n1, n1)
    obs = size(resp, 3)

    for t in 1:(obs-1)
        A_num += resp[:, :, t] * B * pred[:, :, t]'
        A_den += pred[:, :, t] * B'B * pred[:, :, t]'
    end

    return A_den / A_num
end

track_a = fill(NaN, maxiter)
track_b = fill(NaN, maxiter)

for i in 1:maxiter
    A_old = copy(A)
    B_old = copy(B)
    B = update_B(resp, pred, A)
    A = update_A(resp, pred, B)

    norm_A = norm(A)
    A = A / norm_A
    B = B * norm_A

    track_a[i] = norm(A - A_old)
    track_b[i] = norm(B - B_old)

    if norm(A - A_old) < tol && norm(B - B_old) < tol
        break
    end

    if i == maxiter
        @warn "Reached maximum number of iterations"
        return A, B
    end
end










