
"""
Simulate a MAR model
"""
function simulate_mar(
    obs::Int,
    A::AbstractVecOrMat, 
    B::AbstractVecOrMat,
    Sigma1::AbstractMatrix,
    Sigma2::AbstractMatrix,
    )
    n1, n2 = size(A, 1), size(B, 1)
    Y = Array{Float64, 3}(undef, n1, n2, obs)
    Y[:, :, 1] = zeros(n1, n2)
    matrix_normal = MatrixNormal(zeros(n1, n2), Sigma1, Sigma2)
    matrix_errs = rand(matrix_normal, obs)
    for t in 2:obs
        Y[:, :, t] = A * Y[:, :, t-1] * B' + matrix_errs[1]
    end
    return Y
end

"""
Generate MAR coefficients with the normalization that A has a frobenius norm 
of one.
"""
function generate_mar_coefs(n1, n2; maxiters=100)
    A = Array{Float64, 2}(undef, n1, n1)
    B = Array{Float64, 2}(undef, n2, n2)
    scale = 1

    for i in 1:maxiters
        preA = scale * randn(n1, n1)
        normA = norm(preA)
        A .= preA / normA
        preB = randn(n2, n2)
        B .= preB * normA

        if isstable(A, B)
            phi = kron(B, A)
            eig_phi = sort(abs.(eigvals(phi)), rev=true)
            return (;A, B, phi, eig_phi)
        end

        if i % 20 == 0
            # If it doesn't work after 20 mod iterations, decrease scale
            scale = scale * 0.9
        end
    end

    @warn "Reached the maximum number of iterations! May not be stable."
    phi = kron(B, A)
    eig_phi = sort(abs.(eigvals(phi)), rev=true)
    return (;A, B, phi, eig_phi)

end


