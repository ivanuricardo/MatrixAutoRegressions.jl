
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



