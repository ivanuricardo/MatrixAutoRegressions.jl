

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

"""
Simulate a MAR model
"""
function simulate_mar(
    obs::Int;
    n1::Int = 3,
    n2::Int = 4,
    A::Union{Nothing, AbstractVecOrMat} = nothing,
    B::Union{Nothing, AbstractVecOrMat} = nothing,
    Sigma1::Union{Nothing, AbstractMatrix} = nothing,
    Sigma2::Union{Nothing, AbstractMatrix} = nothing,
    burnin::Int = 50,
    snr::Real = 0.7,
    )

    if A === nothing || B === nothing
        coefs = generate_mar_coefs(n1, n2)
        A = coefs.A
        B = coefs.B
    end

    if Sigma1 === nothing || Sigma2 === nothing
        # For snr
        maxeigA = maximum(abs.(eigen(A).values))
        maxeigB = maximum(abs.(eigen(B).values))
        eigval_coef = maxeigA * maxeigB
        eigval_err = eigval_coef / snr
        Sigma = diagm(repeat([eigval_err], n1 * n2))
        Sigma1, Sigma2, Sigma = projection(Sigma, n1, n2)
        Sigma1 = abs.(Sigma1)
        Sigma2 = abs.(Sigma2)
    end

    n1, n2 = size(A, 1), size(B, 1)
    total_obs = obs + burnin
    Y = Array{Float64, 3}(undef, n1, n2, total_obs)
    Y[:, :, 1] = zeros(n1, n2)
    matrix_normal = MatrixNormal(zeros(n1, n2), Sigma1, Sigma2)
    matrix_errs = rand(matrix_normal, total_obs)
    for t in 2:total_obs
        Y[:, :, t] = A * Y[:, :, t-1] * B' + matrix_errs[t]
    end
    Y = Y[:, :, burnin+1:end]
    return (; Y, A, B, Sigma1, Sigma2)
end
