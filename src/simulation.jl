
"""
Generate MAR coefficients with the normalization that A has a frobenius norm 
of one.
"""
function generate_mar_coefs(n1, n2; p::Int=1, maxiter::Int=500)
    A = Array{Float64, 3}(undef, n1, n1, p)
    B = Array{Float64, 3}(undef, n2, n2, p)
    scale = 1

    count = 0
    for i in 1:maxiter
        count += 1
        preA = randn(n1, n1, p)
        preB = randn(n2, n2, p)
        preA .*= scale
        preB .*= scale
        A, B = normalize_slices(preA, preB)

        if i % 20 == 0
            # If it doesn't work after 20 mod iterations, decrease scale
            scale *= 0.9
        end
        if isstable(A, B)
            sorted_eigs = mar_eigvals(A, B)
            return (;A, B, sorted_eigs)
        end
    end

    @warn "Reached the maximum number of iterations! May not be stable."
    sorted_eigs = mar_eigvals(A, B)
    return (;A, B, sorted_eigs)
end

"""
Simulate a MAR model
"""
function simulate_mar(
    obs::Int;
    n1::Int = 3,
    n2::Int = 4,
    p::Int = 1,
    A::Union{Nothing, AbstractArray} = nothing,
    B::Union{Nothing, AbstractArray} = nothing,
    Sigma1::Union{Nothing, AbstractMatrix} = nothing,
    Sigma2::Union{Nothing, AbstractMatrix} = nothing,
    burnin::Int = 50,
    snr::Real = 0.7,
    maxiter::Int = 500,
    )

    if A === nothing || B === nothing
        coefs = generate_mar_coefs(n1, n2; p, maxiter)
        A = coefs.A
        B = coefs.B
    end

    if Sigma1 === nothing || Sigma2 === nothing
        # For snr
        eigval_coef = maximum(mar_eigvals(A, B))
        eigval_err = eigval_coef / snr
        Sigma = diagm(repeat([eigval_err], n1 * n2))
        Sigma1, Sigma2, Sigma = projection(Sigma, (n1, n2))
        Sigma1 = abs.(Sigma1)
        Sigma2 = abs.(Sigma2)
    end

    n1, n2 = size(A, 1), size(B, 1)
    total_obs = obs + burnin
    Y = Array{Float64, 3}(undef, n1, n2, total_obs)
    Y[:, :, 1] = zeros(n1, n2)
    matrix_normal = MatrixNormal(zeros(n1, n2), Sigma1, Sigma2)
    matrix_errs = rand(matrix_normal, total_obs)

    for t in (p+1):total_obs
        Y[:, :, t] = matrix_errs[t]
        for j in 1:p
            Y[:, :, t] .+= A[:, :, j] * Y[:, :, t-j] * B[:, :, j]'
        end
    end

    Y = Y[:, :, burnin+1:end]
    sorted_eigs = mar_eigvals(A, B)

    return (; Y, A, B, Sigma1, Sigma2, sorted_eigs)
end
