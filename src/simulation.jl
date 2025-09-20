
"""
Generate MAR coefficients with the normalization that A has a frobenius norm 
of one.
"""
function generate_mar_coefs(n1::Int, n2::Int; p::Int=1, maxiter::Int=500)

    A = Vector{Matrix{Float64}}(undef, p)
    B = Vector{Matrix{Float64}}(undef, p)
    scale = 1.0

    count = 0
    for i in 1:maxiter
        count += 1

        preA = [scale * randn(n1, n1) for _ in 1:p]
        preB = [scale * randn(n2, n2) for _ in 1:p]

        A, B = normalize_slices(preA, preB)

        # Decrease scale every 20 tries if not stable
        if i % 20 == 0
            scale *= 0.9
        end

        if isstable(A, B)
            sorted_eigs = mar_eigvals(A, B)
            return (; A, B, sorted_eigs, count)
        end
    end

    @warn "Reached the maximum number of iterations! May not be stable."
    sorted_eigs = mar_eigvals(A, B)
    return (; A, B, sorted_eigs)
end

"""
Simulate a MAR model
"""
function simulate_mar(
    obs::Int;
    n1::Int = 3,
    n2::Int = 4,
    p::Int = 1,
    A::Union{Nothing, Vector{<:AbstractMatrix}} = nothing,
    B::Union{Nothing, Vector{<:AbstractMatrix}} = nothing,
    Sigma1::Union{Nothing, AbstractMatrix} = nothing,
    Sigma2::Union{Nothing, AbstractMatrix} = nothing,
    burnin::Int = 50,
    snr::Real = 0.7,
    maxiter::Int = 500,
)
    if A === nothing || B === nothing
        coefs = generate_mar_coefs(n1, n2; p, maxiter)
        A, B = coefs.A, coefs.B
    end

    # Construct error covariance if not provided
    if Sigma1 === nothing || Sigma2 === nothing
        eigval_coef = maximum(mar_eigvals(A, B))
        eigval_err = eigval_coef / snr
        Sigma = diagm(repeat([eigval_err], n1 * n2))
        Sigma1, Sigma2, Sigma = projection(Sigma, (n1, n2))
        Sigma1 = abs.(Sigma1)
        Sigma2 = abs.(Sigma2)
    end

    n1, n2 = size(A[1], 1), size(B[1], 1)
    total_obs = obs + burnin

    Y = Array{Float64, 3}(undef, n1, n2, total_obs)
    Y[:, :, 1] .= 0.0

    matrix_normal = MatrixNormal(zeros(n1, n2), Sigma1, Sigma2)
    matrix_errs = rand(matrix_normal, total_obs)

    for t in (p+1):total_obs
        Y[:, :, t] = matrix_errs[t]
        for j in 1:p
            Y[:, :, t] .+= A[j] * Y[:, :, t-j] * B[j]'
        end
    end

    Y = Y[:, :, burnin+1:end]
    sorted_eigs = mar_eigvals(A, B)

    return (; Y, A, B, Sigma1, Sigma2, sorted_eigs)
end
