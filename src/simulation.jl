function make_spd_eig(A::AbstractMatrix; tol=1e-8)
    A = abs.(Symmetric(A))
    F = eigen(A)
    vals = clamp.(F.values, tol, Inf)
    return F.vectors * Diagonal(vals) * F.vectors' |> Symmetric
end

"""
Generate MAR coefficients with the normalization that A has a frobenius norm 
of one.
"""
function generate_mar_coefs(n1::Int, n2::Int; p::Int=1, maxiter::Int=1000)

    A = Vector{Matrix{Float64}}(undef, p)
    B = Vector{Matrix{Float64}}(undef, p)
    scale = 5

    count = 0
    for i in 1:maxiter
        count += 1

        preA = [scale * randn(n1, n1) for _ in 1:p]
        preB = [scale * randn(n2, n2) for _ in 1:p]

        A, B = normalize_slices(preA, preB)

        # Decrease scale every 20 tries if not stable
        if i % 50 == 0
            scale *= 0.75
        end

        if isstable(A, B; mineigen = 0.1)
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
    burnin::Int = 500,
    snr::Real = 1.0,
    maxiter::Int = 1000,
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
        Sigma1 = make_spd_eig(Sigma1; tol=1e-8)
        Sigma2 = make_spd_eig(Sigma2; tol=1e-8)
        Sigma = make_spd_eig(Sigma; tol=1e-8)
    end

    n1, n2 = size(A[1], 1), size(B[1], 1)
    total_obs = obs + burnin

    Y = Array{Float64, 3}(undef, n1, n2, total_obs)
    Y[:, :, 1:p] .= 0.0

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
    C = [kron(B[j], A[j]) for j in 1:p]
    return (; Y, A, B, C, Sigma1, Sigma2, sorted_eigs)
end

function generate_var_coefs(n::Int, p::Int; maxiter::Int = 1000)
    for iter in 1:maxiter
        C = [randn(n, n) * 0.2 for _ in 1:p]  # initial scale
        companion_c = make_companion(C)
        evals = eigvals(companion_c)
        rho = maximum(abs.(evals))
        if rho < 0.90
            sorted_eigs = var_eigvals(C)
            return (; C, sorted_eigs)
        end
        # rescale toward stability
        scale = 0.95 / rho
        for j in 1:p
            C[j] .*= scale
        end
        if isstable(C; mineigen = 0.1)
            sorted_eigs = var_eigvals(C)
            return (; C, sorted_eigs)
        end
    end
    error("generate_var_coefs: failed to produce a stable VAR in $maxiter iterations")
end

function simulate_var(
    obs::Int;
    n::Int = 12,
    p::Int = 1,
    C::Union{Nothing, Vector{<:AbstractMatrix}} = nothing,
    Sigma::Union{Nothing, AbstractMatrix} = nothing,
    burnin::Int = 500,
    snr::Real = 1.0,
    maxiter::Int = 1000,
)
    if C === nothing
        C, _ = generate_var_coefs(n, p; maxiter=maxiter)
    else
        n = size(C[1], 1)
    end

    if Sigma === nothing
        evals = var_eigvals(C)
        eigval_coef = maximum(abs.(evals))
        eigval_coef = eigval_coef == 0.0 ? 1.0 : eigval_coef
        eigval_err = eigval_coef / snr
        Sigma = eigval_err * I(n)
    end

    total_obs = obs + burnin
    Y = zeros(Float64, n, total_obs)
    if p >= 1
        Y[:, 1:p] .= 0.0
    end

    mvn = MvNormal(zeros(n), Symmetric(Matrix(Sigma)))
    eps = rand(mvn, total_obs)

    for t in (p+1):total_obs
        yt = eps[:, t]
        for j in 1:p
            yt .+= C[j] * Y[:, t-j]
        end
        Y[:, t] = yt
    end

    Y = Y[:, burnin+1:end]
    sorted_eigs = var_eigvals(C)

    return (; Y, C, Sigma, sorted_eigs)
end

function simulate_two_term_mar(
    obs::Int;
    n1::Int = 3,
    n2::Int = 4,
    eta::Real = 0.0,
    persistence::Real = 0.8,
    A1::Union{Nothing, AbstractMatrix} = nothing,
    B1::Union{Nothing, AbstractMatrix} = nothing,
    A2::Union{Nothing, AbstractMatrix} = nothing,
    B2::Union{Nothing, AbstractMatrix} = nothing,
    Sigma1::Union{Nothing, AbstractMatrix} = nothing,
    Sigma2::Union{Nothing, AbstractMatrix} = nothing,
    burnin::Int = 500,
)
    # Generate coefficient matrices with spectral radius 1 if not provided
    if A1 === nothing
        A1 = randn(n1, n1)
        A1 = A1 / maximum(abs.(eigvals(A1)))  # spectral radius = 1
    end
    if B1 === nothing
        B1 = randn(n2, n2)
        B1 = B1 / maximum(abs.(eigvals(B1)))
    end
    if A2 === nothing
        A2 = randn(n1, n1)
        A2 = A2 / maximum(abs.(eigvals(A2)))
    end
    if B2 === nothing
        B2 = randn(n2, n2)
        B2 = B2 / maximum(abs.(eigvals(B2)))
    end

    C = persistence * kron(B1, A1) + persistence * eta * kron(B2, A2)

    max_eig = maximum(abs.(eigvals(C)))
    if max_eig >= 1.0
        @warn "Process is not stationary: max eigenvalue = $max_eig"
    end

    # Default covariance
    if Sigma1 === nothing
        Sigma1 = Matrix{Float64}(I, n1, n1)
    end
    if Sigma2 === nothing
        Sigma2 = Matrix{Float64}(I, n2, n2)
    end

    total_obs = obs + burnin
    Y = zeros(n1, n2, total_obs)
    matrix_normal = MatrixNormal(zeros(n1, n2), Sigma1, Sigma2)
    errs = rand(matrix_normal, total_obs)

    for t in 2:total_obs
        Y[:, :, t] = persistence * A1 * Y[:, :, t-1] * B1' +
                     persistence * eta * A2 * Y[:, :, t-1] * B2' +
                     errs[t]
    end

    Y = Y[:, :, burnin+1:end]

    return (; Y, A1, B1, A2, B2, C, Sigma1, Sigma2, eta, max_eig)
end
