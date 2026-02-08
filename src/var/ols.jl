
function estimate_var(model::VAR)
    obs = model.obs  # effective number of observations
    data = model.data
    lambda = model.lambda
    p = model.p
    n = model.n
    if obs <= p
        error("Not enough observations: obs = $obs ≤ p = $p")
    end

    if p == 0
        Y = data
        mu = mean(Y; dims=2)
        U = Y .- mu
        dof = obs - 1
        if dof <= 0
            error("Not enough degrees of freedom: K - 1 = $dof ≤ 0")
        end
        Sigma = (U * U') / dof
        return Vector{Matrix{Float64}}(), Sigma, U
    end

    # build X by vertically concatenating p lag-blocks (lag p down to lag 1)
    X = vcat((@view data[:, (lag):(lag+obs-1)] for lag in p:-1:1)...)
    demeaned_X = X .- mean(X, dims = 2)

    Y = data[:, p+1:(obs+p)]
    demeaned_Y = Y .- mean(Y, dims = 2)
    A_hat = (demeaned_Y * demeaned_X') * inv(demeaned_X * demeaned_X' + lambda * I)
    coeffs = [A_hat[:, (n*(i-1)+1):(n*i)] for i in 1:p]

    U = demeaned_Y - A_hat * demeaned_X
    dof = obs - n*p - 1
    if dof <= 0
        error("Not enough degrees of freedom: K - Np = $dof ≤ 0")
    end
    Sigma = (U * U') / dof

    return coeffs, Sigma, U
end

function estimate_var(data::AbstractMatrix{T}; p::Int=1) where T
    n, obs = size(data)
    if obs <= p
        error("Not enough observations: obs = $obs ≤ p = $p")
    end
    K = obs - p

    if p == 0
        μ = mean(data; dims=2)
        U = data .- μ
        dof = obs - 1
        Sigma = (U * U') / dof
        return Vector{Matrix{Float64}}(), Sigma
    end

    # build X by vertically concatenating p lag-blocks (lag p down to lag 1)
    X = vcat((@view data[:, (lag):(lag+K-1)] for lag in p:-1:1)...)
    demeaned_X = X .- mean(X, dims = 2)

    Y = data[:, p+1:obs]
    demeaned_Y = Y .- mean(Y, dims = 2)
    A_hat = demeaned_Y / demeaned_X
    coeffs = [A_hat[:, (n*(i-1)+1):(n*i)] for i in 1:p]

    U = demeaned_Y - A_hat * demeaned_X
    dof = K - n*p - 1
    if dof <= 0
        error("Not enough degrees of freedom: K - Np = $dof ≤ 0")
    end
    Sigma = (U * U') / dof

    return coeffs, Sigma
end

function estimate_var(data::AbstractArray{T}; p::Int=1) where T
    data = vectorize(data)
    n, obs = size(data)
    if obs <= p
        error("Not enough observations: obs = $obs ≤ p = $p")
    end
    K = obs - p

    # build X by vertically concatenating p lag-blocks (lag p down to lag 1)
    X = vcat((@view data[:, (lag):(lag+K-1)] for lag in p:-1:1)...)
    demeaned_X = X .- mean(X, dims = 2)

    Y = data[:, (p+1):obs]
    demeaned_Y = Y .- mean(Y, dims = 2)
    A_hat = demeaned_Y / demeaned_X
    coeffs = [A_hat[:, (n*(i-1)+1):(n*i)] for i in 1:p]

    U = demeaned_Y - A_hat * demeaned_X
    dof = K - n*p - 1
    if dof <= 0
        error("Not enough degrees of freedom: K - Np = $dof ≤ 0")
    end
    Sigma = (U * U') / dof

    return coeffs, Sigma
end

function asymptotic_variance(model::VAR)
    require_fitted(model)
    n = model.n
    p = model.p
    obs = model.obs
    Sigma = model.Sigma
    data = model.data

    X = vcat((@view data[:, (lag):(lag+obs-1)] for lag in p:-1:1)...)
    X = X .- mean(X, dims = 2)

    XX_inv = I / (X * X')

    return kron(XX_inv, Sigma)
end

function std_errors(model::VAR)
    p = model.p
    n = model.n
    V = asymptotic_variance(model)
    se_vec = sqrt.(diag(V))
    se = [reshape(se_vec[(n*(i-1)*n+1):(n*i*n)], n, n) for i in 1:p]
    return se
end

function companion_states(data::AbstractMatrix, p::Int)
    # data: n × T
    n, obs = size(data)
    S = vcat((@view data[:, (p+1 - (i-1)):(obs - (i-1))] for i in 1:p)...)
    return S
end

function pope_kilian_bias(A_hat::Vector{<:AbstractMatrix},
                          Sigma::Matrix{Float64},
                          Y::Matrix{Float64})

    A_hat = hcat(A_hat...)
    n, np = size(A_hat)
    p = div(np, n)
    obs = size(Y, 2) - p

    k = n * p
    C = make_companion(A_hat)

    F = eigen(C)
    evals = F.values

    sum_eigvals = zeros(ComplexF64, k, k)
    for λ in evals
        sum_eigvals += λ * ((I - λ * C') \ I)
    end

    sum_eigvals = real(sum_eigvals)

    first_term = I / (I - C')
    second_term = C' * (I / (I - (C')^2))

    Sigma_e = zeros(k, k)
    Sigma_e[1:n, 1:n] .= Sigma

    S = companion_states(Y, p)
    S_demeaned = S .- mean(S, dims=2)
    Gamma0 = (S_demeaned * S_demeaned') / obs

    # Gamma_x0 = lyap(C, Sigma_e)  # Solves lyapunov equation
    inv_gamma = I / Gamma0

    bias = Sigma_e * (first_term + second_term + sum_eigvals) * inv_gamma

    return bias

end

function pope_kilian_bias(model::VAR)
    A_hat = model.C
    Sigma = model.Sigma
    Y = model.data
    bias = pope_kilian_bias(A_hat, Sigma, Y)
    return bias
end

function bias_correction!(model::VAR)
    require_fitted(model)
    C_hat = model.C
    n = model.n
    p = model.p
    obs = model.obs
    bias_full = pope_kilian_bias(model)
    b_top = bias_full[1:n, :]

    b_mats = [b_top[:, (j-1)*n+1 : j*n] for j in 1:p]

    C_corrected = [C_hat[j] + (b_mats[j] / obs) for j in 1:p]

    model.C = C_corrected

    return model
end

