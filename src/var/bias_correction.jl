
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

function bias_correction(model::AbstractARModel, method::BiasCorrection)
    corrected = deepcopy(model)
    bias_correction!(corrected, method)
    return corrected
end

function bias_correction!(model::VAR, method::Analytical)
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

function simulate_bootstrap_sample(C_hat, U, data_vec, p, obs, n)
    boot_idx = rand(1:obs, obs)
    U_star = U[:, boot_idx]
    Y_star = zeros(n, obs + p)
    Y_star[:, 1:p] .= data_vec[:, 1:p]
    for t in (p+1):(obs+p)
        y_t = zeros(n)
        for j in 1:p
            y_t .+= C_hat[j] * Y_star[:, t-j]
        end
        Y_star[:, t] = y_t + U_star[:, t-p]
    end
    return Y_star
end


function bias_correction!(model::VAR, method::Bootstrap)
    require_fitted(model)
    n = model.n
    p = model.p
    obs = model.obs
    C_hat = model.C
    U = model.residuals

    C_sum = [zeros(n, n) for _ in 1:p]

    for m in 1:method.n_boot
        Y_star = simulate_bootstrap_sample(C_hat, U, model.data, p, obs, n)
        boot_coeffs, _ = estimate_var(Y_star; p=p)
        for j in 1:p
            C_sum[j] .+= boot_coeffs[j]
        end
    end

    C_bar = [C_sum[j] / method.n_boot for j in 1:p]
    model.C = [2 * C_hat[j] - C_bar[j] for j in 1:p]
    return model
end
