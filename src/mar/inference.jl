
sym(X) = (X + X') / 2

function structure_lagged_data(model::MAR)
    data = model.data
    n1, n2 = model.dims
    obs = model.obs
    p = model.p

    structured_data = zeros(n1*p, n2*p, obs)
    structured_data[1:n1, 1:n2, 1:obs] .= data[:, :, p:end-1]

    for i in 1:(p-1)
        first_idx = (n1*i+1):n1*(i+1)
        second_idx = (n2*i+1):n2*(i+1)
        prepared_data = data[:, :, (p-i):(end-1-i)]
        structured_data[first_idx, second_idx, 1:obs] .= prepared_data
    end
    return structured_data

end

function selection_matrix(idx::Int, model::MAR)
    n1, n2 = model.dims
    p = model.p
    E = zeros(n1 * p, n1)
    F = zeros(n2 * p, n2)

    rows_E = ((idx - 1) * n1 + 1):(idx * n1)
    E[rows_E, 1:n1] = I(n1)

    rows_F = ((idx - 1) * n2 + 1):(idx * n2)
    F[rows_F, 1:n2] = I(n2)

    sel = sparse(kron(kron(F, E)', I(n1*n2)))
    return sel
end

function get_c_stderr(model::MAR)
    require_fitted(model)
    p = model.p
    obs = model.obs - p
    n1, n2 = model.dims
    asympt_variance = asymptotic_variance(model)
    cov_full = asympt_variance.cov_full
    C_stderr = Vector{AbstractMatrix{Float64}}(undef, p)

    for i in 1:p
        sel = selection_matrix(i, model)
        C_var = sel * cov_full * sel'
        c_diag = sqrt.(diag(C_var) ./ obs)
        C_stderr[i] = reshape(c_diag, n1*n2, n1*n2)
    end

    return C_stderr
end

function std_errors(model::MAR)
    require_fitted(model)
    p = model.p
    obs = model.obs - p
    n1, n2 = model.dims
    asympt_variance = asymptotic_variance(model)
    xi = asympt_variance.xi
    vec_stderrs = sqrt.(diag(xi) ./ obs)

    A_se = vec_stderrs[1:n1*n1*p]
    B_se = vec_stderrs[(n1*n1*p+1):end]

    A_stderr = Vector{AbstractMatrix{Float64}}(undef, p)
    B_stderr = Vector{AbstractMatrix{Float64}}(undef, p)

    idx1 = 1:n1*n1
    idx2 = 1:n2*n2
    for i in 1:p

        A_stderr[i] = reshape(A_se[idx1], n1, n1)
        B_stderr[i] = reshape(B_se[idx2], n2, n2)

        shift_idx1 = n1*n1
        shift_idx2 = n2*n2
        idx1 = idx1 .+ shift_idx1
        idx2 = idx2 .+ shift_idx2
    end
    C_stderr = get_c_stderr(model)

    return (; A_stderr, B_stderr, C_stderr)
end

function asymptotic_variance(model::MAR)
    if model.method == :proj
        return variance_proj(model)
    end

    if model.method == :als
        return variance_als(model)
    end

    if model.method == :mle
        return variance_mle(model)
    end
end

function stack_coefs(model::MAR)
    A = hcat(model.A...)
    B = hcat(model.B...)
    return (; A, B)
end

# Variance proj only works with 1 lag
function variance_proj(model::MAR)
    require_fitted(model)
    n1, n2 = model.dims
    p = model.p
    Astack, Bstack = stack_coefs(model)
    var_model = VAR(vectorize(model.data))
    fit!(var_model)
    var_ols = asymptotic_variance(var_model)
    perm_mat = kron(I(p^2), permutation_matrix(model.dims))
    xi = perm_mat * var_ols * perm_mat'

    alpha = vec(Astack)
    beta = vec(Bstack) ./ norm(Bstack)
    first_term = kron(beta * beta', I(n1*n1*p))
    second_term = kron(I(n2*n2*p), alpha * alpha')
    third_term = kron(beta * beta', alpha * alpha')
    V1 = first_term + second_term - third_term
    cov_full = V1 * xi * V1'

    return (; cov_full, xi, V1)
end

function construct_gamma(model::MAR)
    n1, n2 = model.dims
    p = model.p
    T = eltype(model.A[1])

    gamma_top = zeros(T, n1 * n1 * p, p)

    for j in 1:p
        gamma_top[(j-1)*n1*n1 + 1 : j*n1*n1, j] .= vec(model.A[j])
    end
    gamma_bottom = zeros(n2*n2*p, p)
    gamma = vcat(gamma_top, gamma_bottom)

    return gamma
end

function construct_WW(model::MAR, sigma::AbstractMatrix)

    data = structure_lagged_data(model)
    data = data .- mean(data, dims = 3)

    n1, n2 = model.dims
    p = model.p
    obs = model.obs - p
    Astack, Bstack = stack_coefs(model)

    WW = zeros(n1*n1*p + n2*n2*p, n1*n1*p + n2*n2*p)
    WSigmaW = zeros(n1*n1*p + n2*n2*p, n1*n1*p + n2*n2*p)

    for i in 1:obs
        first_term = kron(data[:, :, i] * Bstack', I(n1))
        second_term = kron(I(n2), data[:, :, i]' * Astack')
        W = vcat(first_term, second_term)
        WW += W * W'
        WSigmaW += W * sigma *  W'
    end

    WW_scaled = WW ./ obs
    WSigmaW_scaled = WSigmaW ./ obs
    return (; WW_scaled, WSigmaW_scaled)
end

function variance_als(model::MAR)
    require_fitted(model)

    data = structure_lagged_data(model)
    data = data .- mean(data, dims = 3)

    n1, n2 = model.dims
    p = model.p
    obs = model.obs - p
    Astack, Bstack = stack_coefs(model)
    alpha = vec(Astack)
    beta = vec(Bstack)
    gamma = construct_gamma(model)

    sigma = model.Sigma

    WW_scaled, WSigmaW_scaled = construct_WW(model, sigma)

    H = WW_scaled + gamma * gamma'
    F = H \ WSigmaW_scaled
    xi_unperm = F * (H \ I)
    xi_unperm = sym(xi_unperm)

    K = large_commutation_matrix(Bstack', n1, p)
    xi = K * xi_unperm * K'

    P = vectorize_kronecker(Bstack, Astack)
    V = hcat(kron(beta, I(n1*n1*p)), kron(I(n2*n2*p), alpha))

    cov_full = P * (V * xi * V') * P'

    return (; cov_full, xi)
end

function variance_mle(model::MAR)
    require_fitted(model)

    data = structure_lagged_data(model)
    data = data .- mean(data, dims = 3)

    n1, n2 = model.dims
    p = model.p
    obs = model.obs - p
    Astack, Bstack = stack_coefs(model)
    alpha = vec(Astack)
    beta = vec(Bstack)
    gamma = construct_gamma(model)

    sigma = kron(model.Sigma2 \ I, model.Sigma1 \ I)

    _, WSigmaW_scaled = construct_WW(model, sigma)

    Htilde = WSigmaW_scaled + gamma * gamma'
    F = Htilde \ WSigmaW_scaled
    xi_unperm = F * (Htilde \ I)
    xi_unperm = sym(xi_unperm)

    K = large_commutation_matrix(Bstack', n1, p)
    xi = K * xi_unperm * K'

    P = vectorize_kronecker(Bstack, Astack)
    V = hcat(kron(beta, I(n1*n1*p)), kron(I(n2*n2*p), alpha))

    cov_full = P * (V * xi * V') * P'

    return (; cov_full, xi)

end

