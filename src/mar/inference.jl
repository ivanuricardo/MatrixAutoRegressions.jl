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

function variance_ols(model::MAR)
    demeaned_data = model.data .- mean(model.data, dims=3)
    vec_data = vectorize(demeaned_data)
    gamma_0 = (vec_data * vec_data') ./ model.obs
    inv_gamma_0 = inv(gamma_0)

    mat_residuals = residuals(model)
    vec_residuals = vectorize(mat_residuals)
    sigma = (vec_residuals * vec_residuals') ./ model.obs
    return kron(inv_gamma_0, sigma)
end

function variance_proj(model::MAR)
    require_fitted(model)
    n1, n2 = model.dims
    var_ols = variance_ols(model)
    perm_mat = permutation_matrix(model.dims)
    xi1 = perm_mat * var_ols * perm_mat'

    alpha = vec(model.A[1])
    beta = vec(model.B[1]) ./ norm(model.B[1])
    first_term = kron(beta * beta', I(n1*n1))
    second_term = kron(I(n2*n2), alpha * alpha')
    third_term = kron(beta * beta', alpha * alpha')
    V1 = first_term + second_term - third_term

    return V1 * xi1 * V1
end

function variance_als(model::MAR)
    require_fitted(model)
    n1, n2 = model.dims
    obs = model.obs
    alpha = vec(model.A[1])
    beta = vec(model.B[1]')
    gamma = vcat(alpha, zeros(n2*n2))

    mat_residuals = residuals(model)
    vec_residuals = vectorize(mat_residuals)
    sigma = (vec_residuals * vec_residuals') ./ model.obs

    WW = zeros(n1*n1 + n2*n2, n1*n1 + n2*n2)
    WSigmaW = zeros(n1*n1 + n2*n2, n1*n1 + n2*n2)

    for i in 1:obs
        first_term = kron(model.data[:, :, i] * model.B[1]', I(n1))
        second_term = kron(I(n2), model.data[:, :, i]' * model.A[1]')
        W = vcat(first_term, second_term)
        WW += W * W'
        WSigmaW += W * sigma *  W'
    end

    WW_scaled = WW ./ obs
    WSigmaW_scaled = WSigmaW ./ obs

    H = WW_scaled + gamma * gamma'
    xi = inv(H) * WSigmaW_scaled * inv(H)
    V = hcat(kron(beta, I(n1*n1)), kron(I(n2*n2), alpha))
    cov_full = V * xi * V'

    return (; cov_full, xi)
end

function variance_mle(model::MAR)
    require_fitted(model)
    n1, n2 = model.dims
    obs = model.obs
    alpha = vec(model.A[1])
    beta = vec(model.B[1]')
    gamma = vcat(alpha, zeros(n2*n2))

    sigma = kron(model.Sigma2, model.Sigma1)

    WW = zeros(n1*n1 + n2*n2, n1*n1 + n2*n2)
    WSigmaW = zeros(n1*n1 + n2*n2, n1*n1 + n2*n2)

    for i in 1:obs
        first_term = kron(model.data[:, :, i] * model.B[1]', I(n1))
        second_term = kron(I(n2), model.data[:, :, i]' * model.A[1]')
        W = vcat(first_term, second_term)
        WW += W * W'
        WSigmaW += W * sigma *  W'
    end

    WSigmaW_scaled = WSigmaW ./ obs

    Htilde = WSigmaW_scaled + gamma * gamma'
    xi = inv(Htilde) * WSigmaW_scaled * inv(Htilde)
    V = hcat(kron(beta, I(n1*n1)), kron(I(n2*n2), alpha))
    cov_full = V * xi * V'


    return (; cov_full, xi)

end

