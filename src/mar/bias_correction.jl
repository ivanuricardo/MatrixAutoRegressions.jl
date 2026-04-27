
function bias_correction!(model::MAR, method::Bootstrap)
    require_fitted(model)
    n1, n2 = model.dims
    n = n1 * n2
    p = model.p
    obs = model.obs
    C_hat = model.C

    data_vec = vectorize(model.data)
    U = vectorize(model.residuals)

    C_sum = [zeros(n, n) for _ in 1:p]

    for m in 1:method.n_boot
        Y_star = simulate_bootstrap_sample(C_hat, U, data_vec, p, obs, n)

        if method.restricted
            Y_star_mat = matricize(Y_star, n1, n2)
            boot_model = MAR(Y_star_mat; p=p)
            fit!(boot_model)
            boot_C = boot_model.C
        else
            boot_C, _ = estimate_var(Y_star; p=p)
        end

        for j in 1:p
            C_sum[j] .+= boot_C[j]
        end
    end

    C_bar = [C_sum[j] / method.n_boot for j in 1:p]
    model.C = [2 * C_hat[j] - C_bar[j] for j in 1:p]
    return model
end
