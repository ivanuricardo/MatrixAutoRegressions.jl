
function irf_bootstrap(model::MAR, bias_method::BiasCorrection;
                       boot_runs::Int=2000,
                       hmax::Int=20,
                       shock_idx::AbstractVector=[1,1],
                       ident::Symbol=:reduced,
                       alpha::Float64=0.05,
                       shortcut::Bool=true,
                       project::Bool=true,
                       precomputed_bias=nothing)
    require_fitted(model)
    p, obs = model.p, model.obs
    n1, n2 = model.dims
    n = n1 * n2
    U = model.residuals
    vec_data = vectorize(model.data)
    vec_residuals = vectorize(U)

    # Step 1a: bias-correct the original estimate
    b_hat = if precomputed_bias === nothing
        bias(model, bias_method)
    else
        precomputed_bias
    end

    # Step 1b: enforce stationarity via shrinkage
    kron_dims = project ? model.dims : nothing
    C_bc = enforce_stationarity(model.C, b_hat; p, dims=kron_dims)

    # Step 2a: bootstrap from the bias-corrected DGP
    irf_store = zeros(n, hmax + 1, boot_runs)
    for m in 1:boot_runs
        Y_star = simulate_bootstrap_sample(C_bc, vec_residuals, vec_data,
                                           p, obs, n)
        matrix_data = matricize(Y_star, n1, n2)
        boot_model = MAR(matrix_data; p=p, maxiter=model.maxiter, tol=model.tol)
        fit!(boot_model)

        # estimate bias of this replicate
        b_star = shortcut ? b_hat : bias(boot_model, bias_method)

        # Step 2b: enforce stationarity on the replicate
        C_star_bc = enforce_stationarity(boot_model.C, b_star; p, dims=kron_dims)
        boot_model.C = C_star_bc
        if ident === :cholesky
            A_star, B_star, _ = projection(C_star_bc, model.dims)
            boot_model.A = A_star
            boot_model.B = B_star
            boot_model.residuals = calculate_residuals(boot_model)
            sig = flipflop_covariance(boot_model.residuals;
                                       maxiter=model.maxiter, tol=model.tol)
            boot_model.Sigma1 = Symmetric(sig.sigma1)
            boot_model.Sigma2 = Symmetric(sig.sigma2)
            boot_model.Sigma = kron(boot_model.Sigma2, boot_model.Sigma1)
        end
        irf_star = reduced_form_irf(boot_model; hmax=hmax,
                                    shock_idx=shock_idx, ident=ident)
        irf_store[:, :, m] = irf_star
    end

    # Step 3: percentile intervals
    lo = alpha / 2
    hi = 1 - lo
    ci_lower = zeros(n, hmax + 1)
    ci_upper = zeros(n, hmax + 1)
    for i in 1:n, j in 1:(hmax + 1)
        v = @view irf_store[i, j, :]
        ci_lower[i, j] = quantile(v, lo)
        ci_upper[i, j] = quantile(v, hi)
    end

    # Point IRFs from bias-corrected model
    bc_model = deepcopy(model)
    bc_model.C = C_bc
    A_bc, B_bc, _ = projection(C_bc, model.dims)
    bc_model.A = A_bc
    bc_model.B = B_bc
    bc_model.residuals = calculate_residuals(bc_model)
    sigma_ests = flipflop_covariance(bc_model.residuals;
                                      maxiter=model.maxiter, tol=model.tol)
    bc_model.Sigma1 = Symmetric(sigma_ests.sigma1)
    bc_model.Sigma2 = Symmetric(sigma_ests.sigma2)
    bc_model.Sigma = kron(bc_model.Sigma2, bc_model.Sigma1)
    point_irfs = reduced_form_irf(bc_model; hmax=hmax,
                                  shock_idx=shock_idx, ident=ident)

    return (; irfs=point_irfs, ci_lower, ci_upper, irf_store)
end
