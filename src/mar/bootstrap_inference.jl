
"""
    impact_weights_to_shock(model, w; block=1, mode=:uniform)

Shock vector `d` for the leading block under the block-Cholesky identification,
normalized so that the `w`-weighted aggregate of the block rises by one unit on
impact.

`mode=:uniform` (default): every series in the block rises by the same amount,
so each series contributes to the aggregate in proportion to its weight.
`mode=:proportional`: the impact responses themselves are proportional to `w`,
so contributions go as the squared weights.
"""
function impact_weights_to_shock(model::AbstractARModel, w::AbstractVector;
                                 block::Int=1, mode::Symbol=:uniform)
    require_fitted(model)
    length(w) == block ||
        throw(ArgumentError("w must have length block = $block"))
    sw = sum(w)
    sw > 0 || throw(ArgumentError("weights must sum to a positive number"))

    # target impact responses of the leading block
    r = if mode === :uniform
        fill(inv(sw), block)
    elseif mode === :proportional
        w ./ dot(w, w)
    else
        throw(ArgumentError("unknown mode: $mode. Valid: :uniform, :proportional"))
    end

    P = get_block_cholesky_innovation_matrix(model; block=block)
    return P[1:block, 1:block] \ r          # d = L₁₁⁻¹ r
end

function irf_bootstrap(model::MAR, bias_method::BiasCorrection;
                       boot_runs::Int=2000,
                       hmax::Int=20,
                       shock_idx::AbstractVector=[1,1],
                       ident::Symbol=:reduced,
                       alpha::Float64=0.05,
                       shortcut::Bool=true,
                       project::Bool=true,
                       block::Int=1,
                       shock_weights::Union{Nothing,AbstractVector}=nothing,
                       impact_weights::Union{Nothing,AbstractVector}=nothing,
                       impact_mode::Symbol=:uniform,
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
    C_bc, delta_hat = enforce_stationarity(model.C, b_hat; p, dims=kron_dims)

    # Step 1c: bias-corrected model, built here rather than after the loop so
    # that the shock normalization below can be taken from it
    bc_model = deepcopy(model)
    bc_model.C = C_bc
    if project
        A_bc, B_bc, _ = projection(C_bc, model.dims)
        bc_model.A = A_bc
        bc_model.B = B_bc
    end
    bc_model.residuals = residuals_from_C(bc_model.data, C_bc, p)
    centered_res_bc = bc_model.residuals .- mean(bc_model.residuals, dims=3)
    sigma_ests = flipflop_covariance(centered_res_bc;
                                     maxiter=model.maxiter, tol=model.tol)
    bc_model.Sigma1 = Symmetric(sigma_ests.sigma1)
    bc_model.Sigma2 = Symmetric(sigma_ests.sigma2)
    bc_model.Sigma = kron(bc_model.Sigma2, bc_model.Sigma1)

    # Step 1d: resolve the shock vector
    if impact_weights !== nothing
        if shock_weights !== nothing
            throw(ArgumentError("pass impact_weights or shock_weights, not both"))
        end
        if ident !== :block_cholesky
            throw(ArgumentError("impact_weights requires ident=:block_cholesky"))
        end
        shock_weights = impact_weights_to_shock(bc_model, impact_weights;
                                                block=block, mode=impact_mode)
    end

    # Step 2a: bootstrap from the bias-corrected DGP
    irf_store = zeros(n, hmax + 1, boot_runs)
    delta_store = zeros(boot_runs)
    for m in 1:boot_runs
        Y_star = simulate_bootstrap_sample(C_bc, vec_residuals, vec_data,
                                           p, obs, n)
        matrix_data = matricize(Y_star, n1, n2)
        boot_model = MAR(matrix_data; p=p, maxiter=model.maxiter, tol=model.tol,
                         method=model.method)
        fit!(boot_model)

        # estimate bias of this replicate
        b_star = shortcut ? b_hat : bias(boot_model, bias_method)

        # Step 2b: enforce stationarity on the replicate
        C_star_bc, delta_store[m] = enforce_stationarity(boot_model.C, b_star; p, dims=kron_dims)
        boot_model.C = C_star_bc
        if ident === :cholesky || ident === :block_cholesky
            if project
                A_star, B_star, _ = projection(C_star_bc, model.dims)
                boot_model.A = A_star
                boot_model.B = B_star
            end
            boot_model.residuals = residuals_from_C(boot_model.data, C_star_bc, p)
            centered_res_boot = boot_model.residuals .- mean(boot_model.residuals, dims=3)
            sig = flipflop_covariance(centered_res_boot;
                                      maxiter=model.maxiter, tol=model.tol,
                                      sigma1=Matrix(boot_model.Sigma1),
                                      sigma2=Matrix(boot_model.Sigma2))
            boot_model.Sigma1 = Symmetric(sig.sigma1)
            boot_model.Sigma2 = Symmetric(sig.sigma2)
            boot_model.Sigma = kron(boot_model.Sigma2, boot_model.Sigma1)
        end
        # recompute d from this replicate's covariance, so that every draw
        # describes the same one-unit impact on the leading block
        sw = impact_weights === nothing ? shock_weights :
             impact_weights_to_shock(boot_model, impact_weights;
                                     block=block, mode=impact_mode)
        irf_star = reduced_form_irf(boot_model; hmax=hmax,
                                    shock_idx=shock_idx, ident=ident, block=block,
                                    shock_weights=sw)
        irf_store[:, :, m] = irf_star
    end

    # Step 3: Efron percentile intervals
    lo = alpha / 2
    hi = 1 - lo
    ci_lower = zeros(n, hmax + 1)
    ci_upper = zeros(n, hmax + 1)
    for i in 1:n, j in 1:(hmax + 1)
        v = @view irf_store[i, j, :]
        ci_lower[i, j] = quantile(v, lo)
        ci_upper[i, j] = quantile(v, hi)
    end

    # Point IRFs from the bias-corrected model
    point_irfs = reduced_form_irf(bc_model; hmax=hmax,
                                  shock_idx=shock_idx, ident=ident, block=block,
                                  shock_weights=shock_weights)

    return (; irfs=point_irfs, ci_lower, ci_upper, irf_store, delta_store,
              delta=delta_hat)
end
