"""
    enforce_stationarity(C_hat, bias_mats; p=1, dims=nothing)

Kilian Step 1b/2b: if the bias-corrected companion matrix has a root
on or outside the unit circle, shrink the bias correction by a factor
δ (starting at 1, decreasing in steps of 0.01) until stationarity is
achieved. Returns `(C_corrected, δ)`, where δ = 1.0 means the
adjustment did not bind and δ = 0.0 means no correction could be
applied and the raw estimate is returned unchanged.
"""
function enforce_stationarity(C_hat::Vector{<:AbstractMatrix},
                              bias_mats::Vector{<:AbstractMatrix}; p::Int=1,
                              dims::Union{Nothing,Tuple{Int,Int}}=nothing)
    for k in 100:-1:1
        δ = k / 100                       # exact grid, no fp drift
        C_corrected = [C_hat[j] - δ * bias_mats[j] for j in 1:p]
        if dims !== nothing
            _, _, C_corrected = projection(C_corrected, dims)
        end
        comp = make_companion(hcat(C_corrected...))
        if maximum(abs.(eigvals(comp))) < 1.0
            return C_corrected, δ
        end
    end
    return copy(C_hat), 0.0
end

"""
    irf_bootstrap(model, bias_method; boot_runs=1000, ...)

Kilian (1998) bootstrap-after-bootstrap confidence intervals for IRFs.

- `bias_method`: how to estimate the OLS bias (first stage).
  `Analytical()` uses Pope's closed-form expression.
  `Bootstrap(bias_runs=1000)` uses resampling.
- `boot_runs`: number of bootstrap replications for the confidence
  intervals (second stage).
- `shortcut`: if `true`, reuse the first-stage bias estimate for
  all replications instead of re-estimating per replicate.
"""
function irf_bootstrap(model::VAR, bias_method::BiasCorrection;
                       boot_runs::Int=2000,
                       hmax::Int=20,
                       shock_idx::Int=1,
                       ident::Symbol=:reduced,
                       alpha::Float64=0.05,
                       shortcut::Bool=true,
                       precomputed_bias=nothing)
    require_fitted(model)
    n, p, obs = model.n, model.p, model.obs
    U = model.residuals
    # Step 1a: bias-correct the original estimate
    b_hat = if precomputed_bias === nothing
        bias(model, bias_method)
    else
        precomputed_bias
    end
    # Step 1b: enforce stationarity via shrinkage
    C_bc, delta_hat = enforce_stationarity(model.C, b_hat; p)
    # Step 2a: bootstrap from the bias-corrected DGP
    irf_store = zeros(n, hmax + 1, boot_runs)
    delta_store = zeros(boot_runs)
    for m in 1:boot_runs
        Y_star = simulate_bootstrap_sample(C_bc, U, model.data,
                                           p, obs, n)
        boot_model = VAR(Y_star; p=p)
        fit!(boot_model)
        # estimate bias of this replicate
        b_star = shortcut ? b_hat : bias(boot_model, bias_method)
        # Step 2b: enforce stationarity on the replicate
        C_star_bc, delta_store[m] = enforce_stationarity(boot_model.C, b_star; p)
        boot_model.C = C_star_bc
        if ident === :cholesky
            data = boot_model.data
            resid = copy(data[:, (p+1):end])
            for j in 1:p
                resid .-= C_star_bc[j] * data[:, (p+1-j):(end-j)]
            end
            boot_model.residuals = resid
            boot_model.Sigma = (resid * resid') / size(resid, 2)
        end
        irf_star = reduced_form_irf(boot_model; hmax=hmax,
                                    shock_idx=shock_idx, ident=ident)
        irf_store[:, :, m] = irf_star
    end
    # Step 3: percentile intervals
    lo = alpha / 2
    hi = 1 - lo
    ci_lower = mapslices(x -> quantile(x, lo), irf_store; dims=3)[:,:,1]
    ci_upper = mapslices(x -> quantile(x, hi), irf_store; dims=3)[:,:,1]
    # Point IRFs from bias-corrected model
    bc_model = deepcopy(model)
    bc_model.C = C_bc
    if ident === :cholesky
        data = bc_model.data
        resid = copy(data[:, (p+1):end])
        for j in 1:p
            resid .-= C_bc[j] * data[:, (p+1-j):(end-j)]
        end
        bc_model.residuals = resid
        bc_model.Sigma = (resid * resid') / size(resid, 2)
    end
    point_irfs = reduced_form_irf(bc_model; hmax=hmax,
                                  shock_idx=shock_idx, ident=ident)
    return (; irfs=point_irfs, ci_lower, ci_upper, irf_store, delta_store, 
              delta=delta_hat)
end

