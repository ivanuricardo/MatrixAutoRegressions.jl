"""
    enforce_stationarity(C_hat, bias_mats, n, p)

Kilian Step 1b/2b: if the bias-corrected companion matrix has a root
on or outside the unit circle, shrink the bias correction by
multiplying δ (starting at 1, decreasing by 0.01) until stationarity
is achieved. Returns the (possibly shrunk) corrected coefficients.
"""
function enforce_stationarity(C_hat::Vector{<:AbstractMatrix},
                              bias_mats::Vector{<:AbstractMatrix}; p::Int=1,
                              dims::Union{Nothing,Tuple{Int,Int}}=nothing)
    n = size(C_hat[1], 1)
    δ = 1.0
    while δ > 0.0
        C_corrected = [C_hat[j] - δ * bias_mats[j] for j in 1:p]
        if dims !== nothing
            _, _, C_corrected = projection(C_corrected, dims)
        end
        A_big = hcat(C_corrected...)
        comp = make_companion(A_big)
        max_mod = maximum(abs.(eigvals(comp)))
        if max_mod < 1.0
            return C_corrected
        end
        δ -= 0.01
    end
    return copy(C_hat)
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
    C_bc = enforce_stationarity(model.C, b_hat; p)

    # Step 2a: bootstrap from the bias-corrected DGP
    irf_store = zeros(n, hmax + 1, boot_runs)
    for m in 1:boot_runs
        Y_star = simulate_bootstrap_sample(C_bc, U, model.data,
                                           p, obs, n)
        boot_model = VAR(Y_star; p=p)
        fit!(boot_model)

        # estimate bias of this replicate
        b_star = shortcut ? b_hat : bias(boot_model, bias_method)

        # Step 2b: enforce stationarity on the replicate
        C_star_bc = enforce_stationarity(boot_model.C, b_star; p)

        boot_model.C = C_star_bc
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
    point_irfs = reduced_form_irf(bc_model; hmax=hmax,
                                  shock_idx=shock_idx, ident=ident)

    return (; irfs=point_irfs, ci_lower, ci_upper, irf_store)
end

