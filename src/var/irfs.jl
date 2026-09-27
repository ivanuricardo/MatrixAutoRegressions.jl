
function irf_ma(model::VAR; hmax::Int=20, ident::Symbol=:reduced, block::Int=1)
    require_fitted(model)
    C = model.C
    n = size(C[1],1)
    p = length(C)
    T = eltype(C[1])
    I_n = Matrix{T}(I, n, n)

    theta = Vector{Matrix{T}}(undef, hmax+1)  # Θ[1] = Θ₀
    theta[1] = I_n

    # MA recursion
    for h in 1:hmax
        M = zeros(T, n, n)
        for j in 1:min(p,h)
            M .+= C[j] * theta[h+1-j]
        end
        theta[h+1] = M
    end

    # Apply identification
    if ident == :cholesky
        B = get_cholesky_innovation_matrix(model)
        for h in 0:hmax
            theta[h+1] = theta[h+1] * B
        end
    elseif ident == :block_cholesky
        B = get_block_cholesky_innovation_matrix(model; block=block)
        for h in 0:hmax
            theta[h+1] = theta[h+1] * B
        end
    elseif ident == :reduced
        nothing
    else
        error("Unknown identification scheme: $ident. Valid: :reduced, :cholesky, :block_cholesky")
    end

    return theta
end

function get_cholesky_innovation_matrix(model::VAR)
    require_fitted(model)
    Σ = model.Sigma
    return cholesky(Symmetric(Σ)).L
end

# Build IRFs to a shock in the structural innovation space.
# shock_idx    -> unit shock to a single structural innovation
# shock_weights -> δ* over the leading block, as in Billio et al. eq. (12);
#                  requires ident=:block_cholesky
function reduced_form_irf(model::VAR; hmax::Int=20,
                          shock_idx::Int=1,
                          theta=nothing,
                          ident::Symbol=:reduced,
                          block::Int=1,
                          shock_weights::Union{Nothing,AbstractVector}=nothing)

    require_fitted(model)

    if theta === nothing
        theta = irf_ma(model; hmax=hmax, ident=ident, block=block)
    end

    n = size(theta[1],1)
    T = eltype(theta[1])

    # shock vector in structural shock space; theta already carries the
    # impact matrix, so this is δ* (or a unit vector) rather than an innovation
    e = zeros(T, n)
    if shock_weights === nothing
        if shock_idx < 1 || shock_idx > n
            throw(ArgumentError("shock_idx out of bounds"))
        end
        if ident === :block_cholesky && shock_idx > block
            throw(ArgumentError("shock_idx must lie in the leading block of size $block"))
        end
        e[shock_idx] = one(T)
    else
        if ident !== :block_cholesky
            throw(ArgumentError("shock_weights requires ident=:block_cholesky"))
        end
        if length(shock_weights) != block
            throw(ArgumentError("shock_weights must have length block = $block"))
        end
        e[1:block] .= shock_weights
    end

    irf = zeros(T, n, hmax+1)
    for h in 0:hmax
        irf[:,h+1] = theta[h+1] * e
    end

    return irf
end

function all_irf_variances(model::VAR, theta::Vector{<:AbstractMatrix}; hmax::Integer=20, ident::Symbol=:reduced)
    require_fitted(model)
    T = eltype(model.C[1])
    n = model.n
    p = model.p
    C = model.C
    obs = model.obs
    cov_full = asymptotic_variance(model)

    all_irf_var = Vector{Matrix{T}}(undef, hmax+1)
    all_irf_var[1] = zeros(T, n, n)
    for h in 1:hmax
        G = make_g(theta, C, h)

        diag_variance = diag(G * cov_full * G')
        all_irf_var[h+1] = reshape(diag_variance, (n, n))

    end

    # the block Cholesky factor equals the Cholesky factor of Σ, so the h = 0
    # asymptotic variance is the same for both schemes
    if ident == :cholesky || ident == :block_cholesky
        all_irf_var[1] = revech_lower(diag(avar_cholesky(model))) / obs
        return all_irf_var
    elseif ident == :reduced
        return all_irf_var
    else
        error("Unknown identification scheme: $ident. Valid: :reduced, :cholesky, :block_cholesky")
    end


end

function irf_variance(model::VAR, theta::Vector{<:AbstractMatrix};
                      hmax::Integer=20, shock_idx::Int=1, ident::Symbol=:reduced)
    require_fitted(model)
    T = eltype(model.C[1])

    n = model.n
    if shock_idx < 1 || shock_idx > n
        throw(ArgumentError("shock_idx out of bounds"))
    end

    all_vars = all_irf_variances(model, theta; hmax=hmax, ident=ident)
    irf_var = zeros(T, n, hmax+1)

    for h in 0:hmax
        varmat = all_vars[h+1]  # varmat[i,j] = var(response i to shock j) at horizon h
        irf_var[:, h+1] = varmat[:, shock_idx]   # select column for the chosen shock
    end

    return irf_var
    
end

# ---------------------------------------------------------------------------
# Delta method for impact-normalized block shocks (impact_weights)
#
# Under ident = :block_cholesky, the shock resolved from impact_weights moves the
# leading block by a fixed vector r on impact (r = ι / sum(w) under :uniform,
# r = w / w'w under :proportional), so its impact vector is
#     v = P[:, 1:block] d = [r; Σ_ba Σ_aa⁻¹ r],
# and the response at horizon h is Θ_h v. It depends on the coefficients through
# Θ_h and on the covariance through K = Σ_ba Σ_aa⁻¹. The coefficient and
# covariance estimators are asymptotically independent, so the delta-method
# variance is the sum of a propagation term and an impact term. Shared by the
# VAR and MAR methods of irf; the MAR-specific pieces are in mar/irfs.jl.
# ---------------------------------------------------------------------------

# Symmetric basis matrices in vech order (column-major lower triangle, as in
# vech), so that dΣ = Σ_k dσ_k E_k for σ = vech(Σ)
function sym_basis(k::Int)
    basis = Matrix{Float64}[]
    for j in 1:k, i in j:k
        E = zeros(k, k)
        E[i, j] = 1.0
        E[j, i] = 1.0
        push!(basis, E)
    end
    return basis
end

# K = Σ_ba Σ_aa⁻¹ and a = Σ_aa⁻¹ r at the estimate
function impact_derivative_terms(Σ::AbstractMatrix, r::AbstractVector, block::Int)
    Σ = Matrix(Σ)
    Saa = Symmetric(Σ[1:block, 1:block])
    K = Matrix((Saa \ Σ[1:block, (block+1):end])')
    a = Saa \ r
    return K, a
end

# Derivative of v = [r; Σ_ba Σ_aa⁻¹ r] in the direction dΣ, with r held fixed:
# d(Σ_ba Σ_aa⁻¹ r) = dΣ_ba a - K dΣ_aa a
function impact_direction(dΣ::AbstractMatrix, K::AbstractMatrix, a::AbstractVector,
                          block::Int)
    dv = zeros(size(dΣ, 1))
    dv[(block+1):end] = dΣ[(block+1):end, 1:block] * a - K * (dΣ[1:block, 1:block] * a)
    return dv
end

# guards the closed form of v against changes to the block Cholesky factor
function check_impact_vector(v::AbstractVector, K::AbstractMatrix, block::Int)
    isapprox(v[(block+1):end], K * v[1:block]; rtol=1e-6, atol=1e-10 * norm(v)) ||
        error("impact vector does not equal [r; Σ_ba Σ_aa⁻¹ r]; the delta method for " *
              "impact_weights assumes this form")
    return nothing
end

"""
    impact_irf_variance(theta, C, v, cov_C, Jv, cov_sigma; hmax)

Delta-method variance of Θ_h v for h = 0, ..., hmax. `theta` holds the reduced-form
Θ_0, ..., Θ_hmax, `cov_C` the covariance of vec([C_1 ⋯ C_p]), `Jv` the Jacobian
∂v/∂σ' and `cov_sigma` the covariance of σ̂, both covariances on the finite-sample
scale. Returns an n × (hmax + 1) matrix on the finite-sample scale.
"""
function impact_irf_variance(theta, C, v::AbstractVector, cov_C::AbstractMatrix,
                             Jv::AbstractMatrix, cov_sigma::AbstractMatrix;
                             hmax::Integer)
    n = length(v)
    theta = [Matrix{Float64}(Θ) for Θ in theta]
    C = [Matrix{Float64}(c) for c in C]
    Sv = Jv * cov_sigma * Jv'                                # covariance of v̂
    Vt = kron(reshape(v, 1, n), Matrix{Float64}(I, n, n))    # vec(Θ v) = (v' ⊗ I) vec(Θ)

    irf_var = zeros(n, hmax + 1)
    irf_var[:, 1] = diag(Sv)                                 # Θ_0 = I
    for h in 1:hmax
        M = Vt * make_g(theta, C, h)                         # ∂vec(Θ_h v)/∂vec(C)'
        irf_var[:, h+1] = diag(M * cov_C * M') .+ diag(theta[h+1] * Sv * theta[h+1]')
    end
    return max.(irf_var, 0.0)                                # round-off at exact zeros
end

# Covariance of vech(Σ̂) on the finite-sample scale, from the fourth moments of the
# residuals (n × T), so that it does not rely on Gaussian innovations
function vech_sigma_covariance(U::AbstractMatrix)
    n, T = size(U)
    W = reduce(hcat, [vech(U[:, t] * U[:, t]') for t in 1:T])
    W = W .- mean(W, dims=2)
    return (W * W') ./ T^2
end

function impact_irf(model::VAR, impact_weights::AbstractVector;
                    hmax::Integer=20, block::Int=1, mode::Symbol=:uniform)
    require_fitted(model)
    n = model.n

    d = impact_weights_to_shock(model, impact_weights; block=block, mode=mode)
    irfs = reduced_form_irf(model; hmax=hmax, ident=:block_cholesky, block=block,
                            shock_weights=d)

    P = get_block_cholesky_innovation_matrix(model; block=block)
    v = P[:, 1:block] * d
    K, a = impact_derivative_terms(model.Sigma, v[1:block], block)
    check_impact_vector(v, K, block)
    Jv = reduce(hcat, [impact_direction(E, K, a, block) for E in sym_basis(n)])

    theta = irf_ma(model; hmax=hmax)                   # reduced form
    cov_C = asymptotic_variance(model)                 # (XX')⁻¹ ⊗ Σ, finite-sample scale
    cov_sigma = vech_sigma_covariance(model.residuals)
    irf_var = impact_irf_variance(theta, model.C, v, cov_C, Jv, cov_sigma; hmax=hmax)

    return (; irfs, irf_var, irf_cov=copy(irf_var), irf_se=sqrt.(irf_var))
end

function irf(model::VAR; hmax::Integer=20,
             shock_idx::Int=1,
             ident::Symbol=:reduced,
             block::Int=1,
             shock_weights::Union{Nothing,AbstractVector}=nothing,
             impact_weights::Union{Nothing,AbstractVector}=nothing,
             impact_mode::Symbol=:uniform)

    require_fitted(model)

    # impact-normalized block shock: point estimate and delta-method variance
    if impact_weights !== nothing
        if shock_weights !== nothing
            throw(ArgumentError("pass impact_weights or shock_weights, not both"))
        end
        if ident !== :block_cholesky
            throw(ArgumentError("impact_weights requires ident=:block_cholesky"))
        end
        return impact_irf(model, impact_weights; hmax=hmax, block=block, mode=impact_mode)
    end

    theta = irf_ma(model; hmax=hmax, ident=ident, block=block)

    irfs = reduced_form_irf(model;
        hmax=hmax,
        shock_idx=shock_idx,
        theta=theta,
        ident=ident,
        block=block,
        shock_weights=shock_weights
    )

    # irf_variance selects a single column, so it does not describe a weighted
    # composite shock; the bootstrap path is unaffected
    if shock_weights !== nothing
        return (; irfs)
    end

    irf_var = irf_variance(model, theta; hmax=hmax, shock_idx=shock_idx, ident=ident)

    irf_cov = copy(irf_var)
    irf_se = sqrt.(irf_cov)

    return (; irfs, irf_var, irf_cov, irf_se)

end
