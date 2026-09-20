
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

function irf(model::VAR; hmax::Integer=20,
             shock_idx::Int=1,
             ident::Symbol=:reduced,
             block::Int=1,
             shock_weights::Union{Nothing,AbstractVector}=nothing,
             impact_weights::Union{Nothing,AbstractVector}=nothing)

    require_fitted(model)

    # resolve the shock vector
    if impact_weights !== nothing
        if shock_weights !== nothing
            throw(ArgumentError("pass impact_weights or shock_weights, not both"))
        end
        if ident !== :block_cholesky
            throw(ArgumentError("impact_weights requires ident=:block_cholesky"))
        end
        shock_weights = impact_weights_to_shock(model, impact_weights; block=block)
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
