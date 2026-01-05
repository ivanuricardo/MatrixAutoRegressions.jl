
function irf_ma(model::VAR; hmax::Int=1, ident::Symbol=:reduced)
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
    elseif ident == :reduced
        nothing
    else
        error("Unknown identification scheme: $ident. Valid: :reduced, :cholesky")
    end

    return theta
end

function get_cholesky_innovation_matrix(model::VAR)
    require_fitted(model)
    Σ = model.Sigma
    return cholesky(Symmetric(Σ)).L
end

function reduced_form_irf(model::VAR; hmax::Int=1,
                          shock_idx::Int=1,
                          theta=nothing,
                          ident::Symbol=:reduced)

    require_fitted(model)

    if theta === nothing
        theta = irf_ma(model; hmax=hmax, ident=ident)
    end

    n = size(theta[1],1)
    if shock_idx < 1 || shock_idx > n
        throw(ArgumentError("shock_idx out of bounds"))
    end

    e = zeros(eltype(theta[1]), n)
    e[shock_idx] = one(eltype(theta[1]))

    irf = zeros(eltype(theta[1]), n, hmax+1)
    for h in 0:hmax
        irf[:,h+1] = theta[h+1] * e
    end

    return irf
end

function all_irf_variances(model::VAR, theta::Vector{<:AbstractMatrix}; hmax::Integer=1, ident::Symbol=:reduced)
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

    if ident == :cholesky
        all_irf_var[1] = revech_lower(diag(avar_cholesky(model))) / obs
        return all_irf_var
    elseif ident == :reduced
        return all_irf_var
    else
        error("Unknown identification scheme: $ident. Valid: :reduced, :cholesky")
    end


end

function irf_variance(model::VAR, theta::Vector{<:AbstractMatrix};
                      hmax::Integer=1, shock_idx::Int=1, ident::Symbol=:reduced)
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

function irf(model::VAR; hmax::Integer=1,
             shock_idx::Int=1,
             ident::Symbol=:reduced)

    theta = irf_ma(model; hmax=hmax, ident=ident)

    irfs = reduced_form_irf(model;
        hmax=hmax,
        shock_idx=shock_idx,
        theta=theta,
        ident=ident
    )

    irf_var = irf_variance(model, theta; hmax=hmax, shock_idx=shock_idx, ident=ident)

    irf_cov = copy(irf_var)
    if ident == :cholesky
        irf_se = sqrt.(irf_cov)
        return (; irfs, irf_var, irf_cov, irf_se)
    end

    irf_se = sqrt.(irf_cov)
    return (; irfs, irf_var, irf_cov, irf_se)

end
