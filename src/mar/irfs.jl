
function irf_ma(model::MAR; hmax::Int=1)
    C = model.C
    n = size(C[1],1)
    p = length(C)
    T = eltype(C[1])
    I_n = Matrix{T}(I, n, n)

    theta = Vector{Matrix{T}}(undef, hmax+1)  # Θ[1] = Θ_0, Θ[h+1] = Θ_h
    theta[1] = I_n
    for h in 1:hmax
        M = zeros(T, n, n)
        for j in 1:min(p,h)
            M .+= C[j] * theta[h+1-j]
        end
        theta[h+1] = M
    end
    return theta
end

function make_g(theta::Vector{Matrix{T}}, C::Vector{Matrix{T}}, h::Int) where T
    p = length(C)
    n = size(C[1], 1)
    g = zeros(T, n^2, n^2 * p)
    @assert length(theta) >= h+1 "theta must contain Θ_0..Θ_h stored as theta[1]==Θ_0"

    for k in 1:h
        # build left block row [Θ_{h−k}ᵀ  Θ_{h−k−1}ᵀ … Θ_{h−k−p+1}ᵀ]
        block_matrix = zeros(n, n * p)
        for i in 1:p
            idx = h - k - i + 1   # corresponds to Θ_{h−k−i+1}
            if idx < 0  # past IRFs are zeros
                continue
            end
            block_idx = (n*(i-1) + 1):(n*i)
            block_matrix[:, block_idx] .= theta[idx+1]'
        end
        # accumulate term Θ_{k−1}
        g .+= kron(block_matrix, theta[k])
    end
    return g
end

function all_irf_variances(model::MAR, theta::Vector{<:AbstractMatrix}; hmax::Integer=1)
    require_fitted(model)
    T = eltype(model.C[1])
    n1, n2 = model.dims
    n = prod(n1*n2)
    p = model.p
    C = model.C
    cov_full, _ = asymptotic_variance(model)
    Q = Q_matrix(n1, n2, p)
    kronQ = kron(Q, I(n1*n2))
    selected_cov = kronQ * cov_full * kronQ'

    all_irf_var = Vector{Matrix{T}}(undef, hmax+1)
    all_irf_var[1] = zeros(T, n, n)
    for h in 1:hmax
        G = make_g(theta, C, h)

        diag_variance = diag(G * selected_cov * G')
        all_irf_var[h+1] = reshape(diag_variance, (n1*n2, n1*n2))

    end
    return all_irf_var

end

function irf_variance(model::MAR, theta::Vector{<:AbstractMatrix};
                      hmax::Integer=1, shock_idx::Vector=[1,1])
    require_fitted(model)
    T = eltype(model.C[1])

    n1, n2 = model.dims
    n = size(model.C[1], 1)
    vec_shock_idx = shock_idx[1] + (shock_idx[2]-1) * n1
    if vec_shock_idx < 1 || vec_shock_idx > n
        throw(ArgumentError("shock_idx out of bounds"))
    end

    all_vars = all_irf_variances(model, theta; hmax=hmax)  # Vector of (n × n) variance matrices
    irf_var = zeros(T, n, hmax+1)

    for h in 0:hmax
        varmat = all_vars[h+1]  # varmat[i,j] = var(response i to shock j) at horizon h
        irf_var[:, h+1] = varmat[:, vec_shock_idx]   # select column for the chosen shock
    end

    return irf_var
end

# helper: return B such that Σ_u = B * B'
function get_cholesky_innovation_matrix(model::AbstractARModel)
    if model isa VAR
        require_fitted(model)
        sigma = model.Sigma
        ch = cholesky(Symmetric(sigma))
        return ch.L
    elseif model isa MAR
        require_fitted(model)
        sigma1 = model.Sigma1
        sigma2 = model.Sigma2
        ch1 = cholesky(Symmetric(sigma1)).L
        ch2 = cholesky(Symmetric(sigma2)).L
        # Cov(vec(E)) = kron(σ2, σ1). Cholesky of kron = kron(ch2, ch1)
        return kron(ch2, ch1)
    else
        error("Unsupported model type for cholesky identification")
    end
end

# Build IRFs to a unit shock in variable `shock_idx`.
# ident=:reduced (default) -> reduced-form raw innovation
# ident=:cholesky -> orthogonalized structural shock via Cholesky
# returns matrix irf where columns are horizons 0..H and rows are variables 1..n
function reduced_form_irf(model::MAR; hmax::Int=1, shock_idx::Vector=[1,1], theta=nothing, ident::Symbol=:reduced)
    require_fitted(model)
    if theta === nothing
        theta = irf_ma(model; hmax=hmax)
    end

    n1, n2 = model.dims
    vec_shock_idx = shock_idx[1] + (shock_idx[2] - 1) * n1
    n = size(theta[1], 1)
    if vec_shock_idx < 1 || vec_shock_idx > n
        throw(ArgumentError("shock_idx out of bounds"))
    end

    # unit vector in reduced-form innovation space
    e = zeros(eltype(theta[1]), n)
    e[vec_shock_idx] = one(eltype(theta[1]))

    # if cholesky identification, map structural unit shock -> reduced-form shock
    if ident === :cholesky
        B = get_cholesky_innovation_matrix(model)
        e_trans = B * e
    elseif ident === :reduced
        e_trans = e
    else
        error("Unknown identification scheme: $ident. Valid: :reduced, :cholesky")
    end

    irf = zeros(eltype(theta[1]), n, hmax+1)
    for h in 0:hmax
        irf[:, h+1] = theta[h+1] * e_trans
    end
    return irf
end

function irf(model::MAR; hmax::Integer=1, shock_idx::Vector=[1,1], ident::Symbol=:reduced)
    obs = model.obs
    theta = irf_ma(model; hmax)
    irfs = reduced_form_irf(model; hmax=hmax, shock_idx=shock_idx, theta=theta, ident=ident)

    irf_var = irf_variance(model, theta; hmax=hmax, shock_idx=shock_idx)
    irf_cov = irf_var ./ obs
    irf_se = sqrt.(irf_cov)

    return (; irfs, irf_var, irf_cov, irf_se)
end

