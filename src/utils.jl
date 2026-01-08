
function vectorize(data::AbstractArray{T}) where T
    n1, n2, obs = size(data)
    return reshape(data, n1*n2, obs)
end

function matricize(data::AbstractMatrix{T}, n1::Int, n2::Int) where T
    obs = size(data, 2)
    return reshape(data, n1, n2, obs)
end

"""
    make_companion(B::AbstractMatrix{T}) where {T}
    
Create the VAR companion matrix.

Given a VAR of the form yₜ = b₀ + B₁ y_t-1 + … + Bₚy_t-p + εₜ


Create the companion matrix.
Thus, ``B`` is a ``np\times np`` matrix. 

## Arguments

-`B::AbstractMatrix{T}`: Lag matrix in the form required for a `VAR` model. See
    the documentation of `VAR`.


## References

- Kilian, L., & Lütkepohl, H. (2017). Structural Vector Autoregressive Analysis:
  (1st ed.). Cambridge University Press. https://doi.org/10.1017/9781108164818


"""
function make_companion(C::AbstractMatrix{T}) where {T}
    n = Int(size(C, 1))
    p = Int(size(C, 2) / n)
    ident = diagm(fill(T(1), n * (p - 1)))
    companionlower = hcat(ident, zeros(n * (p - 1), n))
    companion = vcat(C, companionlower)
    return companion
end

function make_companion(C::Vector{<:AbstractMatrix{T}}) where {T}
    C = hcat(C...)
    n = Int(size(C, 1))
    p = Int(size(C, 2) / n)
    ident = diagm(fill(T(1), n * (p - 1)))
    companionlower = hcat(ident, zeros(n * (p - 1), n))
    companion = vcat(C, companionlower)
    return companion
end

function make_companion(A::Vector{<:AbstractMatrix{T}}, B::Vector{<:AbstractMatrix{T}}) where {T}
    C = map(kron, B, A)
    return make_companion(C)
end

function make_companion_var(model::MAR)
    cov_full, _ = asymptotic_variance(model)
    Fdim = prod(model.dims) * model.p
    companion_var = zeros(Fdim * Fdim, Fdim * Fdim)
    companion_var[1:Fdim, 1:Fdim] = cov_full
    return companion_var
end

function mar_eigvals(A::Vector{<:AbstractMatrix}, B::Vector{<:AbstractMatrix})
    p = length(A)
    @assert length(B) == p "A and B must have the same number of lags"

    # Build stacked phi
    phi = hcat([kron(B[i], A[i]) for i in 1:p]...)
    companion_phi = make_companion(phi)
    eig_phi = sort(abs.(eigvals(companion_phi)), rev=true)
    return eig_phi
end

function var_eigvals(C::Vector{<:AbstractMatrix})
    p = length(C)
    n = size(C[1], 1)
    if p == 1
        return eigvals(C[1])
    end
    companion_c = make_companion(C)
    return eigvals(companion_c)
end


"""
    isstable(var)

Check the stability of a VAR (Vector Autoregressive) model.

This function checks the stability of a VAR model by analyzing its companion matrix eigenvalues.
A VAR model is considered stable if all the eigenvalues of its companion matrix are within the unit circle.

## Arguments

- `var`: Lag matrix in the form required for a `VAR` model. See the documentation of `VAR`.

## Returns

- `Bool`: Returns `true` if the VAR model is stable, and `false` otherwise.

## Example

```julia
B = [1.0 2.0;
     3.0 4.0]
var_stable = isstable(B)  # Returns true or false based on the stability of the VAR model
```
## Note
The stability of a VAR model is determined by analyzing the eigenvalues of its companion matrix.
The companion matrix is constructed using the makecompanion function.

"""
function isstable(var::AbstractMatrix{T}; mineigen::Real=0.0, maxeigen::Real=0.90) where {T}
    C = make_companion(var)
    max_case = maximum(abs.(eigen(C).values)) < maxeigen
    min_case = minimum(abs.(eigen(C).values)) > mineigen
    return max_case && min_case
end

function isstable(A::Vector{<:AbstractMatrix}, B::Vector{<:AbstractMatrix}; mineigen::Real=0.0, maxeigen::Real=0.90)
    max_case = maximum(mar_eigvals(A, B)) < maxeigen
    min_case = minimum(mar_eigvals(A, B)) > mineigen

    return max_case && min_case
end

function isstable(C::Vector{<:AbstractMatrix}; mineigen::Real=0.0, maxeigen::Real=0.90)
    companion_c = make_companion(C)
    max_case = maximum(abs.(eigen(companion_c).values)) < maxeigen
    min_case = minimum(abs.(eigen(companion_c).values)) > mineigen
    return max_case && min_case
end

function is_fitted(model::MAR)
    return !(isnothing(model.residuals))
end

function is_fitted(model::VAR)
    return !(isnothing(model.C))
end

function require_fitted(model::AbstractARModel)
    is_fitted(model) && return true
    error("$(typeof(model)) must first be estimated.")
end

function normalize_slices(A::Vector{<:AbstractMatrix}, B::Vector{<:AbstractMatrix})
    p = length(A)
    @assert length(B) == p "A and B must have the same number of lags"

    A_normalized = Vector{Matrix{Float64}}(undef, p)
    B_normalized = Vector{Matrix{Float64}}(undef, p)

    for i in 1:p
        scale = norm(A[i])
        A_normalized[i] = A[i] / scale
        B_normalized[i] = B[i] * scale

        if A[i][1] < 0
            A_normalized[i] .= -A_normalized[i]
            B_normalized[i] .= -B_normalized[i]
        end
    end

    return A_normalized, B_normalized
end

function normalize_slices(A::AbstractMatrix, B::AbstractMatrix)
    n1, Np = size(A)
    n2, _ = size(B)
    p = div(Np, n1)  # number of blocks

    @inbounds for i in 0:(p-1)
        Ablock = view(A, :, i*n1+1:(i+1)*n1)
        Bblock = view(B, :, i*n2+1:(i+1)*n2)

        norm_f = norm(Ablock)

        Ablock ./= norm_f
        Bblock .*= norm_f

        if Ablock[1] < 0
            Ablock .= -Ablock
            Bblock .= -Bblock
        end
    end

    return A, B
end

function define_c(model::MAR)
    p = model.p
    C = Vector{Matrix{Float64}}(undef, p)
    for i in 1:p
        C[i] = kron(model.B[i], model.A[i])
    end
    return C
end

function commutation_matrix(A::AbstractMatrix)
    m, n = size(A)
    w = vec(reshape(reshape(1:m*n, m, n)', m*n))

    return I(m*n)[w, :]
end

function commutation_matrix(m::Int, n::Int)
    w = vec(reshape(reshape(1:m*n, m, n)', m*n))

    return I(m*n)[w, :]
end

function large_commutation_matrix(A::AbstractMatrix, n1::Integer, p::Integer)
    Kstar = commutation_matrix(A)
    iden = I(n1*n1*p)
    size_K = size(Kstar) .+ size(iden)
    large_comm = zeros(size_K)
    large_comm[1:size(iden, 1), 1:size(iden, 1)] = iden
    large_comm[(size(iden, 1)+1):end, (size(iden, 1)+1):end] = Kstar
    return large_comm

end

make_model(data, ::Type{VAR}; p) = VAR(data; p)
make_model(data, ::Type{MAR}; p) = MAR(data; p)

function fit_and_select!(model::AbstractARModel; ic_type::Symbol=:bic)
    # fit the provided (largest-p) model once
    fit!(model)
    fixed_data = model.data
    p_max = model.p
    ps = collect(0:p_max)
    ics = fill(NaN, length(ps))

    # record the ic for the provided model (p_max)
    ic_best = ic(model; ic_type=ic_type)
    ics[end] = ic_best
    model_best = model

    # evaluate smaller-lag models using the same effective sample
    for p in (p_max-1):-1:0
        start = p_max - p + 1
        data = isa(model, VAR) ? fixed_data[:, start:end] :
               isa(model, MAR) ? fixed_data[:, :, start:end] :
               error("Unsupported model type: $(typeof(model))")

        model_tmp = make_model(data, typeof(model); p=p)
        fit!(model_tmp)

        ic_tmp = ic(model_tmp; ic_type=ic_type)
        ics[p+1] = ic_tmp

        if ic_tmp < ic_best
            ic_best = ic_tmp
            model_best = model_tmp
        end
    end
    return model_best, hcat(ps, ics)
end

"""
    Q_matrix(N1, N2, p; sparse=true)

Build selector matrix Q of size (p*N1*N2) × (p^2*N1*N2) such that
Q * vec(X) = [ vec(Y_{t-1}); vec(Y_{t-2}); ...; vec(Y_{t-p}) ]
when X is the p×p block-diagonal matrix with diagonal blocks Y_{t-1},...,Y_{t-p}
(each Y is N1×N2).

Returns a SparseMatrixCSC if `sparse=true`, otherwise a dense Matrix.
"""
function Q_matrix(N1::Int, N2::Int, p::Int)
    n_out = p * N1 * N2
    n_in  = p * p * N1 * N2

    row_inds = Int[]
    col_inds = Int[]
    vals = Float64[]

    # loop over diagonal blocks k = 0:(p-1)
    for k in 0:(p-1)
        # output rows for this block (positions in stacked vec(Y)'s)
        out_block_start = k * (N1 * N2)
        out_positions = out_block_start .+ (1:(N1 * N2))

        # compute linear indices inside vec(X) that correspond to block (k+1,k+1)
        block_row_offset = k * N1           # rows start at (k*N1 + 1)
        block_col_offset = k * N2           # cols start at (k*N2 + 1)
        row_vec = block_row_offset .+ (1:N1)                  # N1
        col_vec = (block_col_offset .+ (0:(N2-1))) .* (p * N1) # N2 scaled by full-column stride p*N1

        # mat(i,j) = row_vec[i] + col_vec[j] gives the linear indices for that block
        mat = row_vec .+ col_vec'   # N1 x N2
        inds = vec(mat)             # length N1*N2, in correct column-major order

        append!(row_inds, out_positions)
        append!(col_inds, inds)
        append!(vals, ones(length(inds)))
    end

    return sparse(row_inds, col_inds, vals, n_out, n_in)
end

function vectorize_kronecker(A::AbstractMatrix, B::AbstractMatrix)
    m, n = size(A)
    p, q = size(B)
    K = commutation_matrix(q, m)
    P = kron(kron(I(n), K), I(p))
    return P
end

function number_parameters(model::MAR)
    if model.C == Matrix{Float64}[]
        return 0
    end
    number_A = sum(length, model.A)
    number_B = sum(length, model.B)

    return number_A + number_B
end

function number_parameters(model::VAR)
    if model.C == Matrix{Float64}[]
        return 0
    else
        return sum(length, model.C)   # 0 automatically when p = 0
    end
end

function calculate_residuals(model::MAR)
    data = model.data
    A = model.A
    B = model.B
    p = length(A)

    obs = size(data, 3)
    obs_eff = obs - p

    resp = data[:, :, (p+1):end]
    resp = resp .- mean(resp, dims=3)
    residuals = copy(resp)

    @inbounds for i in 1:p
        Ai, Bi = A[i], B[i]
        pred = data[:, :, (p+1-i):(end-i)]
        pred = pred .- mean(pred, dims=3)

        @inbounds for t in 1:obs_eff
            residuals[:, :, t] .-= Ai * pred[:, :, t] * Bi'
        end
    end

    return residuals
end

function _specification_test(data::AbstractArray)
    mar_model = MAR(data; method=:proj)
    fit!(mar_model)
    n1, n2 = mar_model.dims
    obs = mar_model.obs

    vecdata = vectorize(data)
    var_model = VAR(vecdata)
    fit!(var_model)

    unrestricted_coef = var_model.C[1]
    tensor_coef = reshape(unrestricted_coef, (n1, n2, n1, n2))
    phi = reshape(permutedims(tensor_coef, (1, 3, 2, 4)), n1 * n1, n2 * n2)

    a = vec(mar_model.A[1])
    b = vec(mar_model.B[1])
    phi_est = a * b'
    _, xi, V1 = variance_proj(mar_model)
    P = I - V1
    D = phi - phi_est
    M = P * xi * P
    M = Symmetric((M + M') / 2)
    test_statistic = vec(D)' * pinv(M) * vec(D)
    return test_statistic
end

function specification_test(data::AbstractArray)
    spec_results = _specification_test(data)
    n1, n2, _ = size(data)

    df = (n1^2 - 1) * (n2^2 - 1)
    p_value = 1 - cdf(Chisq(df), spec_results)
    return p_value
end

function calculate_residuals(model::VAR)
    return model.residuals
end

function show(io::IO, model::MAR)
    print(io, "MAR model (p=$(model.p), method=$(model.method))\n")
    print(io, "dims=$(model.dims), obs=$(model.obs)\n")
    print(io, "A=$(isempty(model.A) ? "unset" : size(model.A[1])) ")
    print(io, "B=$(isempty(model.B) ? "unset" : size(model.B[1])) ")
    print(io, "C=$(isempty(model.C) ? "unset" : size(model.C[1])) ")
    if isnothing(model.Sigma1) && isnothing(model.Sigma2)
        print(io, "Σ₁= unset, Σ₂= unset")
    else
        print(io, "Σ₁=$(size(model.Sigma1)), Σ₂=$(size(model.Sigma2))")
    end
end

function show(io::IO, model::VAR)
    print(io, "VAR model (p=$(model.p))\n")
    print(io, "obs=$(model.obs)\n")

    sz =
        if model.C === nothing
            "unset"
        elseif isempty(model.C)
            "[]"
        else
            string(size(model.C[1]))
        end

    print(io, "C=$sz")
end

