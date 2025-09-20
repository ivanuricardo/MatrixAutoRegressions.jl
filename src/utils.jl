
function vectorize(data::AbstractArray{T}) where T
    n1, n2, obs = size(data)
    return reshape(data, n1*n2, obs)
end

function matricize(data::AbstractMatrix{T}, n1::Int, n2::Int) where T
    obs = size(data, 2)
    return reshape(data, n1, n2, obs)
end

"""
    makecompanion(B::AbstractMatrix{T}) where {T}
    
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
function make_companion(B::AbstractMatrix{T}) where {T}
    n = Int(size(B, 1))
    p = Int(size(B, 2) / n)
    ident = diagm(fill(T(1), n * (p - 1)))
    companionlower = hcat(ident, zeros(n * (p - 1), n))
    companion = vcat(B, companionlower)
    return companion
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
function isstable(var::AbstractMatrix{T}; maxeigen::Real=0.9) where {T}
    C = makecompanion(var)
    return maximum(abs.(eigen(C).values)) < maxeigen
end

function isstable(A::Vector{<:AbstractMatrix}, B::Vector{<:AbstractMatrix}; maxeigen::Real=0.9)
    p = length(A)
    @assert length(B) == p "A and B must have the same number of lags"

    if p == 1
        maxeigA = maximum(abs.(eigen(A[1]).values))
        maxeigB = maximum(abs.(eigen(B[1]).values))
        return maxeigA * maxeigB < maxeigen
    else
        return maximum(mar_eigvals(A, B)) < maxeigen
    end
end

function ols(resp::AbstractArray, pred::AbstractArray)
    vec_resp = vectorize(resp)
    vec_pred = vectorize(pred)

    return vec_resp * vec_pred' / (vec_pred * vec_pred')
end

function estimate_var(Y::Matrix{Float64}, p::Int)
    N, obs = size(Y)
    obs_eff = obs - p

    pred = zeros(N*p, obs_eff)
    for t in 1:obs_eff
        pred[:, t] = vec(reverse(Y[:, t:(t+p-1)]))
    end

    resp = Y[:, p+1:end]
    A_hat = resp * pred' / (pred * pred')

    A = reshape(A_hat, N, N, p)
    return A
end

function is_fitted(model::MAR)
    return !(isnothing(model.A) || isnothing(model.B))
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
    end

    return A_normalized, B_normalized
end

function Base.show(io::IO, model::MAR)
    print(io, "MAR model (p=$(model.p), method=$(model.method))\n")
    print(io, "dims=$(model.dims), obs=$(model.obs)\n")
    print(io, "A=$(model.A === nothing ? "unset" : size(model.A)) ")
    print(io, "B=$(model.B === nothing ? "unset" : size(model.B)) ")
    if model.Sigma1 !== nothing && model.Sigma2 !== nothing
        print(io, "Σ₁=$(size(model.Sigma1)), Σ₂=$(size(model.Sigma2))")
    end
end

