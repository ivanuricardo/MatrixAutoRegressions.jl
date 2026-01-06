
function vech(x::AbstractMatrix)
    @assert size(x,1) == size(x,2) "Matrix must be square"
    k = size(x,1)
    out = Vector{eltype(x)}(undef, k*(k+1)÷2)
    m = 1
    @inbounds for j in 1:k
        for i in j:k
            out[m] = x[i,j]
            m += 1
        end
    end
    return out
end

function revech_lower(v::AbstractVector)
    L = length(v)
    n = Int(floor((-1 + sqrt(1 + 8L)) / 2))
    n * (n + 1) ÷ 2 == L || throw(ArgumentError("length(v) must be n(n+1)/2"))

    T = eltype(v)
    M = zeros(T, n, n)

    # linear indices of the lower triangle in column-major order
    lower_inds = [i + (j-1)*n for j in 1:n for i in j:n]

    M[lower_inds] = v
    return M
end

function revech(v::AbstractVector)
    M = revech_lower(v)
    M .= LowerTriangular(M) .+ LowerTriangular(M)'.- Diagonal(diag(M))
    return M
end

function vech_selection_mat(n::Int)
    p = n*(n+1) ÷ 2

    rows = Vector{Int}(undef, p)
    cols = Vector{Int}(undef, p)

    k = 1
    for j in 1:n
        base = (j-1)*n
        for i in j:n
            rows[k] = k
            cols[k] = base + i
            k += 1
        end
    end

    return sparse(rows, cols, ones(p), p, n*n)
end

function avar_sigma(model::VAR)
    require_fitted(model)
    sigma = model.Sigma
    n = model.n

    K = commutation_matrix(sigma)
    S = vech_selection_mat(n)

    omega = (I + K) * kron(sigma, sigma)

    return S * omega * S'
end

function avar_cholesky(model::VAR)
    require_fitted(model)
    n = model.n
    sigma = model.Sigma

    omega = avar_sigma(model)
    C = cholesky(Symmetric(sigma)).L
    K = commutation_matrix(sigma)

    A = kron(C, I(n)) + kron(I(n), C) * K
    S = vech_selection_mat(n)

    G = I / (S * A * S')
    return G * omega * G'

end
