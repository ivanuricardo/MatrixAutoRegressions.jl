
function structure_lagged_data(data::AbstractArray; p::Int=1)
    n1, n2, obs = size(data)

    structured_data = zeros(n1*p, n2*p, obs-p)
    structured_data[1:n1, 1:n2, 1:obs-p] .= data[:, :, p:end-1]

    for i in 1:(p-1)
        first_idx = (n1*i+1):n1*(i+1)
        second_idx = (n2*i+1):n2*(i+1)
        prepared_data = data[:, :, (p-i):(end-1-i)]
        structured_data[first_idx, second_idx, 1:obs-p] .= prepared_data
    end
    return structured_data

end

# function structure_lagged_data(data::AbstractArray; p::Int=1)
#    n1, n2, obs = size(data)
#    structured_data = zeros(n1*p, n2*p, obs-1)
#    structured_data[1:n1, 1:n2, 1:obs-1] .= data[:, :, 1:end-1]
#    for i in 1:(p-1)
#        first_idx = (n1*i+1):n1*(i+1)
#        second_idx = (n2*i+1):n2*(i+1)
#        prepared_data = data[:, :, 1:end-1]
#        structured_data[first_idx, second_idx, 1:obs-1] .= prepared_data
#    end
#    return structured_data
#
# end

function update_A(resp::AbstractArray{T},
                  pred::AbstractArray{T},
                  B::AbstractVecOrMat{T};
                  Sigma2=I) where T
    n1 = size(resp, 1)
    n2 = size(resp, 2)
    p = Int(size(pred, 1) / n1)
    n1p = n1 * p
    n2p = n2 * p

    A_num = zeros(T, n1, n1p)
    A_den = zeros(T, n1p, n1p)
    obs = size(resp, 3)

    S2invB = Sigma2 \ B  # I \ B just returns B
    BtS2invB = B' * S2invB

    tmp1 = zeros(T, n1, n2p)
    tmp2 = zeros(T, n1p, n2p)

    for t in 1:obs
        Rt = resp[:, :, t]
        Pt = pred[:, :, t]

        mul!(tmp1, Rt, S2invB)
        mul!(A_num, tmp1, Pt', one(T), one(T))

        mul!(tmp2, Pt, BtS2invB)
        mul!(A_den, tmp2, Pt', one(T), one(T))
    end

    return A_num / A_den
end

function update_B(resp::AbstractArray{T},
                  pred::AbstractArray{T},
                  A::AbstractVecOrMat{T};
                  Sigma1=I) where T
    n1 = size(resp, 1)
    n2 = size(resp, 2)
    p = Int(size(pred, 2) / n2)
    n1p = n1 * p
    n2p = n2 * p

    B_num = zeros(T, n2, n2p)
    B_den = zeros(T, n2p, n2p)
    obs = size(resp, 3)

    S1invA = Sigma1 \ A
    AtS1invA = A' * S1invA

    tmp1 = zeros(T, n2, n1p)
    tmp2 = zeros(T, n2p, n1p)

    for t in 1:obs
        Rt = resp[:, :, t]
        Pt = pred[:, :, t]

        mul!(tmp1, Rt', S1invA)
        mul!(B_num, tmp1, Pt, one(T), one(T))

        mul!(tmp2, Pt', AtS1invA)
        mul!(B_den, tmp2, Pt, one(T), one(T))
    end

    return B_num / B_den
end

"""
    als(A_init, B_init, resp, pred; maxiter=500, tol=1e-6)

Alternating Least Squares estimation for the MAR(1) model:

    y_t = (B ⊗ A) y_{t-1} + e_t

# Arguments
- `A_init`, `B_init`: initial guesses for A and B.
- `resp`: response matrix (N × T).
- `pred`: predictor matrix (N × T).

# Keyword arguments
- `maxiter`: maximum number of ALS iterations.
- `tol`: convergence tolerance (Frobenius norm change).

# Returns
- `(A, B)`: estimated matrices.
"""
function als(
    data::AbstractArray{T},
    A::Vector{<:AbstractMatrix},
    B::Vector{<:AbstractMatrix};
    maxiter::Int=500,
    tol::Real=1e-6,
    Sigma1=I,
    Sigma2=I,
    warn=true,
    ) where T

    n1, n2 = size(A[1], 1), size(B[1], 1)
    p = length(A)
    resp = data[:, :, (p+1):end]
    pred = structure_lagged_data(data; p)

    resp = resp .- mean(resp, dims=3)
    pred = pred .- mean(pred, dims=3)
    obj = ls_objective(data, A, B; p)

    Astack = hcat(A...)
    Bstack = hcat(B...)

    track_obj = fill(NaN, maxiter)

    num_iter = 0
    for i in 1:maxiter
        num_iter += 1
        obj_old = copy(obj)

        Astack = update_A(resp, pred, Bstack; Sigma2=Sigma2)
        Bstack = update_B(resp, pred, Astack; Sigma1=Sigma1)

        Astack, Bstack = normalize_slices(Astack, Bstack)

        obj = ls_objective(resp, pred, Astack, Bstack)
        track_obj[i] = abs(obj - obj_old)

        if track_obj[i] < tol
            A = [@view Astack[:, (k-1)*n1+1 : k*n1] for k in 1:p]
            B = [@view Bstack[:, (k-1)*n2+1 : k*n2] for k in 1:p]
            track_obj = track_obj[.!isnan.(track_obj)]
            return (; A, B, track_obj, obj, num_iter)
        end

        if i == maxiter
            if warn
                @warn "Reached maximum number of iterations"
            end
            A = [@view Astack[:, (k-1)*n1+1 : k*n1] for k in 1:p]
            B = [@view Bstack[:, (k-1)*n2+1 : k*n2] for k in 1:p]
            return (; A, B, track_obj, obj, num_iter)
        end

    end

end

