
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

#=function structure_lagged_data(data::AbstractArray; p::Int=1)=#
#=    n1, n2, obs = size(data)=#
#=    structured_data = zeros(n1*p, n2*p, obs-1)=#
#=    structured_data[1:n1, 1:n2, 1:obs-1] .= data[:, :, 1:end-1]=#
#=    for i in 1:(p-1)=#
#=        first_idx = (n1*i+1):n1*(i+1)=#
#=        second_idx = (n2*i+1):n2*(i+1)=#
#=        prepared_data = data[:, :, 1:end-1]=#
#=        structured_data[first_idx, second_idx, 1:obs-1] .= prepared_data=#
#=    end=#
#=    return structured_data=#
#==#
#=end=#

function update_A(resp::AbstractArray{T},
                  pred::AbstractArray{T},
                  B::AbstractVecOrMat{T};
                  Sigma2=I) where T
    n1 = size(resp, 1)
    p = Int(size(pred, 1) / n1)
    A_num = zeros(T, n1, n1 * p)
    A_den = zeros(T, n1 * p, n1 * p)
    obs = size(resp, 3)

    for t in 1:obs
        Rt = @view resp[:, :, t]
        Pt = @view pred[:, :, t]
        A_num += (Rt / Sigma2) * B * Pt'
        A_den += (Pt * B' / Sigma2) * B * Pt'
    end

    return A_num / A_den
end

function update_B(resp::AbstractArray{T},
                  pred::AbstractArray{T},
                  A::AbstractVecOrMat{T};
                  Sigma1=I) where T
    n2 = size(resp, 2)
    p = Int(size(pred, 2) / n2)
    B_num = zeros(T, n2, n2 * p)
    B_den = zeros(T, n2 * p, n2 * p)
    obs = size(resp, 3)

    for t in 1:obs
        Rt = @view resp[:, :, t]
        Pt = @view pred[:, :, t]
        B_num += (Rt' / Sigma1) * A * Pt
        B_den += (Pt' * A' / Sigma1) * A * Pt
    end

    return B_num / B_den
end

"""
    als(A_init, B_init, resp, pred; maxiter=100, tol=1e-6)

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
    maxiter::Int=100,
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

