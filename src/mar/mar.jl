
"""
Create an un-fitted `MAR` model object from a 3D data array.

# Arguments
- `data::AbstractArray` — Observed data with shape `(nrow, ncol, T)`.
- `p::Int=1` — Lag order.
- `method::Symbol=:mle` — Default estimation method used by `fit!`. Common choices: `:mle`, `:als`, or `:proj`.
- `A::Vector{<:AbstractMatrix}=Vector{Matrix{Float64}}()` — Optional initial guesses for left coefficient matrices.
- `B::Vector{<:AbstractMatrix}=Vector{Matrix{Float64}}()` — Optional initial guesses for right coefficient matrices.
- `C::Vector{<:AbstractMatrix}=Vector{Matrix{Float64}}()` — Optional initial guesses for intercept matrices.
- `maxiter::Int=100` — Maximum iterations for iterative estimation routines.
- `tol::Real=1e-6` — Convergence tolerance for iterative estimation.

# Returns
A `MAR` struct with the provided fields filled and placeholders (`nothing`) for estimated covariances, residuals, and `iters`. Use `fit!(m)` to estimate parameters.

# Examples
    ```julia
    data = randn(3, 2, 200)               # 3×2 observation matrix, T=200
    m = MAR(data; p=1, method=:ols)
    fit!(m)                               # populates A, B, C, Sigma, residuals, iters
    ```
"""
function MAR(data::AbstractArray;
    p::Int=1,
    method::Symbol=:mle,
    A::Vector{<:AbstractMatrix}=Vector{Matrix{Float64}}(),
    B::Vector{<:AbstractMatrix}=Vector{Matrix{Float64}}(),
    C::Vector{<:AbstractMatrix}=Vector{Matrix{Float64}}(),
    maxiter::Int=100,
    tol::Real=1e-6,
    )

    dims = size(data)[1:2]
    eff_obs = size(data, 3) - p
    iters = nothing

    return MAR(A, B, C, p, nothing, nothing, nothing, dims, eff_obs, method, data, maxiter, tol, iters, nothing)
end

function fit!(model::MAR)

    n1, n2 = model.dims

    if model.p == 0
        model.residuals = model.data .- mean(model.data, dims = 3)
        flat_residuals = vectorize(model.residuals)
        cov_est = (flat_residuals * flat_residuals') / model.obs
        proj_est = projection(cov_est, (n1, n2))

        sigma_ests = flipflop_covariance(model.residuals; model.maxiter, tol=model.tol,
                        sigma1=proj_est.A, sigma2=proj_est.B)
        model.Sigma1 = sigma_ests.sigma1
        model.Sigma2 = sigma_ests.sigma2
        model.Sigma = kron(sigma_ests.sigma2, sigma_ests.sigma1)
        model.iters = sigma_ests.iters
        return model
    end

    ols_est, cov_est = estimate_var(model.data; model.p)
    proj_est = projection(ols_est, model.dims)
    proj_cov = projection([cov_est], model.dims)

    if model.method == :proj
        model.A = proj_est.A
        model.B = proj_est.B
        model.C = define_c(model)
        model.residuals = calculate_residuals(model)

    elseif model.method == :als

        A0 = isempty(model.A) ? proj_est.A : copy(model.A)
        B0 = isempty(model.B) ? proj_est.B : copy(model.B)

        results = als(model.data, A0, B0; maxiter=model.maxiter, tol=model.tol)

        model.A = results.A
        model.B = results.B
        model.C = define_c(model)
        model.iters = results.num_iter
        model.residuals = residuals(model)
        vec_residuals = vectorize(model.residuals)
        num_params = number_parameters(model)
        num_eq = prod(model.dims)
        dof = model.obs - (num_params / num_eq)
        model.Sigma = (vec_residuals * vec_residuals') ./ dof

    elseif model.method == :mle

        A0 = isempty(model.A) ? proj_est.A : copy(model.A)
        B0 = isempty(model.B) ? proj_est.B : copy(model.B)
        Sigma1_init = proj_cov.A[1]
        Sigma2_init = proj_cov.B[1]

        results = mle(model.data, A0, B0, Sigma1_init, Sigma2_init;
            maxiter=model.maxiter, tol=model.tol)

        model.A = results.A
        model.B = results.B
        model.C = define_c(model)
        model.Sigma = kron(results.Sigma2, results.Sigma1)
        model.Sigma1 = results.Sigma1
        model.Sigma2 = results.Sigma2
        model.iters = results.num_iter
        model.residuals = residuals(model)

    else
        throw(ArgumentError("Unknown method: $(model.method)"))
    end

    return model
end

function ls_objective(resp::AbstractArray{T}, pred::AbstractArray{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where T
    obs = size(resp, 3)
    ssr = 0
    for i in 1:obs
        residual = resp[:, :, i] - A * pred[:, :, i] * B'
        ssr += sum(abs2, residual)
    end
    return ssr
end

function ls_objective(data::AbstractArray{T}, A::Vector{<:AbstractMatrix}, B::Vector{<:AbstractMatrix}; p=1) where T
    n1, n2, obs = size(data)

    ssr = 0
    for i in (p+1):obs
        pred = zeros(n1, n2)
        for j in 1:p
            pred .+= A[j] * data[:, :, i-j] * B[j]'
        end
        residual = data[:, :, i] - pred
        ssr += sum(abs2, residual)
    end
    
    return ssr
end

function ls_objective(model::MAR)
    require_fitted(model)
    return ls_objective(model.data, model.A, model.B; model.p)
end

function mle_objective(
    data::AbstractArray{T},
    A::Vector{<:AbstractMatrix},
    B::Vector{<:AbstractMatrix},
    Sigma1::AbstractMatrix,
    Sigma2::AbstractMatrix,
    ) where T

    n1, n2, obs = size(data)
    p = length(A)

    ssr = 0

    for i in (p+1):obs
        pred = zeros(n1, n2)
        for j in 1:p
            pred .+= A[j] * data[:, :, i-j] * B[j]'
        end

        residual = data[:, :, i] - pred
        ssr += tr((Sigma1 \ residual) / Sigma2 * residual')
    end
    eff_obs = obs - p
    first_cov = -n2 * eff_obs * logdet(Sigma1)
    second_cov = -n1 * eff_obs * logdet(Sigma2)

    return first_cov + second_cov - ssr
end

function mle_objective(model::MAR)
    require_fitted(model)
    if model.Sigma1 == nothing || model.Sigma2 == nothing
        model.Sigma1 = I(model.dims[1])
        model.Sigma2 = I(model.dims[2])
    end
    return mle_objective(model.data, model.A, model.B, model.Sigma1, model.Sigma2)
end

function coef(model::MAR)
    require_fitted(model)
    A = model.A
    B = model.B
    C = model.C
    return (;A, B, C)
end

# This is only forecasting using forecasted values
function predict(model::MAR; h::Int=1)
    require_fitted(model)

    n1, n2 = model.dims
    p = model.p
    A, B = coef(model)

    Yhist = model.data[:, :, end-p+1:end]

    forecasts = Array{Float64,3}(undef, n1, n2, h)

    for step in 1:h

        # first step is initialized
        Ypred = zeros(n1, n2)
        for j in 1:p
            Ypred .+= A[j] * Yhist[:, :, end-j+1] * B[j]'
        end
        forecasts[:, :, step] .= Ypred

        # update history with predicted value
        Yhist = cat(Yhist, reshape(Ypred, n1, n2, 1); dims=3)
    end

    return forecasts
end

function train_test_split(model::MAR; h::Int=1)
    test_data = model.data[:, :, end-h+1:end]
    train_data = model.data[:, :, 1:end-h]

    return train_data, test_data
end
