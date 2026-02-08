
# We expect a vectorized dataset
function VAR(data::AbstractMatrix; p::Int=1, C::Union{Nothing, AbstractMatrix}=nothing, lambda::Float64=0.0)
    method = :mle
    n, obs = size(data)
    eff_obs = obs - p
    return VAR(C, n, p, nothing, eff_obs, method, data, nothing, lambda)
end

function fit!(model::VAR)
    ols_est, Sigma, U = estimate_var(model)
    model.C = ols_est
    model.Sigma = Sigma
    model.residuals = U

    return model
end

function select_lambda(model::VAR, lambda_range::AbstractRange{<:Real}; ic_type::Symbol=:bic)
    nlambda = length(lambda_range)
    ic_matrix = fill(NaN, nlambda, 2)
    data = model.data
    p = model.p
    
    best_ic = Inf
    optimal_lambda = first(lambda_range)
    
    for (idx, lambda) in enumerate(lambda_range)
        lambda_model = VAR(data; p, lambda = lambda)
        fit!(lambda_model)
        ic_val = ic(lambda_model; ic_type)

        ic_matrix[idx, 1] = lambda
        ic_matrix[idx, 2] = ic_val

        if ic_val < best_ic
            best_ic = ic_val
            optimal_lambda = lambda
        end
    end
    
    return (;optimal_lambda, ic_matrix)
end
