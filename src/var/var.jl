
# We expect a vectorized dataset
function VAR(data::AbstractMatrix; p::Int=1, C::Union{Nothing, AbstractMatrix}=nothing)
    method = :mle
    n, obs = size(data)
    eff_obs = obs - p
    return VAR(C, n, p, nothing, eff_obs, method, data, nothing)
end

function fit!(model::VAR)
    ols_est, Sigma, U = estimate_var(model)
    model.C = ols_est
    model.Sigma = Sigma
    model.residuals = U

    return model
end

