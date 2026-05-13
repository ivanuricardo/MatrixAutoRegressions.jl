
function _logdet_sigma(model::AbstractARModel)
    S = Symmetric(model.Sigma)
    isposdef(S) || return nothing
    ch = cholesky(S)
    return 2 * sum(log, diag(ch.L))
end

function residuals(model::AbstractARModel)
    return calculate_residuals(model)
end

function loglikelihood(model::VAR)
    require_fitted(model)
    E = residuals(model)
    n, obs = size(E)

    const_term = - (obs * n / 2) * log(2π)

    Sigma = Symmetric(model.Sigma)
    ch = cholesky(Sigma)
    # omit dividing by 2 because its a cholesky factor
    logdetsigma = obs * sum(log.(abs.(diag(ch.L))))
    sol = ch \ E
    quad = sum(sol .* E)
    return const_term - logdetsigma - 0.5 * quad
end

function loglikelihood(model::MAR)
    require_fitted(model)
    if model.method != :mle
        error("Method must be Maximum Likelihood!")
    end
    E = residuals(model)

    n1, n2, obs = size(E)

    const_term = - (obs * n1 * n2) / 2 * log(2π)

    sigma1 = Symmetric(model.Sigma1)
    sigma2 = Symmetric(model.Sigma2)

    ch1 = cholesky(sigma1)
    ch2 = cholesky(sigma2)

    logdet1 = 2 * sum(log, diag(ch1.L))
    logdet2 = 2 * sum(log, diag(ch2.L))

    term_logdet = (obs * n2) / 2 * logdet1 + (obs * n1) / 2 * logdet2

    sum_tr = 0.0
    for t in 1:obs
        Et = @view E[:, :, t]
        U = ch1.L \ Et
        W = (ch2.L \ U')'
        sum_tr += sum(abs2, W)
    end
    term_quad = 0.5 * sum_tr
    return const_term - term_logdet - term_quad

end

function aic(model::AbstractARModel)
    obs = model.obs
    k = number_parameters(model)
    if model.method == :mle
        return 2 * k - 2 * loglikelihood(model)
    end
    logdetterm = _logdet_sigma(model)
    logdetterm === nothing && return Inf
    return 2 * k + obs * logdetterm
end

function bic(model::AbstractARModel)
    obs = model.obs
    k = number_parameters(model)
    if model.method == :mle
        return k * log(obs) - 2 * loglikelihood(model)
    end
    logdetterm = _logdet_sigma(model)
    logdetterm === nothing && return Inf
    return k * log(obs) + obs * logdetterm
end

function hqc(model::AbstractARModel)
    k = number_parameters(model)
    obs = model.obs
    if model.method == :mle
        return k * 2 * log(log(obs)) - 2 * loglikelihood(model)
    end
    logdetterm = _logdet_sigma(model)
    logdetterm === nothing && return Inf
    return 2 * k * log(log(obs)) + obs * logdetterm
end

function ic(model::AbstractARModel; ic_type::Symbol=:bic)
    if ic_type == :bic
        return bic(model)
    elseif ic_type == :aic
        return aic(model)
    elseif ic_type == :hqc
        return hqc(model)
    else
        error("No valid IC type given!")
    end
end


