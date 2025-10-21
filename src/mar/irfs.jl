function irf(model::MAR; S=nothing, h::Int=1)
    require_fitted(model)

    A = model.A
    B = model.B
    p = model.p
    k = prod(model.dims)
    kp = prod(k) * p

    phi = make_companion(hcat([kron(B[i], A[i]) for i in 1:p]...))
    if S == nothing
        S = I(k)
    end

    # selection matrix J to pick the first k elements of companion state
    J = zeros(eltype(phi), k, kp)
    J[:, 1:k] .= I(k)

    Sbig = zeros(eltype(phi), kp, k)
    Sbig[1:prod(k), :] .= S
    irf = Array{eltype(phi)}(undef, k, k, h+1)
    phi_power = Matrix(I, kp, kp)   # phi^0 = I

    for t in 0:h
        irf[:, :, t+1] .= J * phi_power * Sbig
        phi_power = phi_power * phi    # advance power: phi^(t+1)
    end

    return irf

end

