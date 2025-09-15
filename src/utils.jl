function make_companion()
    return nothing
end

function vectorize(data::AbstractArray{T}) where T
    n1, n2, obs = size(data)
    return reshape(data, n1*n2, obs)
end

function matricize(data::AbstractMatrix{T}, n1::Int, n2::Int) where T
    obs = size(data, 2)
    return reshape(data, n1, n2, obs)
end

