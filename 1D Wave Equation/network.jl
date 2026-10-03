include("hyperdual.jl")

struct Layer
    W::Matrix{Float64}   
    b::Vector{Float64}   
end

function init_params(layer_sizes::Vector{Int})
    layers = Layer[]
    for i in 1:length(layer_sizes)-1
        in_dim = layer_sizes[i]
        out_dim = layer_sizes[i + 1]

        lim = sqrt(6 / (in_dim + out_dim))
        W = (2 .* rand(out_dim, in_dim) .- 1) .* lim
        b = zeros(out_dim)

        push!(layers, Layer(W, b)) 
    end

    return layers
end

zero_like(params::Vector{Layer}) = [Layer(zero(l.W), zero(l.b)) for l in params]

function forward(params::Vector{Layer}, x::Float64, t::Float64)
    h = [x, t]
    for layer in params[1:end-1]
        h = tanh.(layer.W * h .+ layer.b)
    end
    h = params[end].W * h .+ params[end].b
    return h[1]
end

function forward_hd(params::Vector{Layer}, x::Float64, t::Float64, seed_var::Symbol)
    hx = (seed_var == :x) ? seed(x) : constant(x)
    ht = (seed_var == :t) ? seed(t) : constant(t)
    h = [hx, ht]

    for (k, layer) in enumerate(params)
        out_dim, in_dim = size(layer.W)
        h_new = Vector{typeof(hx)}(undef, out_dim)
        for i in 1:out_dim
            acc = add(scale(h[1], layer.W[i,1]), constant(layer.b[i]))
            for j in 2:in_dim
                acc = add(acc, scale(h[j], layer.W[i,j]))
            end
            h_new[i] = (k < length(params)) ? tanh_hd(acc) : acc
        end
        h = h_new
    end
    return h[1]
end

function forward_dt(params::Vector{Layer}, x::Float64, t::Float64)
    hd = forward_hd(params, x, t, :t)
    return (hd.val, hd.eps1)

end