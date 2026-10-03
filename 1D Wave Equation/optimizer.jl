struct AdamState
    m::Vector{Layer}
    v::Vector{Layer}
    t::Int
end

init_adam(params::Vector{Layer}) = AdamState(zero_like(params), zero_like(params), 0)

function adam_update(p, g, m, v, t; lr, beta1, beta2, eps)
    m_new = beta1 .* m .+ (1 - beta1) .* g
    v_new = beta2 .* v .+ (1 - beta2) .* g .^ 2

    m_hat = m_new ./ (1 - beta1^t)
    v_hat = v_new ./ (1 - beta2^t)

    p_new = p .- lr .* m_hat ./ (sqrt.(v_hat) .+ eps)
    return p_new, m_new, v_new
end

function adam_step(params::Vector{Layer}, grads::Vector{Layer}, state::AdamState;
                   lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8)

    t = state.t + 1
    n = length(params)

    new_params = Vector{Layer}(undef, n)
    new_m      = Vector{Layer}(undef, n)
    new_v      = Vector{Layer}(undef, n)

    for k in 1:n
        W, mW, vW = adam_update(params[k].W, grads[k].W, state.m[k].W, state.v[k].W, t;
                                lr=lr, beta1=beta1, beta2=beta2, eps=eps)
        b, mb, vb = adam_update(params[k].b, grads[k].b, state.m[k].b, state.v[k].b, t;
                                lr=lr, beta1=beta1, beta2=beta2, eps=eps)

        new_params[k] = Layer(W, b)
        new_m[k]      = Layer(mW, mb)
        new_v[k]      = Layer(vW, vb)
    end

    return new_params, AdamState(new_m, new_v, t)
end