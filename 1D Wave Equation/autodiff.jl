using Statistics

function tanh_derivs(th::Vector{Float64})
    d1 = 1 .- th .^ 2
    d2 = -2 .* th .* d1
    d3 = d1 .* (6 .* th .^ 2 .- 2)
    return (d1, d2, d3)
end

struct JetCache
    h   :: Vector{Vector{Float64}}
    hd  :: Vector{Vector{Float64}}
    hdd :: Vector{Vector{Float64}}
    zd  :: Vector{Vector{Float64}}
    zdd :: Vector{Vector{Float64}}
end

function forward_jet(params, x, t, seed_var::Symbol)
    L = length(params)

    h   = Vector{Vector{Float64}}(undef, L + 1)
    hd  = Vector{Vector{Float64}}(undef, L + 1)
    hdd = Vector{Vector{Float64}}(undef, L + 1)
    zd  = Vector{Vector{Float64}}(undef, L)
    zdd = Vector{Vector{Float64}}(undef, L)

    h[1]   = [Float64(x), Float64(t)]
    hd[1]  = seed_var == :x ? [1.0, 0.0] : [0.0, 1.0]
    hdd[1] = [0.0, 0.0]

    for k in 1:L
        W, b = params[k].W, params[k].b
        z      = W * h[k] + b 
        zd[k]  = W * hd[k]       
        zdd[k] = W * hdd[k]         

        if k < L                      
            th = tanh.(z)
            d1, d2, _ = tanh_derivs(th)
            h[k+1]   = th
            hd[k+1]  = d1 .* zd[k]
            hdd[k+1] = d1 .* zdd[k] .+ d2 .* zd[k] .^ 2
        else                          
            h[k+1], hd[k+1], hdd[k+1] = z, zd[k], zdd[k]
        end
    end

    return h[L+1][1], hd[L+1][1], hdd[L+1][1], JetCache(h, hd, hdd, zd, zdd)
end

function backward!(acc::Vector{Layer}, params, cache::JetCache, g0, g1, g2)
    L = length(params)

    H0, H1, H2 = [Float64(g0)], [Float64(g1)], [Float64(g2)]

    for k in L:-1:1
        W = params[k].W

        if k == L
            zbar, zdbar, zddbar = H0, H1, H2
        else
            d1, d2, d3 = tanh_derivs(cache.h[k+1])      # h at level k
            zbar   = H0 .* d1 .+ H1 .* d2 .* cache.zd[k] .+
                     H2 .* (d2 .* cache.zdd[k] .+ d3 .* cache.zd[k] .^ 2)
            zdbar  = H1 .* d1 .+ H2 .* (2 .* d2 .* cache.zd[k])
            zddbar = H2 .* d1
        end

        acc[k].W .+= zbar * cache.h[k]' .+ zdbar * cache.hd[k]' .+ zddbar * cache.hdd[k]'
        acc[k].b .+= zbar

        if k > 1
            H0, H1, H2 = W' * zbar, W' * zdbar, W' * zddbar
        end
    end
    return nothing
end

function compute_gradients(params, interior_pts, ic_pts, bc_pts,
                           c, lambdas, causal_eps, n_bins, use_causal)
    acc = zero_like(params)

    N_f  = size(interior_pts, 1)
    N_ic = size(ic_pts, 1)
    N_bc = size(bc_pts, 1)

    u_xx    = zeros(N_f)
    u_tt    = zeros(N_f)
    cache_x = Vector{JetCache}(undef, N_f)
    cache_t = Vector{JetCache}(undef, N_f)

    for i in 1:N_f
        x_i, t_i = interior_pts[i, 1], interior_pts[i, 2]
        _, _, u_xx[i], cache_x[i] = forward_jet(params, x_i, t_i, :x)
        _, _, u_tt[i], cache_t[i] = forward_jet(params, x_i, t_i, :t)
    end

    r = u_tt .- c^2 .* u_xx

    t_col = interior_pts[:, 2]
    w = use_causal ?
        causal_weights_cumulative(r, bin_indices(t_col, n_bins), n_bins, causal_eps) :
        ones(N_f)

    L_pde = mean(w .* r .^ 2)

    for i in 1:N_f
        rbar = 2 * w[i] * r[i] / N_f
        backward!(acc, params, cache_t[i], 0.0, 0.0,  rbar)
        backward!(acc, params, cache_x[i], 0.0, 0.0, -c^2 * rbar)
    end

    L_ic0 = 0.0
    L_ic1 = 0.0
    for i in 1:N_ic
        x_i = ic_pts[i, 1]
        u, u_t, _, cache = forward_jet(params, x_i, 0.0, :t)
        g0 = 2 * lambdas.ic0 * (u - sin(π * x_i)) / N_ic
        g1 = 2 * lambdas.ic1 * u_t / N_ic
        backward!(acc, params, cache, g0, g1, 0.0)
        L_ic0 += (u - sin(π * x_i))^2 / N_ic
        L_ic1 += u_t^2 / N_ic
    end

    L_bc = 0.0
    for i in 1:N_bc
        x_i, t_i = bc_pts[i, 1], bc_pts[i, 2]
        u, _, _, cache = forward_jet(params, x_i, t_i, :t)
        backward!(acc, params, cache, 2 * lambdas.bc * u / N_bc, 0.0, 0.0)
        L_bc += u^2 / N_bc
    end

    total = L_pde + lambdas.ic0 * L_ic0 + lambdas.ic1 * L_ic1 + lambdas.bc * L_bc
    return acc, (pde = L_pde, ic0 = L_ic0, ic1 = L_ic1, bc = L_bc, total = total)
end