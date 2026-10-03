using Statistics
include("network.jl")

function pde_residuals(params, interior_pts, c::Float64)
    return [
        forward_hd(params, x, t, :t).eps12 - c^2 * forward_hd(params, x, t, :x).eps12
        for (x, t) in eachrow(interior_pts)]
end

function ic0_loss(params, ic_pts)
    errors = [forward(params, x, 0.0) - sin(pi * x) for (x, t) in eachrow(ic_pts)]
    return mean(errors.^2)
end

function ic1_loss(params, ic_pts)
    errors = [forward_dt(params, x, 0.0)[2] for (x, t) in eachrow(ic_pts)]
    return mean(errors.^2)
end

function bc_loss(params, bc_pts)
    errors = [forward(params, x, t) for (x, t) in eachrow(bc_pts)]
    return mean(errors.^2)
end

function causal_weights_naive(t_values::Vector{Float64}, epsilon::Float64)
    return exp.(-epsilon .* t_values)
end

function bin_indices(t_values::Vector{Float64}, n_bins::Int)
    return [max(1, ceil(Int, t * n_bins)) for t in t_values]
end

function causal_weights_cumulative(residuals::Vector{Float64}, bin_idx::Vector{Int}, n_bins::Int, epsilon::Float64)
    bin_losses = zeros(n_bins)

    for b in 1:n_bins
        pts_in_bin = residuals[bin_idx .== b]        
        bin_losses[b] = isempty(pts_in_bin) ? 0.0 : mean(pts_in_bin.^2)
    end

    cumulative = zeros(n_bins)
    running_sum = 0.0

    for b in 1:n_bins
        cumulative[b] = running_sum
        running_sum += bin_losses[b]
    end

    bin_weights = exp.(-epsilon .* cumulative)
    per_point_weights = [bin_weights[bin_idx[i]] for i in 1:length(residuals)]
    
    return per_point_weights
end

function total_loss(params, interior_pts, ic0_pts, ic1_pts, bc_pts, c, lambdas, causal_epsilon, n_bins, use_causal::Bool)
    raw_r = pde_residuals(params, interior_pts, c)

    if use_causal
        bidx = bin_indices(interior_pts[:, 2], n_bins)
        weights = causal_weights_cumulative(raw_r, bidx, n_bins, causal_epsilon)
    else
        weights = ones(length(raw_r))
    end

    L_pde = mean(weights .* raw_r.^2)
    L_ic0 = ic0_loss(params, ic0_pts)
    L_ic1 = ic1_loss(params, ic1_pts)
    L_bc = bc_loss(params, bc_pts)

    return L_pde + lambdas.ic0 * L_ic0 + lambdas.ic1 * L_ic1 + lambdas.bc * L_bc
end