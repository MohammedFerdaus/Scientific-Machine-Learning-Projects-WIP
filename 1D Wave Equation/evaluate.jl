using CairoMakie

analytical_solution(x, t, c) = sin(π * x) * cos(π * c * t)

function make_grid(n)
    return collect(range(0.0, 1.0; length=n)), collect(range(0.0, 1.0; length=n))
end

function predict_grid(params, xs, ts)
    U = zeros(length(xs), length(ts))
    for j in eachindex(ts), i in eachindex(xs)
        U[i, j] = forward(params, xs[i], ts[j])
    end
    return U
end

exact_grid(xs, ts, c) = [analytical_solution(x, t, c) for x in xs, t in ts]

function compute_l2_error(params, c; n=50)
    xs, ts = make_grid(n)
    U, Ue = predict_grid(params, xs, ts), exact_grid(xs, ts, c)
    return sqrt(sum((U .- Ue) .^ 2)) / sqrt(sum(Ue .^ 2))
end

time_bin(t, n_bins) = clamp(ceil(Int, t * n_bins), 1, n_bins)

function per_bin_error(params, c, n_bins; n=60)
    xs, ts = make_grid(n)
    U, Ue = predict_grid(params, xs, ts), exact_grid(xs, ts, c)
    sums, counts = zeros(n_bins), zeros(n_bins)
    for j in eachindex(ts)
        b = time_bin(ts[j], n_bins)
        sums[b]   += sum((U[:, j] .- Ue[:, j]) .^ 2)
        counts[b] += length(xs)
    end
    return sums ./ max.(counts, 1)
end

function per_bin_pde_residual(params, c, n_bins; n=30)
    xs, ts = make_grid(n)
    sums, counts = zeros(n_bins), zeros(n_bins)
    for t in ts, x in xs
        _, _, uxx, _ = forward_jet(params, x, t, :x)
        _, _, utt, _ = forward_jet(params, x, t, :t)
        b = time_bin(t, n_bins)
        sums[b]   += (utt - c^2 * uxx)^2
        counts[b] += 1
    end
    return sums ./ max.(counts, 1)
end

function plot_solution(params, c, path; n=100)
    xs, ts = make_grid(n)
    U, Ue = predict_grid(params, xs, ts), exact_grid(xs, ts, c)
    E = abs.(U .- Ue)

    fig = Figure(size=(1200, 850))
    panels = [("Predicted u", U, :RdBu, (-1, 1)),
              ("Exact u", Ue, :RdBu, (-1, 1)),
              ("|Error|", E, :viridis, (0, max(maximum(E), 1e-12)))]
    for (col, (title, M, cmap, crange)) in enumerate(panels)
        ax = Axis(fig[1, col]; title=title, xlabel="x", ylabel="t")
        hm = heatmap!(ax, xs, ts, M; colormap=cmap, colorrange=crange)
        Colorbar(fig[2, col], hm; vertical=false)
    end

    ax = Axis(fig[3, 1:3]; title="Slices (solid = predicted, dashed = exact)",
              xlabel="x", ylabel="u")
    colors = Makie.wong_colors()
    for (i, tk) in enumerate((0.0, 0.25, 0.5, 0.75, 1.0))
        j = argmin(abs.(ts .- tk))
        lines!(ax, xs, U[:, j];  color=colors[i], label="t=$tk")
        lines!(ax, xs, Ue[:, j]; color=colors[i], linestyle=:dash)
    end
    axislegend(ax; position=:rb)
    save(path, fig)
end

col(history, s) = [getproperty(h, s) for h in history]

function plot_training(history, bin_hist, path)
    length(history) < 2 && return
    epochs = col(history, :epoch)
    n_bins = length(bin_hist[1][2])

    fig = Figure(size=(1300, 900))

    ax1 = Axis(fig[1, 1]; title="Loss terms", xlabel="epoch", yscale=log10)
    for s in (:total, :pde, :ic0, :ic1, :bc)
        lines!(ax1, epochs, max.(col(history, s), 1e-16); label=String(s))
    end
    axislegend(ax1)

    ax2 = Axis(fig[1, 2]; title="Relative L2 error vs exact", xlabel="epoch", yscale=log10)
    lines!(ax2, epochs, max.(col(history, :l2), 1e-16))

    be = [bin_hist[i][2][b] for i in eachindex(bin_hist), b in 1:n_bins]
    br = [bin_hist[i][3][b] for i in eachindex(bin_hist), b in 1:n_bins]
    be_epochs = [bh[1] for bh in bin_hist]

    ax3 = Axis(fig[2, 1][1, 1]; title="log10 error per time bin", xlabel="epoch", ylabel="time bin (1 = early)")
    hm3 = heatmap!(ax3, be_epochs, 1:n_bins, log10.(max.(be, 1e-16)); colormap=:viridis)
    Colorbar(fig[2, 1][1, 2], hm3)

    ax4 = Axis(fig[2, 2][1, 1]; title="log10 PDE residual per time bin", xlabel="epoch", ylabel="time bin (1 = early)")
    hm4 = heatmap!(ax4, be_epochs, 1:n_bins, log10.(max.(br, 1e-16)); colormap=:viridis)
    Colorbar(fig[2, 2][1, 2], hm4)

    save(path, fig)
end