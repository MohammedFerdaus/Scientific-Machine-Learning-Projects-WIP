using Random
using Statistics
using Printf
using Serialization

const SRC = @__DIR__
include(joinpath(SRC, "hyperdual.jl"))
include(joinpath(SRC, "network.jl"))
include(joinpath(SRC, "sampling.jl"))
include(joinpath(SRC, "losses.jl"))
include(joinpath(SRC, "autodiff.jl"))
include(joinpath(SRC, "optimizer.jl"))
include(joinpath(SRC, "evaluate.jl"))

geometric(a, b, e, n) = a * (b / a)^((e - 1) / max(n - 1, 1))

function train(; name="run", n_epochs=5000,
                 N_f=2000, N_ic=100, N_bc=100, c=1.0,
                 layer_sizes=[2, 20, 20, 20, 1],
                 lr=1e-3, lr_final=1e-4,
                 lambdas=(ic0=1.0, ic1=1.0, bc=1.0),
                 use_causal=false, n_bins=10, eps_start=1e-2, eps_end=1e1,
                 log_every=100, checkpoint_every=500, seed=1,
                 outdir=joinpath(SRC, "results"))

    Random.seed!(seed)
    dir = joinpath(outdir, name)
    mkpath(dir)
    logio = open(joinpath(dir, "log.txt"), "w")
    function logline(s)
        println(s); println(logio, s); flush(logio)
    end

    params   = init_params(layer_sizes)
    state    = init_adam(params)
    interior = sample_interior(N_f)
    ic       = sample_ic(N_ic)
    bc       = sample_bc(N_bc)

    history  = NamedTuple[]
    bin_hist = Tuple{Int,Vector{Float64},Vector{Float64}}[]

    logline("run=$name  c=$c  causal=$use_causal  N_f=$N_f  net=$layer_sizes  epochs=$n_epochs")
    t_start = time()

    function checkpoint()
        serialize(joinpath(dir, "params.jls"), params)
        try
            plot_training(history, bin_hist, joinpath(dir, "training.png"))
            plot_solution(params, c, joinpath(dir, "solution.png"))
        catch e
            logline("plot failed (training continues): $e")
        end
    end

    for epoch in 1:n_epochs
        eps  = geometric(eps_start, eps_end, epoch, n_epochs)
        lr_e = geometric(lr, lr_final, epoch, n_epochs)

        grads, losses = compute_gradients(params, interior, ic, bc, c,
                                          lambdas, eps, n_bins, use_causal)

        if !isfinite(losses.total)
            logline("STOP: non-finite loss at epoch $epoch: $losses")
            break
        end

        params, state = adam_step(params, grads, state; lr=lr_e)

        if epoch == 1 || epoch % log_every == 0 || epoch == n_epochs
            l2  = compute_l2_error(params, c)
            err = per_bin_error(params, c, n_bins)
            res = per_bin_pde_residual(params, c, n_bins)
            push!(history, (epoch=epoch, total=losses.total, pde=losses.pde,
                            ic0=losses.ic0, ic1=losses.ic1, bc=losses.bc,
                            l2=l2, eps=eps, lr=lr_e))
            push!(bin_hist, (epoch, err, res))
            elapsed = time() - t_start
            eta = elapsed / epoch * (n_epochs - epoch)
            logline(@sprintf("ep %6d | total %.3e pde %.3e ic0 %.3e ic1 %.3e bc %.3e | L2 %.4f | eps %.2e | %.0fs (ETA %.0fs)",
                             epoch, losses.total, losses.pde, losses.ic0, losses.ic1,
                             losses.bc, l2, eps, elapsed, eta))
        end

        epoch % checkpoint_every == 0 && checkpoint()
    end

    checkpoint()
    logline("done. results in $dir")
    close(logio)
    return params, history
end

if abspath(PROGRAM_FILE) == @__FILE__
    train(name="baseline_c1", use_causal=false, c=1.0)
    train(name="causal_c1",   use_causal=true,  c=1.0)
end