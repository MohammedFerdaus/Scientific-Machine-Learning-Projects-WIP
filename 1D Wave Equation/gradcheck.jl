using Random
using Statistics

function perturb!(params, k, field::Symbol, idx, delta)
    arr = (field == :W) ? params[k].W : params[k].b
    arr[idx] += delta
    
    return nothing
end

function central_diff(loss_fn, params, k, field, idx; h=1e-5)
    perturb!(params, k, field, idx, +h);   Lp = loss_fn(params)
    perturb!(params, k, field, idx, -2h);  Lm = loss_fn(params)
    perturb!(params, k, field, idx, +h)
    
    return (Lp - Lm) / (2h)
end

mixed_err(a, b) = abs(a - b) / max(abs(a), abs(b), 1e-4)

function sample_param_indices(params, n)
    L = length(params)
    pick(k, field) = (k, field, rand(eachindex(field == :W ? params[k].W : params[k].b)))

    samples = Tuple{Int,Symbol,Int}[pick(1, :W), pick(L, :W), pick(rand(1:max(L - 1, 1)), :b)]
    while length(samples) < n
        push!(samples, pick(rand(1:L), rand((:W, :b))))
    end
    
    return samples
end

grad_entry(grads, k, field, idx) = (field == :W) ? grads[k].W[idx] : grads[k].b[idx]

function make_test_params(sizes)
    params = init_params(sizes)
    for l in params
        l.b .= 0.1 .* randn(length(l.b))
    end
    
    return params
end

function check(name, got, expected; tol=1e-12)
    ok = abs(got - expected) < tol
    println(ok ? "PASS  " : "FAIL  ", name, "   got=", got, "  expected=", expected)
    
    return ok
end

function test_hyperdual()
    ok = true

    x = seed(2.0)
    f = mul(mul(x, x), x)
    ok &= check("x^3 val",   f.val,   8.0)
    ok &= check("x^3 eps1",  f.eps1,  12.0)
    ok &= check("x^3 eps2",  f.eps2,  12.0)
    ok &= check("x^3 eps12", f.eps12, 12.0)

    th = tanh(0.5); d1 = 1 - th^2; d2 = -2 * th * d1
    g = tanh_hd(seed(0.5))
    ok &= check("tanh val",   g.val,   th)
    ok &= check("tanh eps1",  g.eps1,  d1)
    ok &= check("tanh eps12", g.eps12, d2)

    th = tanh(1.6); d1 = 1 - th^2; d2 = -2 * th * d1
    g = tanh_hd(add(scale(seed(0.3), 2.0), constant(1.0)))
    ok &= check("tanh(2x+1) val",   g.val,   th)
    ok &= check("tanh(2x+1) eps1",  g.eps1,  2 * d1)
    ok &= check("tanh(2x+1) eps12", g.eps12, 4 * d2)

    x0, t0 = 1.3, 0.7
    f = mul(mul(seed(x0), seed(x0)), constant(t0))
    ok &= check("x^2 t, seed x: eps12 (= 2t)", f.eps12, 2 * t0)
    f = mul(mul(constant(x0), constant(x0)), seed(t0))
    ok &= check("x^2 t, seed t: eps12 (= 0)",  f.eps12, 0.0)

    a = HyperDual(0.3, 0.5, -0.2, 0.9)
    s, m = scale(a, 1.7), mul(constant(1.7), a)
    ok &= check("scale vs mul val",   s.val,   m.val)
    ok &= check("scale vs mul eps1",  s.eps1,  m.eps1)
    ok &= check("scale vs mul eps2",  s.eps2,  m.eps2)
    ok &= check("scale vs mul eps12", s.eps12, m.eps12)

    println(ok ? ">> Group 1 PASS" : ">> Group 1 FAIL")
    return ok
end

function test_forward_agreement()
    Random.seed!(42)
    params = make_test_params([2, 10, 10, 1])
    maxdiff = 0.0
    for _ in 1:20
        x, t = rand(), rand()
        u_plain = forward(params, x, t)
        for s in (:x, :t)
            u, us, uss, _ = forward_jet(params, x, t, s)
            hd = forward_hd(params, x, t, s)
            maxdiff = max(maxdiff, abs(u - hd.val), abs(us - hd.eps1),
                          abs(uss - hd.eps12), abs(u - u_plain))
        end
    end

    ok = maxdiff < 1e-12
    println(ok ? ">> Group 2 PASS" : ">> Group 2 FAIL", "   max abs diff = ", maxdiff)
    
    return ok
end

function test_backward_single(; n_probe=20, n_points=3, tol=1e-6)
    Random.seed!(7)
    params = make_test_params([2, 10, 10, 1])
    L = length(params)

    worst = zeros(3, 2)              
    bias_zero_ok = true

    for _ in 1:n_points
        x, t = rand(), rand()
        for (si, s) in enumerate((:x, :t))
            for case in 1:3
                g = case == 1 ? (1.0, 0.0, 0.0) :
                    case == 2 ? (0.0, 1.0, 0.0) : (0.0, 0.0, 1.0)

                _, _, _, cache = forward_jet(params, x, t, s)
                acc = zero_like(params)
                backward!(acc, params, cache, g...)

                loss_fn(p) = forward_jet(p, x, t, s)[case]

                for (k, field, idx) in sample_param_indices(params, n_probe)
                    num = central_diff(loss_fn, params, k, field, idx)
                    ana = grad_entry(acc, k, field, idx)
                    worst[case, si] = max(worst[case, si], mixed_err(ana, num))
                end

                case > 1 && (bias_zero_ok &= all(acc[L].b .== 0.0))
            end
        end
    end

    println("max error   (cols: seed :x, seed :t)")
    for (name, row) in zip(("u   ", "u_s ", "u_ss"), 1:3)
        println("  ", name, "   ", worst[row, 1], "   ", worst[row, 2])
    end
    println("last-layer bias gradient exactly 0 for u_s, u_ss: ", bias_zero_ok)

    ok = all(worst .< tol) && bias_zero_ok
    println(ok ? ">> Group 3 PASS" : ">> Group 3 FAIL")
    
    return ok
end

function test_compute_gradients(; n_probe=30, tol=1e-5)
    Random.seed!(11)
    params   = make_test_params([2, 10, 10, 1])
    interior = sample_interior(50)
    ic       = sample_ic(20)
    bc       = sample_bc(20)

    ok = true
    cases = [(1.0, (ic0=1.0, ic1=1.0, bc=1.0)),
             (3.0, (ic0=2.0, ic1=0.5, bc=3.0))]

    for (c, lam) in cases
        grads, losses = compute_gradients(params, interior, ic, bc, c, lam, 0.0, 5, false)
        loss_fn(p) = compute_gradients(p, interior, ic, bc, c, lam, 0.0, 5, false)[2].total

        worst = 0.0
        for (k, field, idx) in sample_param_indices(params, n_probe)
            num = central_diff(loss_fn, params, k, field, idx)
            worst = max(worst, mixed_err(grad_entry(grads, k, field, idx), num))
        end
        pass = worst < tol
        ok &= pass
        println(pass ? "PASS  " : "FAIL  ", "c=", c, " λ=", lam,
                "   worst err=", worst, "   total loss=", losses.total)
    end

    g, l = compute_gradients(params, interior, ic, bc, 1.0,
                             (ic0=1.0, ic1=1.0, bc=1.0), 1.0, 5, true)
    finite = isfinite(l.total) && all(isfinite, g[1].W)
    ok &= finite
    println(finite ? "PASS  " : "FAIL  ", "causal weights on, smoke test   total loss=", l.total)

    println(ok ? ">> Group 4 PASS" : ">> Group 4 FAIL")
    
    return ok
end

function run_all_checks()
    test_hyperdual()          || return false
    test_forward_agreement()  || return false
    test_backward_single()    || return false
    test_compute_gradients()  || return false
    println("\nAll checks passed.")
    
    return true
end

if abspath(PROGRAM_FILE) == @__FILE__
    for f in ("hyperdual.jl", "network.jl", "sampling.jl", "losses.jl", "autodiff.jl")
        include(joinpath(@__DIR__, f))
    end
    run_all_checks()
end
