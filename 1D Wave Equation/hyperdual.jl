struct HyperDual
    val::Float64
    eps1::Float64
    eps2::Float64
    eps12::Float64
end

constant(a::Float64) = HyperDual(a, 0.0, 0.0, 0.0)

seed(a::Float64) = HyperDual(a, 1.0, 1.0, 0.0)

function add(a::HyperDual, b::HyperDual)
    return HyperDual(a.val + b.val, a.eps1 + b.eps1,
                     a.eps2 + b.eps2, a.eps12 + b.eps12)
end

function sub(a::HyperDual, b::HyperDual)
    return HyperDual(a.val - b.val, a.eps1 - b.eps1,
                     a.eps2 - b.eps2, a.eps12 - b.eps12)
end

function mul(a::HyperDual, b::HyperDual)
    val   = a.val * b.val
    eps1  = a.val*b.eps1 + a.eps1*b.val
    eps2  = a.val*b.eps2 + a.eps2*b.val
    eps12 = a.val*b.eps12 + a.eps1*b.eps2 + a.eps2*b.eps1 + a.eps12*b.val
    return HyperDual(val, eps1, eps2, eps12)
end

function scale(a::HyperDual, k::Float64)
    return HyperDual(k*a.val, k*a.eps1, k*a.eps2, k*a.eps12)
end

function tanh_hd(a::HyperDual)
    t  = tanh(a.val)
    d1 = 1 - t^2          
    d2 = -2 * t * d1     
    return HyperDual(t, d1 * a.eps1, d1 * a.eps2, d1 * a.eps12 + d2 * a.eps1 * a.eps2)
end