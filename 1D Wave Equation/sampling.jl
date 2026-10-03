using Random

function lhs_2d(n::Int)
    x_perm = randperm(n)
    t_perm = randperm(n)

    x = zeros(n)
    t = zeros(n) 

    for i in 1:n
        x[i] = ((x_perm[i] - 1) + rand()) / n
        t[i] = ((t_perm[i] - 1) + rand()) / n
    end

    return hcat(x, t)
end

function lhs_1d(n::Int)
    return [((i - 1) + rand()) / n for i in 1:n]
end

function sample_interior(n::Int)
    return lhs_2d(n)
end

function sample_ic(n::Int)
    x = lhs_1d(n); t = zeros(n)
    
    return hcat(x, t)
end

function sample_bc(n::Int)
    n_half = n ÷ 2
    left  = hcat(zeros(n_half), lhs_1d(n_half))
    right = hcat(ones(n - n_half), lhs_1d(n - n_half))
    
    return vcat(left, right)
end