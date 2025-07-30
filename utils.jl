using Random, Distributions, Plots, Statistics, LinearAlgebra, JLD2, Base.Threads, Roots, Combinatorics, StatsBase

println("Threads: ", nthreads())

const K2C = 2.0;

#################### UTILITIES ####################

@. function CDF_power_bound(x::Float64, lwbd::Int64, upbd::Int64, γ::Float64)
    return ((upbd^(1-γ) - lwbd^(1-γ)) * x + lwbd^(1-γ))^(1 / (1 - γ))
end

@. function CDF_pow_all(x::Float64, γ::Float64, mean_deg::Union{Int,Int32})
    return mean_deg * (γ-2)/(γ-1)*x^(1/(1-γ))
end

@. function safe_sqrt(x::Float64)
        return x>0.0 ? sqrt(x) : 0.0
    end

@. function safe_log(x::Float64)
        return x>0.0 ? log(x) : 0.0
    end

@. function safe_div(x::Float64, y::Float64)
        return y>0.0 ? x/y : 0.0
    end

function CDF_power(x::Float64, lwbd::Int64, upbd::Int64, γ::Float64)
    return ((upbd^(1-γ) - lwbd^(1-γ)) * x + lwbd^(1-γ))^(1 / (1 - γ))
end

###################### NETWORK GENERATION ######################

#Chung-Lu methods
function create_L_CL_UN(n::Int, lwbd::Int, upbd::Int)
    Adj = [Int32[] for _ in 1:n]  # Initialize as Vector{Vector{Int32}}
    target = rand(lwbd:upbd, n)
    sum_deg = sum(target)

    for i in 1:n
        for j in i+1:n
            prob = target[i]*target[j]/sum_deg
            if rand() < prob
                push!(Adj[i], Int32(j))
                push!(Adj[j], Int32(i))
            end
        end
    end
    degree = [length(Adj[i]) for i in 1:n]
    return Adj, degree, target
end

function create_L_CL_ER(n::Int, mean_deg::Int)
    Adj = [Int32[] for _ in 1:n] 
    p = mean_deg / n
    for i in 1:n
        for j in i+1:n
            if rand() < p
                push!(Adj[i], Int32(j))
                push!(Adj[j], Int32(i))
            end
        end
    end
    degree = [length(Adj[i]) for i in 1:n]
    return Adj, degree
end

function create_L_CL_PL_bound(n::Int, γ::Float64, lwbd::Int, upbd::Int)
    Adj = [Int32[] for _ in 1:n]  # Initialize as Vector{Vector{Int32}}
    target = [round(Int32,CDF_power_bound(rand(Uniform(0.0,1.0)), lwbd, upbd, γ)) for i in 1:n]
    sum_deg = sum(target)
    for i in 1:n
        for j in i+1:n
            prob = target[i]*target[j]/sum_deg
            if rand() < prob
                push!(Adj[i], Int32(j))
                push!(Adj[j], Int32(i))
            end
        end
    end
    degree = [length(Adj[i]) for i in 1:n]
    return Adj, degree, target
end

function create_L_CL_PL(n::Int, γ::Float64, mean_deg::Union{Int,Int32})
    Adj = [Int32[] for _ in 1:n]  # Initialize as Vector{Vector{Int32}}
    target = [round(Int32,CDF_pow_all(rand(Uniform(0.0,1.0)), γ, mean_deg)) for i in 1:n]
    sum_deg = sum(target)
    for i in 1:n
        for j in i+1:n
            prob = target[i]*target[j]/sum_deg
            if rand() < prob
                push!(Adj[i], Int32(j))
                push!(Adj[j], Int32(i))
            end
        end
    end
    degree = [length(Adj[i]) for i in 1:n]
    return Adj, degree, target
end

function create_L_CL_NP(n::Int, Np::Int, ps::Vector{Int})
    Adj = [Int32[] for _ in 1:n]  # Initialize as Vector{Vector{Int32}}
    target = [ps[rand(1:Np)] for _ in 1:n]
    sum_deg = sum(target)
    for i in 1:n
        for j in i+1:n
            if rand() < target[i]*target[j]/sum_deg
                push!(Adj[i], Int32(j))
                push!(Adj[j], Int32(i))
            end
        end
    end
    degree = [length(Adj[i]) for i in 1:n]
    return Adj, degree, target
end

function create_L_CL(n::Int, target::Vector{Int})
    Adj = [Int32[] for _ in 1:n]  # Initialize as Vector{Vector{Int32}}
    sum_deg = sum(target)
    for i in 1:n
        for j in i+1:n
            prob = target[i]*target[j]/sum_deg
            if rand() < prob
                push!(Adj[i], Int32(j))
                push!(Adj[j], Int32(i))
            end
        end
    end
    degree = [length(Adj[i]) for i in 1:n]
    return Adj, degree, target
end

#Configuration model

function create_L_CF(n::Int, target::Vector{Int})
    Adj = [Int32[] for _ in 1:n]  # Initialize as Vector{Vector{Int32}}
    
    if sum(target) % 2 != 0
        target[rand(1:n)] += 1
    end
    rd = 0
    remain_stubs = copy(target)
    while any(remain_stubs .> 0) && rd < 1000
        current_stubs = Int32[]
        for i in 1:n
            for _ in 1:remain_stubs[i]
                push!(current_stubs, Int32(i))
            end
        end
        shuffle!(current_stubs)

        k = 1
        while k <= length(current_stubs)-1
            u = current_stubs[k]
            v = current_stubs[k+1]
            if u != v
                push!(Adj[u], v)
                push!(Adj[v], u)
                remain_stubs[u] -= 1
                remain_stubs[v] -= 1
            end
            k += 2
        end
        rd += 1
    end

    degree = [length(Adj[i]) for i in 1:n]
    return Adj, degree, target
end

################## Triangle generation ##################
function create_3D(n::Int, target::Union{Vector{Int}, Vector{Int32}})
    Adj = [Matrix{Int16}(undef, 0, 2) for _ in 1:n]
    sumdeg = sum(target)
    for i in 1:n
        for j in i+1:n
            for k in j+1:n
                prob = 2 * target[i] * target[j] * target[k] / sumdeg^2
                if rand() < prob
                    # Create new rows to add to Adj[i], Adj[j], and Adj[k]
                    new_row_i = reshape(Int16[j k], 1, 2)
                    new_row_i2 = reshape(Int16[k j], 1, 2)
                    new_row_j = reshape(Int16[i k], 1, 2)
                    new_row_j2 = reshape(Int16[k i], 1, 2)
                    new_row_k = reshape(Int16[i j], 1, 2)
                    new_row_k2 = reshape(Int16[j i], 1, 2)
                    
                    # Append the new rows to the matrices
                    Adj[i] = vcat(Adj[i], new_row_i)
                    Adj[i] = vcat(Adj[i], new_row_i2)
                    Adj[j] = vcat(Adj[j], new_row_j)
                    Adj[j] = vcat(Adj[j], new_row_j2)
                    Adj[k] = vcat(Adj[k], new_row_k)
                    Adj[k] = vcat(Adj[k], new_row_k2)
                end
            end
        end
    end
    degree = [Int32(size(Adj[i], 1)/2) for i in 1:n]
    return Adj, degree
end


function create_3D_CF(n::Int, target::Union{Vector{Int}, Vector{Int32}})
    Adj = [Matrix{Int16}(undef, 0, 2) for _ in 1:n]
    if sum(target) % 3 == 2
        target[rand(1:n)] += 1
        target[rand(1:n)] += 1
    elseif sum(target) % 3 == 1
        target[rand(1:n)] += 1
    end
    rd = 0
    remain_stubs = copy(target)
    while any(remain_stubs .> 0) && rd < 1000
        current_stubs = Int32[]
        for i in 1:n
            for _ in 1:remain_stubs[i]
                push!(current_stubs, Int32(i))
            end
        end
        shuffle!(current_stubs)

        k = 1
        while k <= length(current_stubs)-2
            u = current_stubs[k]
            v = current_stubs[k+1]
            w = current_stubs[k+2]
            if u != v && u != w && v != w
                # Create new rows to add to Adj[u], Adj[v], and Adj[w]
                new_row_u = reshape(Int16[v w], 1, 2)
                new_row_u2 = reshape(Int16[w v], 1, 2)
                new_row_v = reshape(Int16[u w], 1, 2)
                new_row_v2 = reshape(Int16[w u], 1, 2)
                new_row_w = reshape(Int16[u v], 1, 2)
                new_row_w2 = reshape(Int16[v u], 1, 2)

                # Append the new rows to the matrices
                Adj[u] = vcat(Adj[u], new_row_u)
                Adj[u] = vcat(Adj[u], new_row_u2)
                Adj[v] = vcat(Adj[v], new_row_v)
                Adj[v] = vcat(Adj[v], new_row_v2)
                Adj[w] = vcat(Adj[w], new_row_w)
                Adj[w] = vcat(Adj[w], new_row_w2)

                remain_stubs[u] -= 1
                remain_stubs[v] -= 1
                remain_stubs[w] -= 1
            end
            k += 3
        end
        rd += 1
    end
    degree = [Int32(size(Adj[i], 1)/2) for i in 1:n]
    return Adj, degree
end

############### 3BD Kuramoto Coupling CONSTANT #################

function Un_gen(n::Int, degree2::Vector{Int64}, A2::Vector{Vector{Int32}})
    Adj2 = zeros(Int32, n, n)
    for i in 1:n
        for j in 1:degree2[i]
            Adj2[i, A2[i][j]] = 1
        end
    end
    e =  eigen(Adj2)
    return e.values[end], e.vectors[:, end]
end

function num_cal(N, Un, Adj2)
    num = 0.0
    for i in 1:N
        for j in 1:N
            for k in 1:N
                num += Un[i] * Un[j]^2 * Un[k] * Adj2[i,j] * Adj2[j,k]
            end
        end
    end
    return num
end

function num_cal(Un)
    return sum(Un .^ 4)
end

function den_cal(N, deg3, A3, Un)
    den = 0.0
    for i in 1:N
        for j in 1:size(A3[i], 1)               
            den += Un[i] * Un[A3[i][j,1]]^2 * Un[A3[i][j,2]] 
        end
    end
    return den
end

###################### 3BD Kuramoto Model ######################

function dthdt(θi::Float64, ωi::Float64, Hi::Float64, ψi::Float64)
    return ωi - Hi * sin(θi - ψi)
end

function cal_order_param(K2::Float64, K3::Float64, θ::Vector{Float64}, params::Any)
    n, degree2, A2, degree3 , A3 = params
    Hn = zeros(Float64, n)
    ψn = zeros(Float64, n)
    Rn = zeros(Complex{Float64}, n)
    @threads for i in 1:n
        sum_r2, sum_r3 = 0.0 + 0.0*1im, 0.0 + 0.0*1im
  
        sum_r2 = sum(exp(1im * θ[A2[i][j]]) for j in 1:degree2[i]; init=0)

        sum_r3 = sum(exp(1im * (2*θ[A3[i][j,1]] - θ[A3[i][j,2]])) for j in 1:Int32(size(A3[i],1)); init=0)

        Comp = K2 * sum_r2 + K3 * sum_r3
        Hn[i] = abs(Comp)
        ψn[i] = angle(Comp)
        Rn[i] = sum_r2
    end
    return Hn, ψn, Rn
end

function Heun_step(θ::Vector{Float64}, ω::Vector{Float64}, dt::Float64, Hn::Vector{Float64},Ψn::Vector{Float64}, Rn::Vector{Complex{Float64}}, K2::Float64, K3::Float64, params::Any)
    n, degree2, A2, degree3, A3 = params
    params = (n, degree2, A2, degree3, A3)
    θ1 = zeros(Float64, n)
    dth1 = zeros(Float64, n)

    @threads for i in 1:n
        dth1[i] = dthdt(θ[i], ω[i], Hn[i], Ψn[i])
        θ1[i] = θ[i] + dth1[i] * dt
    end

    H1, Ψ1, R1 = cal_order_param(K2, K3, θ1, params)
    @threads for i in 1:n
        dth2 = dthdt(θ1[i], ω[i], H1[i], Ψ1[i])
        θ[i] += 0.5 * (dth1[i] + dth2) * dt
    end

    return θ
end

function R1_th_Int(init_val::Float64, K2::Float64, K3::Float64, params::Any)
    n, degree2, A2, degree3, A3 = params
    sumdeg2 = sum(degree2)
    Qn = [init_val*degree2[i] for i in 1:n] 
    Qnc = zeros(Float64, 200, n)
    Qnc[1, :] .= Qn
    Rn = zeros(Float64, n)
    for t in 2:200
        @threads for i in 1:n
            r1 = 0.0
            r2 = 0.0
            for j in 1:degree2[i]
                r1 += safe_div(safe_sqrt(Qnc[t-1,A2[i][j]]^2+1)-1, Qnc[t-1,A2[i][j]])
            end
            for j in 1:Int32(size(A3[i],1))
                p1 = safe_div(safe_sqrt(Qnc[t-1,A3[i][j,2]]^2+1)-1, Qnc[t-1,A3[i][j,2]])
                p2 = safe_div(safe_sqrt(Qnc[t-1,A3[i][j,1]]^2+1)-1, Qnc[t-1,A3[i][j,1]])
                r2 += p1^2 * p2
            end
            Qnc[t,i] = K2*r1 + K3*r2
            Rn[i] = r1
        end
    end
    R = sum(Rn)/sum(degree2)
    return R
end

# function R1_th_Int(init_val::Union{Complex{Float64},Float64}, K2::Float64, K3::Float64, params::Any)
#     n, degree2, A2, degree3, A3 = params
#     sumdeg2 = sum(degree2)
    
#     # Initialize arrays with Complex{Float64}
#     Qn = [init_val * degree2[i] for i in 1:n]
#     Qnc = zeros(Complex{Float64}, 100, n)
#     Qnc[1, :] .= Qn
#     Rn = zeros(Complex{Float64}, n)
    
#     # Time stepping loop
#     for t in 2:100
#         @threads for i in 1:n
#             r1 = Complex{Float64}(0.0)
#             r2 = Complex{Float64}(0.0)
            
#             # Compute r1 for second-order interactions
#             for j in 1:degree2[i]
#                 r1 += (sqrt(Qnc[t-1, A2[i][j]]^2 + 1) - 1) / Qnc[t-1, A2[i][j]]
#             end
            
#             # Compute r2 for third-order interactions
#             for j in 1:Int32(size(A3[i], 1))
#                 p1 = (sqrt(Qnc[t-1, A3[i][j, 2]]^2 + 1) - 1) / Qnc[t-1, A3[i][j, 2]]
#                 p2 = (sqrt(Qnc[t-1, A3[i][j, 1]]^2 + 1) - 1) / Qnc[t-1, A3[i][j, 1]]
#                 r2 += p1^2 * p2
#             end
            
#             # Update Qnc with complex values
#             Qnc[t, i] = K2 * r1 + K3 * r2
#             Rn[i] = r1
#         end
#     end
    
#     # Compute R as a real Float64, taking the real part of the sum
#     R =real(sum(Rn)) / sumdeg2
#     return R
# end

function R1_th_sum(init_val::Float64, K2::Float64, K3::Float64, omega::Vector{Float64}, params::Any)
    n, degree2, A2, degree3, A3 = params
    Qn = [init_val*degree2[i] for i in 1:n]
    Qnc = zeros(Float64, 200, n)
    Qnc[1, :] .= Qn
    Rn = zeros(Float64, n)
    for t in 2:200
        @threads for i in 1:n
            r1 = 0.0
            r2 = 0.0
            for j in 1:degree2[i]
                if abs(omega[A2[i][j]]) < Qnc[t-1,A2[i][j]]
                    r1 += safe_sqrt(1- safe_div(omega[A2[i][j]]^2, Qnc[t-1,A2[i][j]]^2))
                end
            end
            for j in 1:Int32(size(A3[i],1))
                if abs(omega[A3[i][j,1]]) < Qnc[t-1,A3[i][j,1]] && abs(omega[A3[i][j,2]]) < Qnc[t-1,A3[i][j,2]]
                    p1 = 1- 2*safe_div(omega[A3[i][j,1]]^2, Qnc[t-1,A3[i][j,1]]^2)
                    p2 = safe_sqrt(1- safe_div(omega[A3[i][j,2]]^2, Qnc[t-1,A3[i][j,2]]^2))
                    r2 += p1 * p2
                elseif abs(omega[A3[i][j,1]]) > Qnc[t-1,A3[i][j,1]] && abs(omega[A3[i][j,2]]) < Qnc[t-1,A3[i][j,2]]
                    p1 = (2*abs(omega[A3[i][j,1]])/Qnc[t-1,A3[i][j,1]]^2)*(safe_sqrt(omega[A3[i][j,1]]^2 - Qnc[t-1,A3[i][j,1]]^2)-abs(omega[A3[i][j,1]]))+1
                    p2 = safe_sqrt(1- safe_div(omega[A3[i][j,2]]^2, Qnc[t-1,A3[i][j,2]]^2))
                    r2 += p1 * p2
                else 
                    r2 += 0.0
                end
            end
            Qnc[t,i] = K2*r1 + K3*r2
            Rn[i] = r1
        end
    end
    R = sum(Rn)/sum(degree2)
    return R
end

# function R1_th_sum(init_val::Union{Complex{Float64},Float64}, K2::Float64, K3::Float64, omega::Vector{Float64}, params::Any)
#     n, degree2, A2, degree3, A3 = params
#     sumdeg2 = sum(degree2)
    
#     # Initialize arrays with Complex{Float64}
#     Qn = [init_val * degree2[i] for i in 1:n]
#     Qnc = zeros(Complex{Float64}, 100, n)
#     Qnc[1, :] .= Qn
#     Rn = zeros(Complex{Float64}, n)
    
#     # Time stepping loop
#     for t in 2:100
#         @threads for i in 1:n
#             r1 = Complex{Float64}(0.0)
#             r2 = Complex{Float64}(0.0)
            
#             # Compute r1 for second-order interactions
#             for j in 1:degree2[i]
#                 if abs(omega[A2[i][j]]) < abs(Qnc[t-1, A2[i][j]])
#                     r1 += sqrt(1 - omega[A2[i][j]]^2 / Qnc[t-1, A2[i][j]]^2)
#                 end
#             end
            
#             # Compute r2 for third-order interactions
#             for j in 1:Int32(size(A3[i], 1))
#                 if abs(omega[A3[i][j, 1]]) < abs(Qnc[t-1, A3[i][j, 1]]) && abs(omega[A3[i][j, 2]]) < abs(Qnc[t-1, A3[i][j, 2]])
#                     p1 = 1 - 2 * (omega[A3[i][j, 1]]^2 / Qnc[t-1, A3[i][j, 1]]^2)
#                     p2 = sqrt(1 - omega[A3[i][j, 2]]^2 / Qnc[t-1, A3[i][j, 2]]^2)
#                     r2 += p1 * p2
#                 elseif abs(omega[A3[i][j, 1]]) > abs(Qnc[t-1, A3[i][j, 1]]) && abs(omega[A3[i][j, 2]]) < abs(Qnc[t-1, A3[i][j, 2]])
#                     p1 = (2 * abs(omega[A3[i][j, 1]]) / Qnc[t-1, A3[i][j, 1]]^2) * (sqrt(omega[A3[i][j, 1]]^2 - Qnc[t-1, A3[i][j, 1]]^2) - abs(omega[A3[i][j, 1]])) + 1
#                     p2 = sqrt(1 - omega[A3[i][j, 2]]^2 / Qnc[t-1, A3[i][j, 2]]^2)
#                     r2 += p1 * p2
#                 else
#                     r2 += 0.0
#                 end
#             end
            
#             # Update Qnc with complex values
#             Qnc[t, i] = K2 * r1 + K3 * r2
#             Rn[i] = r1
#         end
#     end
    
#     # Compute R as a real Float64, taking the real part of the sum
#     R = real(sum(Rn)) / sumdeg2
#     return R
# end


########################### REWIRE FUNCTIONS ###########################
function Ω_cal(Un, i, j, k)
    Ω = 0.0
    all_perms = collect(permutations([i, j, k]))
    for n in 1:length(all_perms)
        perm = all_perms[n]
        Ω += Un[perm[1]]*Un[perm[2]]^2 * Un[perm[3]]
    end
    return Ω
end

function update_A3!(A3, i,j,k)
    all_perms = collect(permutations([i, j, k]))
    for n in 1:length(all_perms)
        p = all_perms[n]
        A3[p[1]] = vcat(A3[p[1]],reshape(Int16[p[2] p[3]], 1, 2))
    end
end



function generate_sets_adjusted(lst)
    # Count frequency of each number
    freq = countmap(lst)
    result = []              # List of sets
    count_duplicates = 0     # Counter for sets with duplicates
    
    # Continue while at least 3 numbers remain
    while sum(values(freq)) >= 3
        # Get distinct numbers with positive frequency as a sorted array
        available_keys = sort(collect(k for (k, v) in freq if v > 0))
        
        if length(available_keys) >= 3
            # Form set with 3 distinct numbers
            set = sample(available_keys, 3, replace=false)
            for num in set
                freq[num] -= 1
                if freq[num] == 0
                    delete!(freq, num)
                end
            end
        else
            # Form set with possible duplicates
            set = []
            for i in 1:3
                # Recompute available keys with remaining frequency
                available = sort(collect(k for (k, v) in freq if v > 0))
                num = available[1]  # Pick smallest available number
                push!(set, num)
                freq[num] -= 1
                if freq[num] == 0
                    delete!(freq, num)
                end
            end
            count_duplicates += 1  # Set has duplicates since distinct numbers < 3
        end
        push!(result, set)
    end
    
    return result, count_duplicates
end



function rewire_3D_Ω(n, A3, degree3, Un, ways)
    hyperedges = []
    for i in 1:n
        for j in 1:size(A3[i], 1)
            push!(hyperedges, sort([i, A3[i][j,1], A3[i][j,2]]))
        end
    end

    hyperedges= unique(hyperedges);
    # Rewire the stubs
    new_A3 = [Matrix{Int16}(undef, 0, 2) for _ in 1:n]
    L = 0
    if length(hyperedges) %2 != 0
        L = length(hyperedges) -1
    else
        L = length(hyperedges)
    end

    shuffle!(hyperedges)

    for i in 1:2:L

        shuffle!(hyperedges[i])
        shuffle!(hyperedges[i+1])

        Ωi1 = Ω_cal(Un, hyperedges[i][1], hyperedges[i][2], hyperedges[i][3])
        Ωi2 = Ω_cal(Un, hyperedges[i+1][1], hyperedges[i+1][2], hyperedges[i+1][3])
        
        Ωf1 = Ω_cal(Un, hyperedges[i][1], hyperedges[i][2], hyperedges[i+1][3])
        Ωf2 = Ω_cal(Un, hyperedges[i+1][1], hyperedges[i+1][2], hyperedges[i][3])

        
            
        if ways == :decrease #decrease K3crit
            if Ωi1+ Ωi2 > Ωf1 + Ωf2
                # Rewire the stubs
                
                update_A3!(new_A3, hyperedges[i][1], hyperedges[i][2], hyperedges[i][3])
                update_A3!(new_A3, hyperedges[i+1][1], hyperedges[i+1][2], hyperedges[i+1][3])
            else
                # Keep the original stubs
                update_A3!(new_A3, hyperedges[i][1], hyperedges[i][2], hyperedges[i+1][3])
                update_A3!(new_A3, hyperedges[i+1][1], hyperedges[i+1][2], hyperedges[i][3])
            end
        elseif ways == :increase #increase K3crit
            if Ωi1+ Ωi2 < Ωf1 + Ωf2
                # Rewire the stubs
                update_A3!(new_A3, hyperedges[i][1], hyperedges[i][2], hyperedges[i][3])
                update_A3!(new_A3, hyperedges[i+1][1], hyperedges[i+1][2], hyperedges[i+1][3])
            else
                # Keep the original stubs
                update_A3!(new_A3, hyperedges[i][1], hyperedges[i][2], hyperedges[i+1][3])
                update_A3!(new_A3, hyperedges[i+1][1], hyperedges[i+1][2], hyperedges[i][3])
            end
        else
            error("Invalid way specified. Use :decrease or :increase.")
        end
    end
    # Handle the case where L is odd
    if length(hyperedges) %2 != 0
        update_A3!(new_A3, hyperedges[end][1], hyperedges[end][2], hyperedges[end][3])
    end

    deg3p = [Int(size(new_A3[i], 1)/2) for i in 1:n]
    if all(degree3 .== deg3p)
        println("Preserve degree: ", all(degree3 .== deg3p))
    else
        idx_miss = findall(x-> x==0, deg3p .== degree3)
        count_miss = abs.(deg3p .- degree3)[idx_miss]
        println("Missing degree: ", sum(count_miss))
        stubs = []
        for i in 1:length(idx_miss)
            stubs = vcat(stubs, [idx_miss[i] for _ in 1:count_miss[i]])
        end
        println("Complete stubs: ", length(stubs)%3)
        
        limited_combinations, duplicated = generate_sets_adjusted(stubs)
        println("Replica hyperedges: ", duplicated)
        
        for i in 1:length(limited_combinations)
            lc = limited_combinations[i]
            update_A3!(new_A3, lc[1], lc[2], lc[3])
        end
        deg3p = [Int(size(new_A3[i], 1)/2) for i in 1:n]
    end
    return new_A3, deg3p
end

