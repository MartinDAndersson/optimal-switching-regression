"""
This script generates publication-quality plots for the ACLP model
with the OptSwitch package. The plots demonstrate the performance of
various machine learning strategies for optimal switching problems.
"""

# --- 1. Imports and Initialization ---
using DrWatson                           # Project organization and reproducibility
@quickactivate                           # Activate the current project
push!(LOAD_PATH, srcdir())               # Add source directory to load path
current_dir = scriptsdir()
push!(LOAD_PATH, current_dir)            # Add current directory to load path

using Revise, Pkg, JLD2, FileIO, Random, Distributions, LinearAlgebra
using Parameters, BenchmarkTools, Printf, BSON, TimerOutputs, StaticArrays
using ProgressMeter, DataFrames, MLJ, Glob, MLUtils, CairoMakie
using AlgebraOfGraphics, Makie, DataFramesMeta
using Lux, ADTypes, MLUtils, Optimisers, Zygote, Random, Statistics, Printf
import SimpleChains: static
import OptSwitch
# Initialize random seed for reproducibility
rng = MersenneTwister()
Random.seed!(rng, 54321)

# --- 2. Parameter Configuration ---

d = 9  # Dimension of random process
n_fuels = 3
N = 90 # Time slices
#dt = 1//N # dt
J = 4 # Different modes
K = 50000 # Number of trajectories
#N=10 # Number of time points
L = 10# L one-step trajectories in training
t_start = 0f0 # Time start value
t_end = 1.f0 |> Float32 # Time end value
dt = (t_end-t_start)/N

new_params = Dict("d"=>d,"J"=>J,"N"=>N,"dt"=>dt,"K"=>K,"L"=>L,"t_start"=>t_start,
    "t_end"=>t_end,"p"=>(1,),"experiment"=>"aid")

C = [50 10 10; 60 0 10; 60 10 0; 70 0 0] .|> Float32 # Capacities at different modes, fuel s in mode i
total_C = sum(C,dims=2)
P_max = 3000.f0 # max electricity prices

alpha = [4,8,8,8] .|> Float32 # Z and A #
beta = [15 0.1 0.1 0; 0.1 0.5 -0.1 0; 0.1 -0.1 0.5 0; 0 0 0 0.5] .|> Float32 # each row beta f
# The above is probably for the jump process?

s_long = [-4 0 0 1 0; 0 0 0 0 0; 0 2 -1 0 1; 0 0 0 0 0; 0 1 1 1 -1] .|> Float32
s_alpha = [0.4 0 0.8 0 0] .|> Float32

# Unclear why both s_long and s_alpha?
s_sigma = 
[2.5 1.25 1.25 1.25 1.25; 
1.25 5 1.25 1.25 1.25; 
1.25 1.25 15 1.25 1.25;
0.25 0.25 0.25 1.5 1.25;
1.25 1.25 1.25 1.25 3
] ./ 100 .|> Float32 # * sqrt(t)

xi = s_alpha .* s_long

λ_poisson = 12.f0 #average number of jumps in electricity prices per year
λ_exp = 15.f0 #inverse of average intensity of jumps

#exp_params = OptSwitch.parameters(d,N,dt,J,K,L,t_start,t_end,p)
p=(alpha,xi,beta,s_sigma,n_fuels,λ_poisson,λ_exp,d)

new_params = Dict("d"=>d,"J"=>J,"N"=>N,"dt"=>dt,"K"=>K,"L"=>L,"t_start"=>t_start,
    "t_end"=>t_end,"p"=>p,"experiment"=>"aid")


"""
    drift(u, p, t)

Compute the drift term of the stochastic process.
Separates the state vector into Z (first n_fuels+1 components) and S (remaining components)
and applies appropriate drift terms to each.

Returns:
- Drift vector for the stochastic process
"""
function drift(u,p,t)
    α, Ξ, β, Σ, n_fuels, λ_poisson, λ_exp,d = p
    Z = @view u[1:n_fuels+1]
    S = @view u[n_fuels+1+1:end]
    Z_drift = -α .* Z
    S_drift = Ξ * S
    return vcat(Z_drift,S_drift)
end

"""
    dispersion(u, p, t)

Compute the dispersion (diffusion) term of the stochastic process.
Builds a d×d matrix with block structure for Z and S components.

Returns:
- Dispersion matrix for the stochastic process
"""

function dispersion(u,p,t)
    α, Ξ, β, Σ, n_fuels, λ_poisson, λ_exp,d = p
    du = zeros(Float32,d,d)
    Z = @view u[1:n_fuels+1]
    S = @view u[n_fuels+1+1:end]
    Z_disp = β
    S_disp = Σ * diagm(S) 
    du[1:n_fuels+1,1:n_fuels+1] = Z_disp
    du[n_fuels+2:end,n_fuels+2:end] = S_disp
    return du  #* r
end
"""
    jump(u, p, dt)

Compute jump component of the stochastic process.
Implements Poisson jumps with exponential size distribution for price components.

Returns:
- Jump contribution to state vector
"""
function jump(u,p,dt)
    α, Ξ, β, Σ, n_fuels, λ_poisson, λ_exp,d = p
    n_prices = d - (n_fuels + 1)  # number of price components
    deltaN = rand(Binomial(1,λ_poisson*Float64(dt)), n_prices)
    size = rand(Exponential(1/λ_exp), n_prices)
    jump = exp.(size).^deltaN
    du = ones(d)
    du[n_fuels+2:end] = jump  # Apply jumps only to price components
    du = (du .- 1) .* u
    return du
end

x0 = [0,0,0,0,20,60,40,20,120] .|> Float32 #[D0, A^1_0, A^2_0, A^3_0, S^0_0, S^1_0, S^2_0, S^3_0]

D0 = 70.f0
x_init = copy(x0)
x_init[1] = D0
x_init[2:4] .= 1

x_init = repeat(x0,inner=(1,K)) .+ randn(Float32,(d,K)) .* 0.005f0

RandomProcess = OptSwitch.JumpProcess(drift,dispersion,jump)

# --- 4. Payoff and Cost Functions ---

ZseasMax = [1.00, 0.87, 0.87, 0.9]
ZseasMin = [0.70, 0.67, 0.67, 0.7]

ZseasMaxTrans = quantile(Normal(),ZseasMax)
ZseasMinTrans = quantile(Normal(),ZseasMin)

ZseasMaxTrans[1] = ZseasMax[1]
ZseasMinTrans[1] = ZseasMin[1]

ZseasSum = ZseasMaxTrans .+ ZseasMinTrans
ZseasDif = ZseasMaxTrans .- ZseasMinTrans

Zshift = [0,0,0,0]
Dgrowth = 0.0

# Technology parameters
h_CO2 = [0.5, 2, 0]      # CO2 production for technologies
h_tech = [1.0, 1.5, 1.5] # Technology efficiency factors

"""
    create_matrix(M, J)

Create a switching matrix indicating which fuels change when switching between modes.

Args:
- M: Capacity matrix where rows are modes and columns are fuels
- J: Number of modes

Returns:
- 3D array where element [i,j,k] is 1 if fuel k changes when switching from mode i to j
"""

function create_matrix(M,J)
    J,n_fuels = size(M)
    res = zeros(J,J,n_fuels)
    for current_mode in 1:J
        for next_mode in 1:J
            bit_matrix = M[current_mode,:] .!== M[next_mode,:]
            res[current_mode,next_mode,:] = bit_matrix
        end
    end
    return res
end

# Combine all payoff parameters
payoff_params=(ZseasMax,ZseasMax,ZseasMaxTrans,ZseasMinTrans,ZseasSum,ZseasDif,Zshift,Dgrowth,n_fuels,h_CO2,h_tech,create_matrix(C,J))

# Combine all payoff parameters
payoff_params = (ZseasMax, ZseasMax, ZseasMaxTrans, ZseasMinTrans, ZseasSum, ZseasDif, 
                Zshift, Dgrowth, n_fuels, h_CO2, h_tech, create_matrix(C, J))

"""
    getAdjX(X, p, time)

Adjust the state vector for seasonal variations and get availability rates.

Args:
- X: State vector
- p: Payoff parameters
- time: Current time

Returns:
- Adjusted state vector
"""

function getAdjX(X,p,time)
    aX = copy(X)
    ZseasMax,ZseasMax,ZseasMaxTrans,ZseasMinTrans,ZseasSum,ZseasDif,Zshift,Dgrowth,n_fuels,h_CO2,h_tech,switch_M = p
    Zseas = 0.5 .* ZseasSum .+ 0.5 .* ZseasDif .* cos.(2 * π * time .- Zshift)
    aX[1] = X[1] .+ (D0 .+ Dgrowth .* time) .* Zseas[1]
    aX[2:(1 + n_fuels)] = X[2:(1 + n_fuels)] .+ 3. *Zseas[2:(1 + n_fuels)]
    aX[2:(1 + n_fuels)] = cdf.(Normal(), aX[2:(1 + n_fuels)])
    return aX
end

"""
    payoff(x, p, t)

Compute the payoff vector for each operational mode.
Considers demand, availability, CO2 prices, technology prices and electricity price.

Returns:
- Vector of profits for each mode
"""

function payoff(x,p,t)
    aX = getAdjX(x,p,t)
    ZseasMax,ZseasMax,ZseasMaxTrans,ZseasMinTrans,ZseasSum,ZseasDif,Zshift,Dgrowth,n_fuels,h_CO2,h_tech,switch_M = p
    @views D = aX[1] # demand
    @views A = aX[2:(1+n_fuels)] # availability
    @views CO2 = aX[(1+n_fuels)+1] # CO2 prices
    @views S = aX[(1+n_fuels)+2:(end-1)] # technology prices
    @views price = aX[end] # electricity price

    profit = zeros(J)
    for j in 1:J
        S2 = h_CO2 .* CO2  + h_tech .* S
        curr_C = C[j,:] .* A

        curr_total_C = sum(curr_C)

        fixed_cost = sum(C[j,:] .* S2)

        rev = min(curr_total_C,D) * price

        profit[j]=rev-fixed_cost
    end

    return profit #want to return two arrays of shape (M_training, J)
end
"""
    cost(x, p, t)

Compute the cost matrix for switching between modes.
Costs depend on the technology prices and which technologies change.

Returns:
- JxJ matrix where element [i,j] is the cost of switching from mode i to j
"""

function cost(x,p,t)
    ZseasMax,ZseasMax,ZseasMaxTrans,ZseasMinTrans,ZseasSum,ZseasDif,Zshift,Dgrowth,n_fuels,h_CO2,h_tech,switch_M = p
    J = size(switch_M)[1]
    cost_matrix = zeros(J,J)
    @views S =  x[(1+n_fuels)+2:(end-1)]
    for i in 1:J
        for j in 1:J
            cost_matrix[i,j] = dot(switch_M[i,j,:],S) .* 1/3 .+ 0.001
        end
    end
    return cost_matrix
end

"""
    get_revandcost_closure(payoff, cost, payoff_params)

Create closures for revenue and cost functions with fixed payoff parameters.

Returns:
- Tuple of closures (rev, c) that take only state and time as arguments
"""

function get_revandcost_closure(payoff,cost,payoff_params)
    rev(x,t) = payoff(x,payoff_params,t) 
    c(x,t) = cost(x,payoff_params,t)
    return rev,c
end

# Create closures and initialize payoff model
rev, c = get_revandcost_closure(payoff, cost, payoff_params)
payoffmodel = OptSwitch.PayOffModel(payoff_params, rev, c)



# --- 5. Model Loading and Strategy Analysis ---
# Setup for testing with fewer trajectories
K = 2000 
x0 = [70, 0, 0, 0, 20, 60, 40, 20, 120] .|> Float32
D0 = 70.f0
x_init = copy(x0)
x_init[1] = D0
x_init[2:4] .= 1
x_init = repeat(x_init, inner=(1, K)) .+ randn(Float32, (d, K)) .* 0.005f0

dir="aid"
# Load pre-trained machine learning models
data_dir = datadir(dir,"machines")
filter_out = ["network 3", "algorithm=knn", "network 2", "network 4"]
m = filter(x -> occursin("L=1", x) && occursin("50000_", x) && 
    !any(pattern -> occursin(pattern, x), filter_out),
    readdir(data_dir))


# Load models
mods = OptSwitch.load_models(data_dir * "/" .* m)

# --- 6. Trajectory Generation and Visualization ---
# Generate trajectories and select sample for analysis
times = 1:91
trajs = OptSwitch.generate_paths(RandomProcess, x_init, t_start, N, dt, p)
sample_paths = trajs[:, times, 1:100]

# Visualize sample paths
using CairoMakie
plt = lines(times, sample_paths[9, :, 1])
for i in 2:9
    lines!(times, sample_paths[i, :, 1])
end

push!(LOAD_PATH, scriptsdir("plotting"))
using aclp_plots
initial_mode = 1

# --- 7. Strategy Analysis and Evaluation ---
# Calculate benchmark strategies
optimal_value = calculate_value_function(sample_paths, c, rev, times, dt, J)
greedy_strategies = calculate_greedy_value_matrix(sample_paths, rev, c, times, dt, initial_mode, J)[2]
optimal_strategies = determine_optimal_strategy(optimal_value, sample_paths, c, times, dt, J, initial_mode)

# Compare model performance
mean(optimal_value[1, :, 1])
strat_analysis = prepare_strategy_analysis(mods, rev, c, sample_paths, times,
    dt, J, other_strategies=[(optimal_strategies, "a posteriori"),
    (greedy_strategies, "greedy")], initial_mode=initial_mode)
df_summary = sort(strat_analysis.summary_df, :Rank)
strat_analysis.summary_df

# --- 8. Results Analysis and Visualization ---
# Extract and process performance metrics
dist = strat_analysis.strat_dist
dist.distances
joined_df = leftjoin(dist.distances, strat_analysis.summary_df, on=:Strategy)
joined_df = select(joined_df, Not([:Rank, :DifferenceFromOptimal]))

# Calculate performance metrics
df = joined_df
df.decision_similiarity = 1 .- df.Decision_Distance_To_Reference
df.prediction_accuracy = 1 ./ (1 .+ df.Prediction_Error)

# Prepare for output
df = df[:, [:Strategy, :NormalizedFinalValue, :decision_similiarity, :prediction_accuracy]]
df = prepare_for_latex(df)
using CSV
CSV.write(datadir("aid/aid_strategy_summary.csv"), df)

# Generate and save plots
plts = plot_strategy_analysis(strat_analysis)
save(plotsdir("aid/aid_switching_strategies.pdf"), plts[1])
save(plotsdir("aid/aid_strategy_performance.pdf"), plts[2])
plts = plot_value_prediction_error(mods, sample_paths, times, rev, c, dt, J)