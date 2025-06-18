"""
This script generates publication-quality plots for the BSP model
with the OptSwitch package. The plots demonstrate the performance of
various machine learning strategies for optimal switching problems.
"""

# --- 1. Imports and Initialization ---
using DrWatson                           # Project organization and reproducibility
@quickactivate                           # Activate the current project
push!(LOAD_PATH, srcdir())               # Add source directory to load path
current_dir = scriptsdir()
push!(LOAD_PATH, scriptsdir("plot_functions_article"))            # Add current directory to load path

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
"""
    get_parameters()

Configure all parameters needed for the BSP model simulation and analysis.

Returns:
- Dictionary of model parameters including dimensions, modes, timepoints and other settings
"""
function get_parameters()
    d = 1                # Dimension of random process (1D state space)
    J = 10               # Number of operational modes
    K = 20000            # Number of trajectories for Monte Carlo simulation
    N = 36               # Number of time points in simulation
    L = 1                # Number of one-step trajectories in training
    t_start = 0.0f0      # Start time of simulation
    t_end = 1.f0 |> Float32  # End time of simulation
    dt = (t_end - t_start) / N  # Time step
    p = ()               # Additional parameters (empty in this case)
    
    return Dict(
        "d" => d,
        "J" => J,
        "N" => N,
        "dt" => dt,
        "K" => K,
        "L" => L,
        "t_start" => t_start,
        "t_end" => t_end,
        "p" => p,
        "experiment" => "BSP"
    )
end 

# Load parameters
new_params = get_parameters()
@unpack d, J, N, dt, K, L, t_start, t_end, p = new_params

# --- 3. Process Definitions ---
"""
    drift(u, p, t)

Compute the drift term of the stochastic process.
Uses a mean-reverting model with time-varying mean.
"""
function drift(u, p, t)
    # Mean-reverting with nonlinear scaling
    α = 0.5f0  # Mean reversion speed
    μ = sin(2π * t)  # Time-varying mean
    return @SVector [-α * (u[1] - μ)]
end

"""
    dispersion(u, p, t)

Compute the dispersion (diffusion) term of the stochastic process.
Implements state-dependent volatility.
"""
function dispersion(u, p, t)
    # State-dependent volatility
    σ = 0.5f0 + 0.2f0 * abs(u[1])  # Increases with distance from zero
    return @SMatrix [σ]
end

# Initialize random process
RandomProcess = OptSwitch.SDEprocess(drift, dispersion)

# --- 4. Payoff and Cost Functions ---
"""
    payoff(x, t)

Compute the payoff vector for the switching problem.
Returns a 10-dimensional vector of payoffs for each mode.
Payoffs are based on discretized state space intervals.
"""
function payoff(x, t) # return J-vector
    x = x[1]
    if x > 2 || x < -2
        return @SVector zeros(Float32, 10)
    else
        y = x + 2.0f0
        interval = y ÷ 0.4f0 |> Int64
        
        # Base payoff for correct interval
        payoffs = zeros(Float32, 10)
        payoffs[interval+1] = 1.0f0
        
        # Add small payoffs for adjacent modes to create strategic paths
        if interval > 0
            payoffs[interval] = 0.2f0
        end
        if interval < 9 
            payoffs[interval+2] = 0.2f0
        end
        
        return SVector{10}(payoffs)
    end
end

"""
    cost(x, t)

Compute the cost matrix for switching between modes.
Returns a 10x10 matrix where entry (i,j) represents the cost of switching from mode i to j.
Costs increase with mode distance but plateau at a maximum.
"""
function cost(x, t) # Return JxJ matrix
    # Base switching cost
    base_cost = 0.05f0
    
    # Cost increases with mode distance but plateaus
    costs = [i == j ? 0.0f0 : base_cost * min(abs(i-j), 3) for i in 1:10, j in 1:10]
    return SMatrix{10,10}(costs)
end

# Initialize payoff model
payoff_params = (1,)
payoffmodel = OptSwitch.PayOffModel(payoff_params, payoff, cost)

# --- 5. Data Generation and Processing ---
push!(LOAD_PATH, scriptsdir("plotting"))
using bsp_plots

# Load pre-trained machine learning models
data_dir = datadir("BSP/machines")
m = filter(x -> occursin("20000_", x), 
    readdir(data_dir))
mods = OptSwitch.load_models(data_dir * "/" .* m)

# Generate trajectories for analysis
times = 1:N
initial_mode = 1
K_test = 500
x_init = randn(Float32,(d,1000)) .* 0.5f0  # Initial state values with noise
trajs = OptSwitch.generate_paths(RandomProcess, x_init, t_start, N, dt, p)
sample_paths = trajs[:, times, 1:100]

# Filter out unwanted models
filter!(mods) do x
    x.name != "ridge" && x.name != "lasso"
end

# --- 6. Analysis and Visualization ---
println("Starting analysis and visualization...")

# Calculate benchmarks for comparison
println("Calculating benchmark strategies...")

# A posteriori strategy (optimal with perfect hindsight)
optimal_value = calculate_value_function(sample_paths, cost, payoff, times, dt, J)
optimal_strategies = determine_optimal_strategy(optimal_value, sample_paths, cost, times, dt, J, initial_mode)
println("✓ Calculated a posteriori optimal strategy")

# Greedy strategy (myopic decision-making without lookahead)
greedy_strategies = calculate_greedy_value_matrix(sample_paths, payoff, cost, times, dt, initial_mode, J)[2]
println("✓ Calculated greedy strategy")

# --- 7. Model Comparison and Strategy Analysis ---
println("Performing model comparison and strategy analysis...")

# Prepare the comparative analysis of different ML models
strat_analysis = prepare_strategy_analysis(
    mods, payoff, cost, sample_paths, times, dt, J,
    other_strategies=[(optimal_strategies, "a posteriori"),(greedy_strategies,"greedy")],
    initial_mode=initial_mode
)
println("✓ Completed strategy analysis")

# Generate summary dataframe with strategy rankings
df_summary = strat_analysis.summary_df

# Extract distance metrics between strategies
println("Computing performance metrics...")
dist = strat_analysis.strat_dist

# Create enhanced dataframe with decision similarity and prediction accuracy metrics
joined_df = leftjoin(dist.distances, strat_analysis.summary_df, on = :Strategy)
joined_df = select(joined_df, Not([:Rank,:DifferenceFromOptimal]))
df = joined_df

# Calculate additional performance metrics
df.decision_similiarity = 1 .- df.Decision_Distance_To_Reference  # Higher is better
df.prediction_accuracy = 1 ./ (1 .+ df.Prediction_Error)         # Higher is better

# Select relevant columns for the paper
df = df[:,[:Strategy, :NormalizedFinalValue,:decision_similiarity,:prediction_accuracy]]

# Prepare dataframe for LaTeX output and save to CSV
df = prepare_for_latex(df)
using CSV
CSV.write(datadir("BSP/BSP_strategy_summary.csv"), df)
println("✓ Saved strategy summary to CSV")

# Generate visualizations for research article
println("Generating main visualizations...")
df = strat_analysis.summary_df
plts = plot_strategy_analysis(strat_analysis)
mkpath(plotsdir("BSP"))
# Save mode distribution visualization showing operational states over time
save(plotsdir("BSP/BSP_switching_strategies.pdf"), plts[1])
println("✓ Saved switching strategies visualization")

# Save performance comparison of different strategies
save(plotsdir("BSP/BSP_strategy_performance.pdf"), plts[2])
println("✓ Saved strategy performance comparison")

# --- 8. Payoff Structure Visualization ---
println("Creating payoff structure visualizations...")

# Generate 3D visualization of the payoff structure
fig2 = visualize_payoff_3d(payoff, cost)
save(plotsdir("BSP/payoff_3d.png"), fig2, dpi=300)
println("✓ Saved payoff structure visualization")

# --- 9. Finalization ---
println("All visualizations completed successfully!")