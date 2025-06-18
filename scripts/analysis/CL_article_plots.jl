"""
This script generates publication-quality plots for the Carmona-Ludkovski model
in the OptSwitch package. The plots demonstrate the performance of
various machine learning strategies for optimal switching problems.
"""

# --- 1. Imports and Initialization ---
using DrWatson                           # Project organization and reproducibility
@quickactivate                           # Activate the current project
push!(LOAD_PATH, srcdir())               # Add source directory to load path
current_dir = scriptsdir("plot_functions_article")               # Get current scripts directory
push!(LOAD_PATH, current_dir)            # Add current directory to load path

using Revise, Pkg, JLD2, FileIO, Random, Distributions, LinearAlgebra
using Parameters, BenchmarkTools, Printf, BSON, TimerOutputs, StaticArrays
using ProgressMeter, DataFrames, MLJ, Glob, MLUtils, CairoMakie
using AlgebraOfGraphics, Makie, DataFramesMeta
using Lux, ADTypes, MLUtils, Random, Statistics, Printf
using CSV
import SimpleChains: static
import OptSwitch
push!(LOAD_PATH, scriptsdir("plotting"))
using cl_plots

function set_up()
    """
    Initialize all components needed for the Carmona model simulation and analysis.
    
    Returns:
    - RandomProcess: Stochastic process with drift, dispersion, and jump components
    - payoff: Payoff function for operational modes
    - cost: Switching cost function between modes
    - parameters: Dictionary of model parameters
    """
    function get_parameters(L)
        d = 2                # Dimension of random process (electricity and gas prices)
        N = 180              # Number of time points in simulation
        J = 3                # Number of operational modes (off, half, full)
        K = 50000            # Number of trajectories for Monte Carlo simulation
        L = L                # Number of one-step trajectories in training
        t_start = 0.0f0      # Start time of simulation
        t_end = 0.25f0       # End time of simulation
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
            "experiment" => "carmona"
        )
    end

    # --- 3. Process Definitions ---
    """
        drift(u, p, t)

    Compute the drift term of the stochastic process.
    """
    function drift(u, p, t)
        u1 = u[1] * (5.0f0 * log(50.0f0) - 5.0f0 * log(u[1]))
        u2 = u[2] * (2.0f0 * log(6.0f0) - 2.0f0 * log(u[2]))
        return @SVector [u1, u2]
    end

    """
        dispersion(u, p, t)

    Compute the dispersion (diffusion) term of the stochastic process.
    """
    function dispersion(u, p, t)
        u11 = 0.5f0 * u[1]
        u12 = 0.0f0
        u21 = 0.4f0 * 0.8f0 * u[2]
        u22 = 0.4f0 * 0.6f0 * u[2]
        return @SMatrix [u11 u12; u21 u22]
    end

    """
        jump(u, p, dt)

    Compute the jump term of the stochastic process.
    """
    function jump(u, p, dt)
        deltaN = rand(Binomial(1, 32 * Float64(dt)))
        size = rand(Exponential(0.10))
        jump = exp(size)^deltaN .|> Float32
        jump = (jump-1) * u[1]
        return @SVector [jump, 0.0f0]
    end

    # Initialize random process
    RandomProcess = OptSwitch.JumpProcess(drift, dispersion, jump)

    # --- 4. Payoff and Cost Functions ---
    """
        payoff(x, t)

    Compute the payoff vector for the switching problem.
    Returns a 3-dimensional vector of payoffs for each mode.
    """
    function payoff(x, t)
        j1 = -1.0f0
        j2 = 0.438f0 * (x[1] - 7.5f0 * x[2]) - 1.1f0
        j3 = 0.876f0 * (x[1] - 10.0f0 * x[2]) - 1.2f0
        return @SVector [j1, j2, j3]
    end

    """
        cost(x, t)

    Compute the cost matrix for switching between modes.
    Returns a 3x3 matrix where entry (i,j) represents the cost of switching from mode i to j.
    """
    function cost(x, t)
        c = 0.01f0*x[2] + 0.001f0
        return @SMatrix [0 c c; c 0 c; c c 0.0f0]
    end

    # Initialize payoff model
    payoff_params = (1,)
    payoffmodel = OptSwitch.PayOffModel(payoff_params, payoff, cost)

    # --- 5. Data Generation and Processing ---
    # Load parameters
    new_params = get_parameters(1)
    return RandomProcess,payoff,cost,new_params
end
# Initialize random seed for reproducibility
rng = MersenneTwister()
Random.seed!(rng, 54321)

# Set up the models and get parameters
RandomProcess, payoff, cost, new_params = set_up()
@unpack d, J, K, N, L, t_start, t_end, dt, p = new_params
# Initialize starting state and add small noise
x0 = [50.0f0, 6.0f0]  # Initial state values
x_init = repeat(x0, 1, K) .+ 0.01f0 * randn(d, K)  # Add small Gaussian noise
# Load pre-trained machine learning models
data_dir = datadir("CL/machines")
m = filter(x -> occursin("CL", x) && occursin("50000_", x), 
    readdir(datadir("CL/machines")))
mods = OptSwitch.load_models(data_dir * "/" .* m)

# Filter out unwanted models

filter!(mods) do x
    x.name != "ridge" && x.name != "lasso" && x.name != "deep network" && 
    x.name != "network 2" && x.name != "network 3" && x.name != "small"
end

# Generate trajectories for analysis
# Using the first 5000 trajectories out of K total trajectories
trajs = OptSwitch.generate_paths(RandomProcess, x_init, t_start, N, dt, p)[:,:,1:5000]
sample_paths = trajs[:,1:end,1:100]
times = 1:181

# --- Analysis and Visualization ---
println("Starting analysis and visualization...")

# Calculate benchmarks for comparison
println("Calculating benchmark strategies...")

# A posteriori strategy (optimal with perfect hindsight)
# This computes the value function with full information, serving as the upper bound benchmark
initial_mode = 1
optimal_value = calculate_value_function(sample_paths, cost, payoff, times, dt, J)
optimal_strategies = determine_optimal_strategy(optimal_value, sample_paths, cost, times, dt, J, initial_mode)
println("✓ Calculated a posteriori optimal strategy")

# Greedy strategy (myopic decision-making without lookahead)
# This implements a simpler strategy that maximizes immediate rewards without considering future states
greedy_strategies = calculate_greedy_value_matrix(sample_paths, payoff, cost, times, dt, initial_mode, J)[2]
println("✓ Calculated greedy strategy")


# ---- 2. Model Comparison and Strategy Analysis ----
println("Performing model comparison and strategy analysis...")

# Prepare the comparative analysis of different ML models
# against a posteriori and greedy benchmarks, measuring how closely they approximate optimal decisions
strat_analysis = prepare_strategy_analysis(
    mods, payoff, cost, sample_paths, times, dt, J,
    other_strategies=[(optimal_strategies, "a posteriori"),(greedy_strategies, "greedy")],
    initial_mode=initial_mode
)
println("✓ Completed strategy analysis")

# Generate summary dataframe with strategy rankings based on performance
df_summary = sort(strat_analysis.summary_df, :Rank)

# Extract distance metrics between strategies and create enhanced performance metrics
println("Computing performance metrics...")
dist = strat_analysis.strat_dist

# Create enhanced dataframe with decision similarity and prediction accuracy metrics
joined_df = leftjoin(dist.distances, strat_analysis.summary_df, on = :Strategy)
joined_df = select(joined_df, Not([:Rank,:DifferenceFromOptimal]))
df = joined_df

# Calculate additional performance metrics
df.decision_similiarity = 1 .- df.Decision_Distance_To_Reference  # Higher is better
df.prediction_accuracy = 1 ./ ( 1 .+ df.Prediction_Error)         # Higher is better

# Filter out unwanted strategies and select relevant columns for the paper
df = filter(row -> !(row.Strategy in ["network 2", "network 3"]), df)
df = df[:,[:Strategy, :NormalizedFinalValue,:decision_similiarity,:prediction_accuracy]]

# Prepare dataframe for LaTeX output and save to CSV
df = prepare_for_latex(df)
CSV.write(datadir("CL/CL_strategy_summary.csv"), df)
println("✓ Saved strategy summary to CSV")

# Generate visualizations for research article
println("Generating main visualizations...")
df = strat_analysis.summary_df
res = plot_strategy_analysis(strat_analysis)

# Create plots directory
mkpath(plotsdir("CL"))

# Save mode distribution visualization showing operational states over time
save(plotsdir("CL/CL_switching_strategies.pdf"), res[1]) 
println("✓ Saved switching strategies visualization")

# Save performance comparison of different strategies
save(plotsdir("CL/CL_strategy_performance.pdf"), res[2]) 
println("✓ Saved strategy performance comparison")

# Value prediction error analysis
println("Analyzing value prediction error...")
res = plot_value_prediction_error(mods, sample_paths, times, payoff, cost, dt, J)
save(plotsdir("CL/CL_value_prediction_error.pdf"), res[1])
println("✓ Saved value prediction error analysis")

# --- 3. Decision Boundary Visualization ---
println("Creating decision boundary visualizations...")

# Define time points for boundary visualization (representative sampling of time horizon)
time_points = [1; 35; 70; 105; 150; 175]
grid_points = [40 160; 4 9]  # Min/max ranges for electricity and gas prices

# Create a posteriori model for theoretical optimal decision boundaries
println("Creating a posteriori model for boundary comparison...")
apost_model = create_full_value_grid(
    OptSwitch.generate_paths, RandomProcess, grid_points, time_points,
    N, dt, t_start, p, cost, payoff, J
)
new_model = OptSwitch.LearningModel(apost_model, "a posteriori", "apost")
push!(mods, new_model)
println("✓ Created a posteriori model")

# Create switching boundary evolution plots for different models
println("Generating switching boundary visualizations...")

# Process models by name for boundary evolution plots
model_names = [mod.name for mod in mods]
for model_name in model_names
    # Find the model by name
    model = filter(m -> m.name == model_name, mods)[1]
    println("Processing $(model.name) model...")
    
    num_trajs = 100
    
    plt = plot_switching_boundaries_evolution(
        model, cost, payoff, time_points, trajs[:,time_points,1:num_trajs], dt, J
    )
    
    # Use the model name for the file naming
    safe_name = replace(lowercase(model.name), " " => "_")
    save(plotsdir("CL/CL_switching_boundaries_evolution_$(safe_name).pdf"), plt)
end

# Create side-by-side comparison of all models at a specific time point
println("Creating boundary comparison visualization...")
tp = 70  # Time point for comparison (chosen from middle of time range)
plt = plot_switching_boundaries_comparison(
    mods, cost, payoff, tp, trajs[:,:,1:1000], dt, J, initial_mode
)
save(plotsdir("CL/CL_switching_boundaries_comparison.pdf"), plt)
println("✓ Saved all boundary visualizations")

# --- 4. Finalization ---
println("All visualizations completed successfully!")
