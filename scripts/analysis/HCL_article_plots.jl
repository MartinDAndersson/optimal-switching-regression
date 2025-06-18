"""
High-Dimensional-Carmona-Ludkovski (HCL) Experiment

"""

# --- 1. Imports and Initialization ---
using DrWatson               # Project management
@quickactivate               # Activate the current project
push!(LOAD_PATH, srcdir())   # Add source directory to load path
plot_functions_dir = scriptsdir("plot_functions_article")
push!(LOAD_PATH, plot_functions_dir)

# Core packages
using Revise, Pkg, JLD2, FileIO, Random, Distributions, LinearAlgebra
using Parameters, BenchmarkTools, Printf, BSON, TimerOutputs, StaticArrays
using ProgressMeter, DataFrames, MLJ, Glob, MLUtils, CairoMakie
using AlgebraOfGraphics, CairoMakie, DataFramesMeta, Makie
using Lux, ADTypes, MLUtils, Optimisers, Zygote, Random, Statistics, Printf
import SimpleChains: static
import OptSwitch
using CSV, AlgebraOfGraphics
#using Plots
#using StatsPlots


# --- 2. Setup Random Seed and Initial Parameters ---
# Initialize random seed for reproducibilityrng = MersenneTwister()
rng = MersenneTwister()
Random.seed!(rng, 54321)
d=10
# Create a directory to store intermediate results
intermediate_dir = "intermediate_results"
mkpath(intermediate_dir)

# Create an empty DataFrame to store all summaries
all_summaries = DataFrame()
#for d in [2]
#d=10
function setup_carmona_problem(d)
    # Define parameters for the experiment
    function get_parameters(d)
        N = 2 # Time slices
        J = 3 # Different modes
        K = 20000 # Number of trajectories
        N = 180 # Number of time points
        L = 1 # L one-step trajectories in training
        t_start = 0.0f0 # Time start value
        t_end = 0.25f0 |> Float32 # Time end value
        dt = (t_end - t_start) / N
        p = ()
        new_params = Dict("d"=>d,"J"=>J,"N"=>N,"dt"=>dt,"K"=>K,"L"=>L,"t_start"=>t_start,
        "t_end"=>t_end,"p"=>p,"experiment"=>"carmona_dim")
        return new_params
    end

    @unpack d,N,J,K,L,t_start,t_end,dt = get_parameters(d)

    # --- 4. Process Parameters ---
    # Jump process parameters
    lambda_poisson = 32      # Poisson process intensity
    lambda_exp = 10          # Exponential jump size parameter
    kappa = ones(d) .* 2 .|> Float32  # Mean reversion strength
    kappa[1] = 5             # Stronger mean reversion for first component
    lambda_poisson = 32
    lambda_exp = 10
    kappa = ones(d) .* 2 .|> Float32
    kappa[1] = 5

    # Initial state and equilibrium levels

    x0 = ones(d) .* 6
    x0[1] = 50
    x0 = x0 .|> Float32
    mu = log.(x0)
    # Volatility matrix
    sigma = [j == i ? 1.0f0 : 0.0f0 for i in 1:d, j in 1:d] .* 0.24
    sigma[1,:] .= ones(d).*0.32
    sigma[1,1] = 0.5
    sigma = permutedims(sigma)
    sigma = sigma .|> Float32
    # Process parameters
    p = Dict("sigma"=>sigma,"mu"=>mu,"d"=>d,"kappa"=>kappa,
    "lambda_poisson"=>lambda_poisson,"lambda_exp"=>lambda_exp) 

    # --- 5. Process Definition ---
    function dispersion(u,p,t)
        sigma = p["sigma"]
        return sigma
    end

    function drift(u,p,t)
        mu = p["mu"]
        kappa = p["kappa"]
        du = kappa .* u .*  (mu .- log.(u))
        return du
    end

    function jump(u, p,dt)
        d=p["d"]
        deltaN = rand(Binomial(1, lambda_poisson * Float64(dt)))
        size = rand(Exponential(1/lambda_exp))
        jump = exp(size)^deltaN .|> Float32
        du = ones(d)
        du[1] = jump
        du = (du .- 1) .* u
        return du
    end

    RandomProcess = OptSwitch.JumpProcess(drift, dispersion, jump)

    # --- 6. Payoff Model Setup ---
    mat = [0 0.438 0.876; 0 -0.438*7.5 -0.876*10] .|> Float32 |> permutedims
    b = [-1.f0, -1.1f0, -1.2f0]
    payoff_p = Dict("mat"=>mat,"b"=>b,"J"=>J,"d"=>d)

    """
        payoff(x,t,payoff_p)
    
    Computes the payoff vector for each possible mode.
    Uses first component and mean of remaining components.
    """
    function payoff(x,t,payoff_p)
        mat = payoff_p["mat"]
        b = payoff_p["b"]
        J = payoff_p["J"]
        x1 = x[1]
        x2 = mean(x[2:end])
        new_x = [x1,x2]
        res = mat * new_x .+ b
        return SVector{J}(res)
    end

    """
    create_cost_matrix(payoff_p)

    Creates the cost matrix for switching between modes.
    """
    function create_cost_matrix(payoff_p)
        J=payoff_p["J"]
        c = 0.01f0
        C = ones(J,J) .* c .+ 0.001f0 .|> Float32
        C[diagind(C)] .= 0.0f0
        return SMatrix{J,J}(C)
    end

    function cost(x,t,payoff_p)
        C = create_cost_matrix(payoff_p)
        return C
    end
    # Create closures for payoff and cost functions

    function payoff_cost_closure(payoff,cost,payoff_p)
        payoff_c(x,t) = payoff(x,t,payoff_p)
        cost_c(x,t) = cost(x,t,payoff_p)
        return payoff_c,cost_c
    end

    payoff_c,cost_c = payoff_cost_closure(payoff,cost,payoff_p)
    payoffmodel = OptSwitch.PayOffModel(payoff_p,payoff_c,cost_c)

    return RandomProcess, payoffmodel, x0, N, J, K, L, t_start, t_end, dt, p
end
#
# --- 7. Generate Paths and Load Models ---
# Set up the Carmona problem with dimension d
# RandomProcess,payoffmodel,x0,N,J,K,L,t_start,t_end,dt,p = setup_carmona_problem(d)#.+ randn(d,K) .* 0.5f0

# # Initialize starting points and generate trajectories

# x_init = repeat(x0,1,K) #.+ randn(d,K) .* 0.05f0     
# trajs=OptSwitch.generate_paths(RandomProcess,x_init,0.0f0,N,dt,p)#[:,:,1:200]
# sample_paths = trajs[:,1:end,1:100]
# times = 1:181

# # Load plot functions

push!(LOAD_PATH, scriptsdir("plotting"))
using hcl_plots
data_dir = datadir("carmona_dim/machines")


# # --- 8. Load and Filter Models ---
# m = readdir(data_dir)
# m = filter(x -> occursin("d=$d"*"_", x), m)
# if d==2
#     m = filter(x -> occursin("20000", x), m)
# else
#     #m = filter(x -> occursin("20000", x), m)
#     m = filter(x -> occursin("algorithm=pca", x), m)
#     #m = filter(x -> !occursin("algorithm=network 1", x), m)
#     #m = filter(x -> !occursin("algorithm=network 3", x), m)
# end
# for el in m
#     println(el)
# end
# # --- 9. Analyze Performance ---
# # Load models from disk
# mods = OptSwitch.load_models(data_dir * "/" .* m)
# cost_c = payoffmodel.c
# payoff_c = payoffmodel.f
# # Calculate a posteriori value function as benchmark
# optimal_value = calculate_value_function(sample_paths, cost_c, payoff_c, times, dt, J)
# # Calculate greedy strategy (myopic strategy ignoring switching costs)
# greedy_strategies = calculate_greedy_value_matrix(sample_paths, payoff_c, cost_c, times, dt, 1, J)[2]
# # Determine a posteriori strategies
# optimal_strategies = determine_optimal_strategy(optimal_value, sample_paths, cost_c, times, dt, J, 1)
# # --- 10. Strategy Analysis ---
# # Prepare analysis comparing different strategies
# strat_analysis = prepare_strategy_analysis(
#     mods, payoff_c, cost_c, sample_paths, times, dt, J,
#     other_strategies=[(optimal_strategies, "a posteriori"),(greedy_strategies, "greedy")],
#     initial_mode=1
# )
# res = plot_strategy_analysis(strat_analysis)

# dist = strat_analysis.strat_dist

# # --- 11. Save Results ---
# # Join the distances dataframe with summary_df by Strategy
# joined_df = leftjoin(dist.distances, strat_analysis.summary_df, on = :Strategy)
# # Select relevant columns
# joined_df = select(joined_df, Not([:Rank,:DifferenceFromOptimal]))
# # Save results to CSV
# CSV.write(datadir("carmona_dim", "decision_distance_$d.csv"), joined_df)
# println("Joined dataframe dimensions: $(size(joined_df))")
# # If plots/HCL directory not exist, create it
# if !isdir(plotsdir("HCL"))
#     mkpath(plotsdir("HCL"))
# end
# save(plotsdir("carmona_highdim/carmona_dim_strategy_distribution_d=$d.pdf"), dist.figure)


# Save the figures
function prepare_analysis(RandomProcess, payoff_c, cost_c, x0, d, K, N, dt, p, J)
    data_dir = datadir("carmona_dim/machines")

    m = readdir(data_dir)
    m = filter(x -> occursin("d=$d"*"_", x), m)
    if d==2
        m = filter(x -> occursin("20000", x), m)  
    else
        m = filter(x -> occursin("20000", x), m)
        m = filter(x -> !occursin("algorithm=knn", x), m)
        m = filter(x -> !occursin("algorithm=network 1", x), m)
        m = filter(x -> !occursin("algorithm=network 3", x), m)
        #m = filter(x->occursin("network",x),m) 
    end

    mods = OptSwitch.load_models(data_dir * "/" .* m)

    x_init = repeat(x0,1,K) #.+ randn(d,K) .* 0.5f0
    trajs=OptSwitch.generate_paths(RandomProcess,x_init,0.0f0,N,dt,p)#[:,:,1:200]
    sample_paths = trajs[:,1:end,1:1000]
    times = 1:181

    initial_mode = 1
    optimal_value = calculate_value_function(sample_paths, cost_c, payoff_c, times, dt, J)
    greedy_strategies = calculate_greedy_value_matrix(sample_paths, payoff_c, cost_c, times, dt, 1, J)[2]
    optimal_strategies = determine_optimal_strategy(optimal_value, sample_paths, cost_c, times, dt, J, 1)

    strat_analysis = prepare_strategy_analysis(
        mods, payoff_c, cost_c, sample_paths, times, dt, J,
        other_strategies=[(optimal_strategies, "a posteriori"),(greedy_strategies, "greedy")],
        initial_mode=initial_mode
    )
    return strat_analysis
end


d_list = [20,30,40,50]
all_summaries = DataFrame[]  # initialize empty array of dataframes
for d in d_list
    @info "Running experiment for dimension $d"
    RandomProcess, payoffmodel, x0, N, J, K, L, t_start, t_end, dt, p = setup_carmona_problem(d)
    
    # Get full analysis
    strat_analysis = prepare_analysis(RandomProcess, payoffmodel.f, payoffmodel.c, x0, d, K, N, dt, p, J)
    
    # # Get summary dataframe and sort by Rank
    df_summary = sort(strat_analysis.summary_df, :Rank)
    strat_analysis.summary_df
    dist = strat_analysis.strat_dist
    dist.distances
    # Join the distances dataframe with summary_df by Strategy
    joined_df = leftjoin(dist.distances, strat_analysis.summary_df, on = :Strategy)
    # Select all columns except PredictedValue and NonNormalizedFinalValue
    joined_df = select(joined_df, Not([:Rank,:DifferenceFromOptimal]))

    summary_df = joined_df
    

    
    # Plot with filtered data
    res = plot_strategy_analysis(strat_analysis)
    dist = strat_analysis.strat_dist
    dist.figure
    save(plotsdir("carmona_highdim/carmona_dim__strategy_distribution_d=$d.pdf"), dist.figure)
    # Add dimension column
    summary_df[!, :dimension] .= d
    
    # Save individual dataframe
   #CSV.write(datadir("carmona_dim", "carmona_summary_d$(d).csv"), df_summary)
    CSV.write(datadir("carmona_dim", "carmona_summary_d$(d).csv"), summary_df)
    
    push!(all_summaries, df_summary)
    
    save(datadir("carmona_dim/carmona_switching_strategies_$d.pdf"), res[1])
    save(datadir("carmona_dim/carmona_strategy_performance_$d.pdf"), res[2])
end

# --- 15. Final Analysis and Export ---
# Load all dimension results
all_summaries = [CSV.read(datadir("carmona_dim", "carmona_summary_d$(d).csv"),DataFrame) for d in d_list]
# Clean and combine results
all_summaries[1] = filter(row -> row.Strategy != "network 1" && row.Strategy != "network 3", all_summaries[1])
combined_df = vcat(all_summaries...)

# Filter out rows with Strategy "Network 1" or "Network 3"
combined_df = filter(row -> row.Strategy != "Network 1" && row.Strategy != "Network 3", combined_df)
combined_df = vcat(all_summaries...)
# Save the combined dataframe
CSV.write(datadir("carmona_dim", "carmona_combined_summary.csv"), combined_df)

# Process the combined dataframe to make it LaTeX-friendly


# Process the combined dataframe
latex_df = prepare_for_latex(combined_df)

# Save as CSV (easily importable to LaTeX via csvsimple or similar)
CSV.write(datadir("carmona_dim", "carmona_combined_summary_latex.csv"), latex_df)


println("Saved LaTeX-friendly files to carmona_dim directory")

df = CSV.read(datadir("carmona_dim", "carmona_combined_summary.csv"), DataFrame)
using AlgebraOfGraphics
# Rename "pca knn" to "knn" for rows with dimension != 2
df[df.dimension .!= 2 .&& df.Strategy .== "pca knn", :Strategy] .= "knn"
df.decision_similiarity = 1 .- df.Decision_Distance_To_Reference
df.prediction_accuracy = 1 ./ (1 .+ df.Prediction_Error)
#df=filter(row -> !(row.Strategy in ["lasso", "ridge","network 1"]), df)
# Filter out strategies named "lasso" or "ridge"
#df = filter(row -> !(row.Strategy in ["lasso", "ridge","network 1"]), df)
prepare_for_latex(df)
df=df[:,[:dimension, :Strategy, :NormalizedFinalValue,:decision_similiarity,:prediction_accuracy]]
CSV.write(datadir("carmona_dim", "carmona_summary_latex.csv"), prepare_for_latex(df))
