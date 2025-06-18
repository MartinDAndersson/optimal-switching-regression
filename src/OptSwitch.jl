"""
    OptSwitch

A Julia framework for solving optimal switching problems using regression-based methods.

This module implements the Longstaff-Schwartz algorithm with various machine learning
regression methods to approximate value functions in dynamic switching problems.
It is particularly suited for applications in energy markets, financial decision-making,
and other domains requiring optimal mode selection over time.

# Key Components
- Stochastic process simulation (SDE and jump-diffusion processes)
- Multiple regression methods (KNN, neural networks, forests, linear models)
- Backward induction algorithm for value function approximation
- Visualization and performance analysis tools

# Main Functions
- `MLJ_main`: Primary entry point for running experiments
- `generate_paths`: Simulate trajectories of the stochastic process
- `profit_and_cost`: Calculate payoffs and switching costs
- `calculate_Y`: Compute regression targets using backward induction

# Example
```julia
# Define parameters
params = parameters(d=2, N=100, dt=0.01, J=3, K=10000, L=1, t_start=0.0, p=())

# Create stochastic process and payoff model
process = JumpProcess(drift, dispersion, jump)
payoff_model = PayOffModel(p, payoff_func, cost_func)

# Run experiment
MLJ_main(params, process, payoff_model, initial_state; model_types=["knn", "forest"])
```
"""
module OptSwitch

using Distributions, LinearAlgebra, Parameters, .Threads,
    Printf, TimerOutputs, StaticArrays
using StatsBase: mean
using Lux
using Optimisers
using Zygote
using MLUtils: DataLoader
using ProgressMeter
using DataFrames
using DrWatson
push!(LOAD_PATH, srcdir())
using CSV
using NearestNeighborModels
using MLJ
using JLD2
using Lux, ADTypes, MLUtils, Optimisers, Zygote, Random, Statistics, Printf
using TrainingLogger
using TrainingPlots
using LuxCUDA
using Makie
using StaticArrays
using CairoMakie
using DataFramesMeta: @transform, @select, @rename, @subset
import SimpleChains: static

# Load MLJ models that will be used for regression

KNN = MLJ.@load KNNRegressor
RandomForestRegressor = MLJ.@load RandomForestRegressor pkg=DecisionTree
EvoTreeReg = MLJ.@load EvoTreeRegressor

LinearRegressor = MLJ.@load LinearRegressor pkg = MLJLinearModels
RidgeRegressor = MLJ.@load RidgeRegressor pkg = MLJLinearModels
LassoRegressor = MLJ.@load LassoRegressor pkg = MLJLinearModels
PCA = MLJ.@load PCA pkg = MultivariateStats
InteractionTransformer = MLJ.@load InteractionTransformer pkg=MLJModels
LGBMReg = MLJ.@load LGBMRegressor pkg=LightGBM
# Export plot functions
export plot_3d

# Include functions Lux network training
include("Lux.jl")


"""
    profit_and_cost(X_prev, payoffmodel, n, dt, K, J)

Calculate profit matrix and switching cost tensor for all trajectories and modes at a given time step.

# Arguments
- `X_prev::Matrix`: State matrix of size (d, K) where d is dimension and K is number of trajectories
- `payoffmodel::PayOffModel`: Model containing payoff function f and cost function c
- `n::Int`: Current time step index
- `dt::Float32`: Time step size
- `K::Int`: Number of trajectories
- `J::Int`: Number of operational modes

# Returns
- `f_res::Matrix{Float}`: Profit matrix of size (K, J) - payoff for each trajectory in each mode
- `C::Array{Float,3}`: Cost tensor of size (K, J, J) - switching costs between modes for each trajectory

# Details
The payoff function is evaluated at time (n-1)*dt and multiplied by dt to get the period payoff.
The cost function c(x,t) returns a J×J matrix of switching costs for state x at time t.
"""
function profit_and_cost(X_prev, payoffmodel, n, dt, K, J)
    time = (n - 1) * Float64(dt)
    C = zeros(K, J, J)
    c_closure = x -> payoffmodel.c(x, time)
    for i = 1:K
        C[i, :, :] .= c_closure(X_prev[:, i])
    end
    f_closure = x -> payoffmodel.f(x, time)
    f_res = mapslices(f_closure, X_prev, dims=1) .* dt # size=(J,K)
    return f_res', C
end
# Include model abstractions for different types of learning models
include("model_abstractions.jl")


"""
    parameters

Configuration struct for optimal switching experiments.

# Fields
- `d::Int64`: Dimension of the underlying stochastic process
- `N::Int64`: Number of time steps in the discretization
- `dt::Float32`: Size of each time step
- `J::Int64`: Number of different operational modes/states
- `K::Int64`: Number of Monte Carlo trajectories for training
- `L::Int64`: Number of one-step trajectories for value estimation (typically 1)
- `t_start::Float32`: Starting time of the simulation
- `t_end::Float32`: End time of the simulation (automatically computed as t_start + N*dt)
- `p::Tuple`: Additional parameters for the stochastic process (e.g., drift, volatility)

# Example
```julia
params = parameters(
    d = 2,        # 2D process
    N = 100,      # 100 time steps
    dt = 0.01f0,  # time step of 0.01
    J = 3,        # 3 operational modes
    K = 10000,    # 10,000 trajectories
    L = 1,        # 1 one-step trajectory
    t_start = 0f0,
    p = (κ=[5.0, 2.0], σ=[0.5, 0.3])
)
```
"""
@with_kw struct parameters
    d::Int64     # Dimension of underlying process
    N::Int64    # Number of time steps
    dt::Float32 # Discretization step
    J::Int64     # Number of different states
    K::Int64   # Number of trajectories
    L::Int64   # Number of one-step trajectories
    t_start::Float32
    t_end::Float32 = float(N * dt) + t_start
    p::Tuple
end

"""
    PayOffModel{ptype, FuncType1<:Function, FuncType2<:Function}

Model for payoff functions and switching costs in the optimal switching problem.

# Fields
- `p::ptype`: Parameters specific to the payoff model (e.g., fuel prices, capacities)
- `f::FuncType1`: Payoff function mapping (state, time) to a J-vector of profits for each mode
- `c::FuncType2`: Cost function mapping (state, time) to a J×J matrix of switching costs

# Function Signatures
- `f(x::Vector, t::Real) → Vector{Float}`: Returns payoff for each mode at state x and time t
- `c(x::Vector, t::Real) → Matrix{Float}`: Returns cost matrix where c[i,j] is the cost of switching from mode i to j

# Example
```julia
# Define payoff function for 3 modes
function payoff(x, t)
    return [
        -1.0,                          # Mode 1: Off
        0.438*(x[1] - 7.5*x[2]) - 1.1, # Mode 2: Half capacity
        0.876*(x[1] - 10*x[2]) - 1.2   # Mode 3: Full capacity
    ]
end

# Define switching cost function
function cost(x, t)
    base_cost = 0.01*x[2] + 0.001
    C = zeros(3, 3)
    for i in 1:3, j in 1:3
        C[i,j] = (i == j) ? 0.0 : base_cost
    end
    return C
end

payoff_model = PayOffModel(p=nothing, f=payoff, c=cost)
```
"""
@with_kw struct PayOffModel{ptype,FuncType1<:Function,
    FuncType2<:Function}
    p::ptype
    f::FuncType1 # f:X -> J-vector, payoff in mode 1...J
    c::FuncType2 # c:X -> JxJ matrix, cost matrix for J modes. 
end



# Include data generation functionality

include("data_generation.jl")


"""
    create_model_dim(d, architecture_type="wide_shallow")

Create neural network architectures for higher-dimensional problems (d ≥ 5).

# Arguments
- `d::Int`: Input dimension
- `architecture_type::String`: Network type ("wide_shallow", "two_layer", or "pyramid")

# Returns
- Lux Chain model with appropriate architecture for the given dimension

# Architecture Types
- `"wide_shallow"`: Single hidden layer with 128 neurons
- `"two_layer"`: Two hidden layers (128 → 64 neurons)
- `"pyramid"`: Three hidden layers (128 → 64 → 32 neurons)
"""
function create_model_dim(d, architecture_type::String="wide_shallow")
    if architecture_type == "wide_shallow"
        # Wide and shallow with single hidden layer
        return Chain(
            Dense(d, 128, relu),
            Dropout(0.1),
            Dense(128, 1)
        )
    elseif architecture_type == "two_layer"
        # Two layer network with decreasing width
        return Chain(
            Dense(d, 128, relu),
            Dropout(0.1),
            Dense(128, 64, relu),
            Dropout(0.1),
            Dense(64, 1)
        )
    else
        # Pyramid structure with decreasing width
        return Chain(
            Dense(d, 128, relu),
            Dropout(0.1),
            Dense(128, 64, relu),
            Dropout(0.1),
            Dense(64, 32, relu),
            Dropout(0.1),
            Dense(32, 1)
        )
    end
end

"""
    create_model_lowdim(d, architecture_type="wide_shallow")

Create neural network architectures optimized for lower-dimensional problems (d < 5).

# Arguments
- `d::Int`: Input dimension
- `architecture_type::String`: Network type ("wide_shallow", "two_layer", "pyramid", or "deeper")

# Returns
- Lux Chain model with architecture suitable for low-dimensional inputs

# Architecture Types
- `"wide_shallow"`: Single hidden layer with 32 neurons
- `"two_layer"`: Multiple layers with 16 neurons each
- `"pyramid"`: Decreasing layers (32 → 16 → 8 neurons)
- `"deeper"`: Four layers with more depth (32 → 32 → 32 → 16 neurons)
"""
function create_model_lowdim(d, architecture_type::String="wide_shallow")
    if architecture_type == "wide_shallow"
        # Wide and shallow with single hidden layer
        return Chain(
            Dense(d, 32, relu),
            Dropout(0.1),
            Dense(32, 1)
        )
    elseif  architecture_type ==  "two_layer"
        # Two layer network with decreasing width
        return Chain(
            Dense(d, 16, relu),
            Dropout(0.1),
            Dense(16, 16, relu),
            Dropout(0.1),
            Dense(16,16,relu),
            Dense(16, 1)
        )
    elseif architecture_type == "pyramid"
        # Pyramid structure with decreasing width
        return Chain(
            Dense(d, 32, relu),
            Dropout(0.1),
            Dense(32, 16, relu),
            Dropout(0.1),
            Dense(16, 8, relu),
            Dropout(0.1),
            Dense(8, 1)
        )
    elseif architecture_type == "deeper"
        # Deeper network with more layers
        return Chain(
            Dense(d, 32, relu),
            Dropout(0.1),
            Dense(32, 32, relu),
            Dropout(0.1),
            Dense(32, 32, relu),
            Dropout(0.1),
            Dense(32, 16, relu),
            Dropout(0.1),
            Dense(16, 1)
        )
    end
end

"""
    create_hybrid_model(d)

Create a hybrid ensemble model combining linear regression and k-nearest neighbors.

# Arguments
- `d::Int`: Input dimension (currently not used, kept for API consistency)

# Returns
- MLJ Stack ensemble combining linear and KNN models with a linear metalearner

# Details
Creates an ensemble that combines:
- Linear regression component to capture global trends
- KNN component with PCA preprocessing to capture local structure
- Linear metalearner to combine predictions optimally
"""
function create_hybrid_model(d)
    # Create the linear part to capture global trends
    linear_model = Pipeline(
        standardizer = Standardizer(),
        model = LinearRegressor()
    )
    
    # Create the KNN part to capture local structure
    knn_model = Pipeline(
        standardizer = Standardizer(),
        pca = PCA(maxoutdim=6),
        model = KNN(K=10)
    )
    
    # Create ensemble that combines predictions
    ensemble = Stack(        
        resampling = Holdout(fraction_train=0.7),  # Simpler than CV
        cache = false,  # Critical: disable caching
        acceleration = CPUThreads(),  # Proper thread management
        metalearner = LinearRegressor(),
        linear = linear_model,
        knn = knn_model
    )
    
    return ensemble
end

"""
    create_models_by_type(d, N, J; model_types=["pca_knn"])

Create machine learning models based on specified types for the regression step.

# Arguments
- `d::Int`: Dimension of the state space
- `N::Int`: Number of time steps
- `J::Int`: Number of operational modes

# Keyword Arguments
- `model_types::Vector{String}`: List of model types to create. See MLJ_main for available options.

# Returns
- `Vector{LearningModel}`: Array of configured learning models ready for training

# Example
```julia
models = create_models_by_type(2, 100, 3; model_types=["knn", "forest", "neural"])
```
"""
function create_models_by_type(d, N, J; model_types=["pca_knn"])
    selected_models = []
    
    for model_type in model_types
        if model_type == "all"
            # Get all models
            return get_models(d, N, J)
        elseif model_type == "pca_knn"
            # Just the PCA+KNN model
            pca = PCA(maxoutdim=6)
            knn = Standardizer() |> KNN(K=10)
            pca_knn = Standardizer() |> pca |> knn
            push!(selected_models, model_to_learningmodel(pca_knn, "pca knn", "MLJ", N, J))
        elseif model_type == "knn"
            # Just KNN
            knn = Standardizer() |> KNN(K=10)
            push!(selected_models, model_to_learningmodel(knn, "knn", "MLJ", N, J))
        elseif model_type == "weighted_knn"
            # Weighted KNN
            pca = PCA(maxoutdim=6)
            weighted_knn = Standardizer() |> pca |> KNN(K=10, weights=adaptive_gaussian)
            push!(selected_models, model_to_learningmodel(weighted_knn, "weighted knn", "MLJ", N, J))
        elseif model_type == "neural" || model_type == "network"
            # Create neural network models
            if d < 5
                model1 = create_model_lowdim(d, "wide_shallow")
                model2 = create_model_lowdim(d, "two_layer")
            else
                model1 = create_model_dim(d, "wide_shallow")
                model2 = create_model_dim(d, "two_layer")
            end
            
            push!(selected_models, model_to_learningmodel(model1, "network 1", "sc", N, J))
            push!(selected_models, model_to_learningmodel(model2, "network 2", "sc", N, J))
        elseif model_type == "forest"
            # Create random forest model
            forest = RandomForestRegressor(n_trees = 25, max_depth=3)
            push!(selected_models, model_to_learningmodel(forest, "forest", "MLJ", N, J))
        elseif model_type == "ridge"
            # Create ridge regression model
            ridge = Standardizer() |> RidgeRegressor(lambda=0.1)
            push!(selected_models, model_to_learningmodel(ridge, "ridge", "MLJ", N, J))
        elseif model_type == "linear"
            # Linear regression with interactions
            if d < 5
                linear = Standardizer() |> InteractionTransformer(order=6) |> LinearRegressor()
            else
                linear = Standardizer() |> LinearRegressor()
            end
            push!(selected_models, model_to_learningmodel(linear, "linear", "MLJ", N, J))
        elseif model_type == "lasso"
            # Create lasso regression model
            lasso = Standardizer() |> LassoRegressor(lambda=0.1)
            push!(selected_models, model_to_learningmodel(lasso, "lasso", "MLJ", N, J))
        elseif model_type == "lgbm"
            # Create LightGBM model
            lgbm = LGBMReg(
                num_iterations = 200,
                learning_rate = 0.05,
                num_leaves = 31,
                min_data_in_leaf = 100,
                lambda_l2 = 0.5,
                bagging_fraction = 0.8,
                bagging_freq = 1,
                early_stopping_round = 10,
                num_threads = Sys.CPU_THREADS
            )
            push!(selected_models, model_to_learningmodel(lgbm, "lgbm", "MLJ", N, J))
        elseif model_type == "hybrid"
            # Create hybrid model
            hybrid_model = create_hybrid_model(d)
            push!(selected_models, model_to_learningmodel(hybrid_model, "hybrid", "MLJ", N, J))
        else
            @warn "Unknown model type: $model_type"
        end
    end
    
    if isempty(selected_models)
        @warn "No models created for types: $model_types"
        @info "Available types: all, pca_knn, knn, weighted_knn, neural/network, forest, ridge, linear, lasso, lgbm, hybrid"
    end
    
    return selected_models
end

function get_models(d,N,J)
    if d > 5
        model1 = create_model_lowdim(d,"wide_shallow")
        model2 = create_model_lowdim(d,"two_layer")
    else
        model1 = create_model_dim(d,"wide_shallow")
        model2 = create_model_dim(d,"two_layer")
    end
    lm_network_1 = OptSwitch.model_to_learningmodel(model1,"network 1","sc",N,J)
    lm_network_2 = OptSwitch.model_to_learningmodel(model2,"network 2","sc",N,J)

    pca = PCA(maxoutdim=6)
    knn = Standardizer() |> KNN(K=10)
    lgbm = LGBMReg(
        num_iterations       = 200,
        learning_rate        = 0.05,
        num_leaves           = 31,
        min_data_in_leaf     = 100,
        lambda_l2            = 0.5,
        bagging_fraction     = 0.8,
        bagging_freq         = 1,
        early_stopping_round = 10,  # stop if no improvement in 10 rounds
        num_threads          = Sys.CPU_THREADS
    )
    lm_lgbm=OptSwitch.model_to_learningmodel(lgbm,"lgbm","MLJ",N,J)
    pca_knn = Standardizer() |> pca |> knn
    if d < 10
        lm_knn = OptSwitch.model_to_learningmodel(knn,"knn","MLJ",N,J)
    else
        lm_knn = OptSwitch.model_to_learningmodel(pca_knn,"pca knn","MLJ",N,J)
    end
    forest= RandomForestRegressor(n_trees = 25,max_depth=3)
    lm_forest=OptSwitch.model_to_learningmodel(forest,"forest","MLJ",N,J)
    #evotree = EvoTreeReg()
    #lm_evotree = OptSwitch.model_to_learningmodel(evotree,"evotree","MLJ",N,J)
    ridge = Standardizer() |> RidgeRegressor(lambda=0.1)
    lm_ridge = OptSwitch.model_to_learningmodel(ridge,"ridge","MLJ",N,J)
    lasso = Standardizer() |> LassoRegressor(lambda=0.1)
    lm_lasso = OptSwitch.model_to_learningmodel(lasso,"lasso","MLJ",N,J)
    if d < 5 
        linear =  Standardizer() |>  InteractionTransformer(order=6) |>  LinearRegressor() #InteractionTransformer(order=6) |>
    else
        linear =  Standardizer() |>  LinearRegressor() #InteractionTransformer(order=6) |>
    end
    lm_linear = OptSwitch.model_to_learningmodel(linear,"linear","MLJ",N,J)
    return [lm_network_1, lm_network_2, lm_knn, lm_forest, lm_ridge, lm_lasso, lm_linear,lm_lgbm]
end


"""
    MLJ_main(exp_params, RandomProcess, PayOffModel, v0; old_models=[], dir="default", model_types=["forest"], verbose=false, save_results=true)

Main training function for optimal switching problems using various machine learning models.

# Arguments
- `exp_params`: Experiment parameters struct containing d, N, J, K, etc.
- `RandomProcess`: Stochastic process definition (drift, dispersion, jump)
- `PayOffModel`: Payoff structure with reward and cost functions
- `v0`: Initial state values for simulation
- `old_models`: Previously trained models to reuse (default: [])
- `dir`: Directory name for saving results (default: "default")
- `model_types`: List of model types to train (default: ["forest"])
- `verbose`: Enable detailed logging (default: false)
- `save_results`: Whether to save training results (default: true)

# Returns
- Nothing (results are saved to disk)
"""
function MLJ_main(exp_params, RandomProcess, PayOffModel, v0; old_models=[], dir="default",model_types=["forest"], verbose=false, save_results=true)
    @unpack dt, d, N, L, J, K, t_start, p = exp_params
    
    # Log experiment start
    start_time = now()
    println("\n=== Starting Experiment in Directory: $dir ===")
    println("Date: $(Dates.format(start_time, "yyyy-mm-dd HH:MM:SS"))")
    println("Dimensions: $d, Time steps: $N, Modes: $J, Samples: $K")
    
    # Generate all trajectories
    println("Generating $(K) simulation paths...")
    trajs = generate_paths(RandomProcess, v0, t_start, N, dt, p)
    
    # Get models
    if isempty(old_models)
        println("Creating new models...")
        model_list = create_models_by_type(d, N, J; model_types=model_types)  
    else
        println("Using $(length(old_models)) existing models...")
        model_list = old_models
    end
    
    # Create directories
    plots_dir = datadir(dir, "plots")
    logs_dir = datadir(dir, "logs")
    mkpath(plots_dir)
    mkpath(logs_dir)
    
    # Save experiment configuration
    config_file = joinpath(logs_dir, "experiment_config.txt")
    open(config_file, "w") do io
        println(io, "=== Experiment Configuration ===")
        println(io, "Directory: $dir")
        println(io, "Start time: $(Dates.format(start_time, "yyyy-mm-dd HH:MM:SS"))")
        println(io, "Dimensions: $d")
        println(io, "Time steps: $N")
        println(io, "Modes: $J")
        println(io, "Samples: $K")
        println(io, "Models: $(join([m.name for m in model_list], ", "))")
        println(io, "Parameters: $exp_params")
    end
    
    # Main training loop
    @showprogress for n in N:-1:1
        time = t_start + (n-1) * dt  # adjust to start time 0
        X_prev = @view trajs[:, n, :]
        X_next = one_step(repeat(X_prev, inner=(1, L)), time, dt, RandomProcess, p)
        
        # Train all models at this time step
        Threads.@threads for learning_model in model_list
            MLJ_main_loop!(learning_model, X_prev, X_next, n, exp_params, PayOffModel, dir, save_results)
            # Force garbage collection after each model to free memory
            #GC.gc(false)
            # Force terminal flush for immediate feedback
            flush(stdout)
        end
        
        # Periodically free memory
        if n % 10 == 0 || n == N || n == 1
            GC.gc()  # Force garbage collection to free memory
        end
    end
    
    # Final reporting and visualization
    println("\n=== Generating Final Reports ===")
    
    # Generate comprehensive reports for each model
    for (model_name, logger) in model_loggers
        # Export all metrics to CSV
        export_to_csv(logger, :all)
        
        # Generate summary plots
        plot_path = joinpath(plots_dir, "training_history_$(model_name).pdf")
        plot_training_history(logger, plot_path)
        
        # Generate detailed reports if plotting module is available
        if isdefined(Main, :TrainingPlots)
            report_dir = joinpath(plots_dir, model_name)
            mkpath(report_dir)
            Main.TrainingPlots.generate_training_report(logger, report_dir)
        end
        
        # Save model state
        save_model_state(logger)
    end
    
    # Save trained models
    if save_results
        save_models(model_list, exp_params, dir)
    end
    
    # Print final summaries
    end_time = now()
    total_duration = Dates.value(end_time - start_time) / 1000.0  # in seconds
    
    println("\n=== Training Complete ===")
    println("Total duration: $(round(total_duration/60, digits=2)) minutes")
    
    for (_, logger) in model_loggers
        print_summary(logger)
    end
    
    # Update experiment configuration with completion time
    open(config_file, "a") do io
        println(io, "\n=== Experiment Completion ===")
        println(io, "End time: $(Dates.format(end_time, "yyyy-mm-dd HH:MM:SS"))")
        println(io, "Total duration: $(round(total_duration/60, digits=2)) minutes")
    end
    
    return nothing
end


function split_data(X, Y; ratio=0.9)
    train, test = splitobs((X, permutedims(Y)), at=ratio)
    X_train, Y_train = collect.(train)
    X_test, Y_test = collect.(test)
    return X_train, X_test, permutedims(Y_train), permutedims(Y_test)
end

function save_models(model_list, exp_params,dir)
    for learning_model in model_list
        machind = merge(exp_params, Dict(
            "algorithm" => learning_model.name,
            "pkg" => learning_model.pkg
        ))
        
        name = savename(machind, "jld2")
        dir_name = datadir(dir,"machines", name)
        JLD2.save(dir_name, Dict("learning_model" => learning_model))
    end
end


function save_results(results,dir)
    max_retries = 3
    for attempt in 1:max_retries
        try
            wsave(datadir(dir,"dataframes", savename(results, "jld2")), results)
            return  # Successful save, exit the function
        catch e
            if attempt == max_retries
                @error "Failed to save results after $max_retries attempts" exception=(e, catch_backtrace())
                rethrow(e)
            else
                @warn "Failed to save results, retrying (attempt $attempt of $max_retries)"
                sleep(1)  # Wait a bit before retrying
            end
        end
    end
end


# Dictionary to store loggers for each model
const model_loggers = Dict{String, ModelLogger}()

function MLJ_main_loop!(learning_model, X_prev, X_next, n, exp_params, PayOffModel, dir, save_results=true)
    @unpack J, N = exp_params
    
    # Initialize or get logger for this model
    if !haskey(model_loggers, learning_model.name)
        # Create a more detailed logger with metadata
        metadata = Dict(
            "model_type" => learning_model.pkg,
            "dimensions" => size(X_prev, 1),
            "samples" => size(X_prev, 2),
            "experiment_dir" => dir
        )
        
        # Create log directory
        log_base_dir = joinpath(datadir(dir), "logs")
        
        model_loggers[learning_model.name] = ModelLogger(
            learning_model.name, 
            N, 
            J, 
            log_dir=log_base_dir,
            metadata=metadata
        )
    end
    logger = model_loggers[learning_model.name]
    
    Y = calculate_Y(X_prev, X_next, learning_model, exp_params, PayOffModel, n)
    X_train, X_test, Y_train, Y_test = split_data(X_prev, Y)
    
    Threads.@threads for j in 1:J
        # Perform model fitting
        my_fit!(learning_model, X_train, Y_train[:, j], n, j, 
                payoffmodel=PayOffModel, 
                exp_params=exp_params,
                use_switching_constraint=false)
        
        # Calculate losses
        train_loss = my_loss(learning_model, X_train, Y_train[:, j], n, j)
        test_loss = my_loss(learning_model, X_test, Y_test[:, j], n, j)
        
        # Log the losses with our logger
        log_losses!(logger, n, j, train_loss, test_loss)
        
        # Save results to the dataframes directory
        results = Dict(
            "j" => j,
            "testloss" => test_loss,
            "trainloss" => train_loss,
            "algorithm" => learning_model.name,
            "n" => n
        )
        
        if save_results
            save_results(merge(exp_params, results), dir)
        end
        GC.gc(true)  # Force GC after each batch

    end

    
    # Generate comprehensive reports at key stages
    if n == 1 || n == N || n == div(N, 2)
        # Print summary to console
        print_summary(logger)
        
        # Export data for further analysis
        export_to_csv(logger)
        
        # Generate visual report if plotting module is available
        if isdefined(Main, :TrainingPlots)
            plots_dir = joinpath(datadir(dir), "plots", learning_model.name)
            mkpath(plots_dir)
            Main.TrainingPlots.plot_all_metrics(logger, plots_dir)
        end
    end
end



function values(learning_model, X_next, n, exp_params)
    @unpack K, J, N, L = exp_params
    
    n == N && return zeros(Float32, K, J)
    
    preds = [learning_model(X_next, n + 1, j) for j in 1:J]
    return mean(reshape(hcat(preds...), (L, K, J)), dims=1)[1, :, :]
end

"""
    calculate_Y(X_prev, X_next, learning_model, exp_params, payoffmodel, n)

Compute regression targets using backward induction for optimal switching.

# Arguments
- `X_prev::Matrix`: Current states at time step n, size (d, K)
- `X_next::Matrix`: Next states at time step n+1, size (d, K*L)
- `learning_model::LearningModel`: Model for estimating continuation values
- `exp_params::parameters`: Experiment parameters
- `payoffmodel::PayOffModel`: Payoff and cost model
- `n::Int`: Current time step

# Returns
- `Y::Matrix{Float}`: Regression targets of size (K, J) where Y[k,i] is the value of being in mode i for trajectory k

# Details
Implements backward induction for the switching problem:
```
Y[k,i] = max_j { f_i(X[k],t)*dt + E[V(X_{t+1},j)] - c_{i,j}(X[k],t) }
```
where:
- f_i is the payoff in mode i
- V(X_{t+1},j) is the continuation value in mode j
- c_{i,j} is the cost of switching from mode i to j

This creates the regression targets for approximating the value function at time n.
"""
function calculate_Y(X_prev, X_next, learning_model, exp_params, payoffmodel, n)
    @unpack dt, J, K = exp_params
    
    f_res, C = profit_and_cost(X_prev, payoffmodel, n, dt, K, J)
    future_value = values(learning_model, X_next, n, exp_params)
    
    return [maximum(f_res[k, :] .+ future_value[k, :] .- C[k, i, :]) for k in 1:K, i in 1:J]
end

function model_to_learningmodel(model, name, pkg, N, J)
    #list_of_models = LearningModel[]
    if pkg == "sklearn"
        model_array = OptSwitch.arrayofmodels(model, N + 1, J)
        return learningmodel = LearningModel(model_array, name, pkg)
    elseif pkg == "sc"
        d=model.layers[1].in_dims
        sc_array = OptSwitch.arrayoflux(model,rng, N + 1, J,d;sc=true)
        return learningmodel = LearningModel(sc_array,name, pkg)
    elseif pkg == "lux"
        d=model.layers[1].in_dims
        lux_array = OptSwitch.arrayoflux(model, rng,N + 1, J,d;sc=false)
        return learningmodel = LearningModel(lux_array,name, pkg)
    elseif pkg == "MLJ"
        MLJ_array = OptSwitch.arrayofMLJ(model,N+1,J)
        return learningmodel = LearningModel((model,MLJ_array),name,pkg)
    end
end

end
