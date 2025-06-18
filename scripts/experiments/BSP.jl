"""
    Banded Shift Process (BSP) Model

This script implements the BSP model for optimal switching problems.
The model represents a 1-dimensional state space with 10 operational modes,
featuring a simple mean-reverting process with state-dependent volatility
and a grid-like payoff structure.

Key characteristics:
- 1D state space with mean reversion
- 10 operational modes with different optimal regions
- State-dependent volatility structure
- Grid-like payoff pattern testing mode-switching boundaries
"""

# --- 1. Imports and Initialization ---
using DrWatson
@quickactivate
push!(LOAD_PATH, srcdir())
current_dir = scriptsdir("experiments")
push!(LOAD_PATH, current_dir)

using Revise, Pkg, JLD2, FileIO, Random, Distributions, LinearAlgebra
using Parameters, BenchmarkTools, Printf, BSON, TimerOutputs, StaticArrays
using ProgressMeter, Glob, MLUtils
using Lux, ADTypes, MLUtils, Optimisers, Zygote, Random, Statistics, Printf
import SimpleChains: static
import OptSwitch

# Initialize random seed for reproducibility
rng = MersenneTwister()
Random.seed!(rng, 12345)

"""
    main(; L=1, old_models=[], dir="BSP", model_types=["knn"])

Run the Banded Shift Process (BSP) experiment.

# Arguments
- `L`: Number of one-step trajectories in training
- `old_models`: Optional list of pre-trained models to continue training
- `dir`: Directory to save experiment results
- `model_types`: List of models to train (e.g., "knn", "network", "forest")

# Example
```julia
# Run BSP experiment with neural networks
main(model_types=["network"], dir="BSP_neural")
```
"""
function main(;L=1, old_models=[], dir="BSP", model_types = ["knn"])
    
    """
        get_parameters()

    Returns a dictionary containing model parameters for the BSP experiment.
    """
    function get_parameters()
        d = 1                # Dimension of random process (1D state space)
        J = 10               # Number of operational modes
        K = 20000            # Number of trajectories for Monte Carlo simulation
        N = 36               # Number of time points in simulation
        L = L                # Number of one-step trajectories in training
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
    # Load parameters
    new_params = get_parameters()
    @unpack d, J, N, dt, K, L, t_start, t_end, p = new_params

    # Initialize starting points
    x_init = randn(Float32,(d,K)) .* 0.5f0  # Initial state values with noise
    
    # Run experiments
    OptSwitch.MLJ_main(new_params, RandomProcess, payoffmodel, x_init; dir=dir,
    model_types=model_types)
end

# Run main experiment
main(;L=1, dir="BSP", model_types=["knn"])