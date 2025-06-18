"""
    Carmona-Ludkovski (CL) Model

This script implements the benchmark Carmona-Ludkovski model for optimal switching.
The model features a 2-dimensional stochastic process with 3 operational modes,
modeling price dynamics with mean reversion and jump processes.

Key characteristics:
- 2D state space with correlated price processes
- 3 operational modes with different payoff profiles
- Mean-reverting dynamics with jump components
- Benchmark problem from Carmona & Ludkovski literature
- Switching costs based on operational mode transitions
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
using Lux, ADTypes, MLUtils, Random, Statistics, Printf
import OptSwitch

# Initialize random seed for reproducibility
rng = MersenneTwister()
Random.seed!(rng, 12345)

"""
    main(; L=1, old_models=[], dir="Carmona", model_types=["knn"])

Run the Carmona-Ludkovski (CL) benchmark experiment.

# Arguments
- `L`: Number of one-step trajectories in training
- `old_models`: Optional list of pre-trained models to continue training
- `dir`: Directory to save experiment results
- `model_types`: List of models to train (e.g., "knn", "network", "forest")

# Example
```julia
# Run CL experiment with multiple model types
main(model_types=["knn", "forest", "network"], dir="CL_comparison")
```
"""
function main(;L=1, old_models=[], dir="Carmona",model_types = ["knn"])

    """
        get_parameters(L)

    Returns a dictionary containing model parameters for the Carmona experiment.
    """
    function get_parameters(L)
        d = 2                # Dimension of random process
        N = 180             # Number of time points
        J = 3               # Different modes
        K = 50000           # Number of trajectories
        L = L               # L one-step trajectories in training
        t_start = 0.0f0     # Time start value
        t_end = 0.25f0      # Time end value
        dt = (t_end - t_start) / N
        p = ()
        
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
            "experiment" => "CL"
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
    new_params = get_parameters(L)
    @unpack d, J, K, N, L, t_start, t_end, dt, p = new_params

    # Initialize starting points
    x0 = [50.0f0, 6.0f0]
    x_init = repeat(x0, 1, K) .+ 0.01f0 * randn(d, K)
    # Run experiments
    OptSwitch.MLJ_main(new_params,RandomProcess,payoffmodel,x_init;dir=dir,
    model_types=model_types)
end
# Run main experiment
main(;L=1,dir="CL",model_types=["knn"])

