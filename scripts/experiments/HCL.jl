"""
    High-Dimensional Carmona-Ludkovski (HCL) Model

This script implements the high-dimensional extension of the Carmona-Ludkovski 
optimal switching model. It defines the stochastic process, payoff functions,
and cost matrices needed to train machine learning models for optimal switching.

The HCL model extends the 2D CL model to arbitrary dimensions while maintaining
the same core structure for the price process.
"""

# --- 1. Imports and Initialization ---
using DrWatson
@quickactivate
push!(LOAD_PATH, srcdir())
current_dir = scriptsdir("experiments")
push!(LOAD_PATH, current_dir)

using Revise, Pkg, JLD2, FileIO, Random, Distributions, LinearAlgebra
using Parameters, BenchmarkTools, Printf, BSON, TimerOutputs, StaticArrays
using ProgressMeter, DataFrames, MLJ, Glob, MLUtils, CairoMakie
using AlgebraOfGraphics, Makie, DataFramesMeta
using Lux, ADTypes, MLUtils, Optimisers, Zygote, Random, Statistics, Printf
import SimpleChains: static
import OptSwitch

# Initialize random seed for reproducibility
rng = MersenneTwister()
Random.seed!(rng, 12345)

"""
    main(d, L; old_models=[], dir="default", model_types=["knn"])

Main function for running the HCL experiment.

# Arguments
- `d`: Dimension of the random process
- `L`: Number of one-step trajectories in training
- `old_models`: Optional list of pre-trained models to include
- `dir`: Directory to save results
- `model_types`: Types of machine learning models to train (e.g., "knn", "pca_knn", "forest")
"""
function main(d, L; old_models=[], dir="default", model_types=["forest"])
    


    """
        get_parameters(d, L)
    
    Configure parameters for the HCL model experiment.
    
    # Arguments
    - `d`: Number of dimensions
    - `L`: Number of one-step trajectories in training
    
    # Returns
    - Dictionary containing all model parameters
    """
    function get_parameters(d, L)
        # Model structure parameters
        d = d                # Dimension of random process
        J = 3                # Number of operational modes
        
        # Simulation parameters
        K = 20000            # Number of trajectories for Monte Carlo simulation
        N = 180              # Number of time points in simulation
        L = L                # Number of one-step trajectories in training
        
        # Time parameters
        t_start = 0.0f0      # Start time of simulation
        t_end = 0.25f0 |> Float32  # End time of simulation
        dt = (t_end - t_start) / N  # Time step
        
        # Additional parameters (empty for this model)
        p = ()
        
        # Return parameters as dictionary
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
            "experiment" => "carmona_dim"
        )
    end
    @unpack d,N,J,K,L,t_start,t_end,dt = get_parameters(d,L)
    
    lambda_poisson = 32
    lambda_exp = 10

    kappa = ones(d) .* 2 .|> Float32

    kappa[1] = 5

    x0 = ones(d) .* 6
    x0[1] = 50
    x0 = x0 .|> Float32
    mu = log.(x0)

    sigma = [j == i ? 1.0f0 : 0.0f0 for i in 1:d, j in 1:d] .* 0.24
    sigma[1,:] .= ones(d).*0.32
    sigma[1,1] = 0.5
    sigma = permutedims(sigma)
    sigma = sigma .|> Float32
    #diagonal_matrix = SMatrix{d,d}(diagonal_matrix)
    # convert diagonal_matrix to SMatrix
    p=Dict("sigma"=>sigma,"mu"=>mu,"d"=>d) 

    new_params = Dict("d"=>d,"J"=>J,"N"=>N,"dt"=>dt,"K"=>K,"L"=>L,"t_start"=>t_start,
        "t_end"=>t_end,"p"=>p,"experiment"=>"carmona_dim")

    """
        dispersion(u, p, t)
    
    Compute the dispersion (diffusion) term of the stochastic process.
    Uses a pre-defined volatility matrix stored in the parameters.
    
    # Arguments
    - `u`: Current state vector
    - `p`: Process parameters (volatility matrix stored in p["sigma"])
    - `t`: Current time
    
    # Returns
    - Dispersion matrix for the SDE
    """
    function dispersion(u, p, t)
        sigma = p["sigma"]
        return sigma
    end

    """
        drift(u, p, t)
    
    Compute the drift term of the stochastic process.
    Uses a mean-reverting model where each component reverts to its equilibrium level.
    
    # Arguments
    - `u`: Current state vector
    - `p`: Process parameters (equilibrium levels stored in p["mu"])
    - `t`: Current time
    
    # Returns
    - Drift vector for the SDE
    """
    function drift(u, p, t)
        mu = p["mu"]
        du = kappa .* u .* (mu .- log.(u))
        return du
    end

    """
        jump(u, p, dt)
    
    Compute the jump component of the stochastic process.
    Implements Poisson jumps with exponential size distribution for the first component only.
    
    # Arguments
    - `u`: Current state vector
    - `p`: Process parameters (dimension stored in p["d"])
    - `dt`: Time step
    
    # Returns
    - Jump vector (non-zero only for first component)
    """
    function jump(u, p, dt)
        d = p["d"]
        deltaN = rand(Binomial(1, lambda_poisson * Float64(dt)))
        size = rand(Exponential(1/lambda_exp))
        jump = exp(size)^deltaN .|> Float32
        du = ones(d)
        du[1] = jump  # Apply jump only to first component
        du = (du .- 1) .* u
        return du
    end


    RandomProcess = OptSwitch.JumpProcess(drift, dispersion, jump)
    x_init = repeat(x0,1,K) .+ randn(d,K) .* 0.05f0
    OptSwitch.generate_paths(RandomProcess,x_init,0.0f0,N,dt,p);

    b = [-1.f0, -1.1f0, -1.2f0]

    mat = [0 0.438 0.876; 0 -0.438*7.5 -0.876*10] .|> Float32 |> permutedims

    payoff_p = Dict("mat"=>mat,"b"=>b,"J"=>J,"d"=>d)

    """
        payoff(x, t, payoff_p)
    
    Compute the payoff vector for the switching problem.
    Maps the high-dimensional state into a 2D state by using the first component
    and the mean of all other components, then applies a linear transformation.
    
    # Arguments
    - `x`: Current state vector
    - `t`: Current time
    - `payoff_p`: Payoff parameters containing matrix (mat), bias (b), and mode count (J)
    
    # Returns
    - Vector of payoffs for each mode (J-dimensional)
    """
    function payoff(x, t, payoff_p)
        mat = payoff_p["mat"]  # Linear transformation matrix
        b = payoff_p["b"]      # Bias vector
        J = payoff_p["J"]      # Number of modes
        
        # Dimension reduction: use first component and mean of remaining components
        x1 = x[1]
        x2 = mean(x[2:end])
        new_x = [x1, x2]
        
        # Apply linear transformation to get payoffs
        res = mat * new_x .+ b
        return SVector{J}(res)
    end

    """
        create_cost_matrix(payoff_p)
    
    Create the cost matrix for switching between modes.
    
    # Arguments
    - `payoff_p`: Payoff parameters containing mode count (J)
    
    # Returns
    - JxJ matrix where element [i,j] is the cost of switching from mode i to j
    """
    function create_cost_matrix(payoff_p)
        J = payoff_p["J"]
        c = 0.01f0               # Base switching cost
        ε = 0.001f0              # Small offset to avoid zero costs
        
        # Create cost matrix with zeros on diagonal
        C = ones(J, J) .* c .+ ε .|> Float32
        C[diagind(C)] .= 0.0f0   # No cost for staying in same mode
        
        return SMatrix{J, J}(C)
    end

    """
        cost(x, t, payoff_p)
    
    Compute the cost matrix for switching between modes.
    
    # Arguments
    - `x`: Current state vector
    - `t`: Current time
    - `payoff_p`: Payoff parameters
    
    # Returns
    - JxJ matrix where element [i,j] is the cost of switching from mode i to j
    """
    function cost(x, t, payoff_p)
        C = create_cost_matrix(payoff_p)
        return C
    end

    """
        payoff_cost_closure(payoff, cost, payoff_p)
    
    Create closures for payoff and cost functions with fixed payoff parameters.
    
    # Arguments
    - `payoff`: Payoff function
    - `cost`: Cost function
    - `payoff_p`: Payoff parameters
    
    # Returns
    - Tuple of closures (payoff_c, cost_c) that take only state and time as arguments
    """
    function payoff_cost_closure(payoff, cost, payoff_p)
        payoff_c(x, t) = payoff(x, t, payoff_p)
        cost_c(x, t) = cost(x, t, payoff_p)
        return payoff_c, cost_c
    end

    payoff_c,cost_c = payoff_cost_closure(payoff,cost,payoff_p)

    payoffmodel = OptSwitch.PayOffModel(payoff_p,payoff_c,cost_c)

    OptSwitch.MLJ_main(new_params,RandomProcess,payoffmodel,x_init;old_models=old_models,dir="HCL",model_types=model_types)
end

# Choose the number of dimensions and L values to iterate over
d_values = [10]
L_values = [1]
for d in d_values
    @info "Running experiment for dimension $d"
    for L in L_values
        main(d,L;dir=datadir("HCL"),model_types=["forest"])
    end
end


#d,N,J = 2,20000,3
#OptSwitch.create_models_by_type(d,N,J;model_types=["forest"])