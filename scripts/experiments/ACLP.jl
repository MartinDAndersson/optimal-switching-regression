"""
    Aïd-Campi-Langrené-Pham (ACLP) Model

This script implements the ACLP model for optimal switching in energy markets.
The model represents a 9-dimensional stochastic process with 4 operational modes,
modeling electricity production with different fuel sources and accounting for
demand, availability, and price processes.

Key dimensions:
- Demand (1D): Electricity demand process
- Availability (3D): Availability factors for different fuel sources
- CO2 price (1D): Carbon emissions price
- Technology prices (3D): Prices for different fuel technologies
- Electricity price (1D): Market price for electricity
"""

using DrWatson
@quickactivate
push!(LOAD_PATH, srcdir())
current_dir = scriptsdir("experiments")
push!(LOAD_PATH, current_dir)
using Revise
using Pkg
using JLD2
using FileIO
using Random
using Distributions
using LinearAlgebra
using Parameters
using BenchmarkTools
using Printf
using BSON
using TimerOutputs
using StaticArrays
using ProgressMeter
#using StatsPlots
using DataFrames
using MLJ
using Glob
using MLUtils
using CairoMakie
using AlgebraOfGraphics
using Makie
using DataFramesMeta
#using CUDA
using Lux, ADTypes, MLUtils, Optimisers, Zygote, Random, Statistics, Printf
import SimpleChains: static
import OptSwitch

"""
    main(K, L; model_types=["pca_knn"], dir="ACLP")

Main function for running the ACLP experiment.

# Arguments
- `K`: Number of trajectories for Monte Carlo simulation
- `L`: Number of one-step trajectories in training
- `model_types`: Types of machine learning models to train (e.g., "pca_knn", "forest")
- `dir`: Directory to save results

# Example
```julia
# Run with 10,000 trajectories and KNN with PCA dimensionality reduction
main(10000, 1, model_types=["pca_knn"], dir="ACLP")
```
"""
function main(K, L; model_types=["pca_knn"], dir="ACLP",save_results=false)
    
    rng = MersenneTwister()
    Random.seed!(rng, 12345)

    d = 9               # Dimension of random process
    n_fuels = 3          # Number of fuel types
    N = 90              # Number of time slices
    J = 4               # Number of operational modes
    t_start = 0.0f0     # Time start value
    t_end = 1.0f0       # Time end value
    dt = (t_end-t_start)/N  # Time step

    # Initial parameters dictionary - will be updated later with process parameters
    new_params = Dict(
        "d" => d,
        "J" => J,
        "N" => N,
        "dt" => dt,
        "K" => K,
        "L" => L,
        "t_start" => t_start,
        "t_end" => t_end,
        "p" => (1,),
        "experiment" => "aid"
    )

    C = [50 10 10; 60 0 10; 60 10 0; 70 0 0] .|> Float32 # Capacities at different modes, fuel s in mode i
    total_C = sum(C,dims=2)
    P_max = 3000.f0 # max electricity prices

    alpha = [4,8,8,8] .|> Float32 # Z and A #
    beta = [15 0.1 0.1 0; 0.1 0.5 -0.1 0; 0.1 -0.1 0.5 0; 0 0 0 0.5] .|> Float32 # each row beta f
    # Beta matrix represents the correlation structure of Z components
    
    # Long-term price impact factors - matrix where rows represent price components
    # and columns represent different impact factors
    s_long = [-4 0 0 1 0; 0 0 0 0 0; 0 2 -1 0 1; 0 0 0 0 0; 0 1 1 1 -1] .|> Float32
    
    # Alpha coefficients control the strength of mean reversion for each price component
    s_alpha = [0.4 0 0.8 0 0] .|> Float32
    
    # Xi combines alpha and long-term factors to create the actual drift matrix
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

    # Update parameters dictionary with process-specific parameters
    new_params["p"] = p
    new_params["experiment"] = "ACLP"

    """
        drift(u, p, t)
    
    Compute the drift term of the stochastic process.
    Separates the state vector into Z (demand/availability) and S (prices)
    components and applies appropriate drift terms to each.
    
    # Arguments
    - `u`: Current state vector
    - `p`: Process parameters (α, Ξ, β, Σ, n_fuels, λ_poisson, λ_exp, d)
    - `t`: Current time
    
    # Returns
    - Drift vector for the stochastic process
    """
    function drift(u, p, t)
        α, Ξ, β, Σ, n_fuels, λ_poisson, λ_exp, d = p
        
        # Split state vector into demand/availability (Z) and prices (S)
        Z = @view u[1:n_fuels+1]        # Demand and availability components
        S = @view u[n_fuels+1+1:end]    # Price components
        
        # Mean-reverting drift for Z components
        Z_drift = -α .* Z
        
        # Price drift with cross-impacts
        S_drift = Ξ * S
        
        # Combine drift components
        return vcat(Z_drift, S_drift)
    end
    
    """
        dispersion(u, p, t)
    
    Compute the dispersion (diffusion) term of the stochastic process.
    Builds a d×d matrix with block structure for Z and S components.
    
    # Arguments
    - `u`: Current state vector
    - `p`: Process parameters (α, Ξ, β, Σ, n_fuels, λ_poisson, λ_exp, d)
    - `t`: Current time
    
    # Returns
    - Dispersion matrix for the stochastic process
    """
    function dispersion(u, p, t)
        α, Ξ, β, Σ, n_fuels, λ_poisson, λ_exp, d = p
        
        # Initialize empty dispersion matrix
        du = zeros(Float32, d, d)
        
        # Split state vector
        Z = @view u[1:n_fuels+1]
        S = @view u[n_fuels+1+1:end]
        
        # Fixed volatility for demand/availability components
        Z_disp = β
        
        # State-dependent volatility for price components
        S_disp = Σ * diagm(S)
        
        # Build block-diagonal matrix
        du[1:n_fuels+1, 1:n_fuels+1] = Z_disp
        du[n_fuels+2:end, n_fuels+2:end] = S_disp
        
        return du
    end
    
    """
        jump(u, p, dt)
    
    Compute jump component of the stochastic process.
    Implements Poisson jumps with exponential size distribution for price components.
    
    # Arguments
    - `u`: Current state vector
    - `p`: Process parameters (α, Ξ, β, Σ, n_fuels, λ_poisson, λ_exp, d)
    - `dt`: Time step
    
    # Returns
    - Jump contribution to state vector
    """
    function jump(u, p, dt)
        α, Ξ, β, Σ, n_fuels, λ_poisson, λ_exp, d = p
        
        # Number of price components
        n_prices = d - (n_fuels + 1)
        
        # Generate Poisson jump events for price components
        deltaN = rand(Binomial(1, λ_poisson*Float64(dt)), n_prices)
        
        # Generate exponential jump sizes
        size = rand(Exponential(1/λ_exp), n_prices)
        jump = exp.(size).^deltaN
        
        # Apply jumps only to price components
        du = ones(d)
        du[n_fuels+2:end] = jump
        du = (du .- 1) .* u
        
        return du
    end

    x0 = [0,0,0,0,20,60,40,20,120] .|> Float32 #[D0, A^1_0, A^2_0, A^3_0, S^0_0, S^1_0, S^2_0, S^3_0]

    D0 = 70.f0
    x_init = copy(x0)
    x_init[1] = D0
    x_init[2:4] .= 1

    x_init = repeat(x_init,inner=(1,K)) .+ randn(Float32,(d,K)) .* 0.005f0

    RandomProcess = OptSwitch.JumpProcess(drift,dispersion,jump)

    # +
    # Defining PayOff model

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

    h_CO2 = [0.5, 2, 0] # CO2 production for technologies
    h_tech = [1.0, 1.5, 1.5] # h

    """
        create_matrix(M, J)
    
    Create a switching matrix indicating which fuels change when switching between modes.
    
    # Arguments
    - `M`: Capacity matrix where rows are modes and columns are fuels
    - `J`: Number of modes
    
    # Returns
    - 3D array where element [i,j,k] is 1 if fuel k changes when switching from mode i to j
    """
    function create_matrix(M, J)
        J, n_fuels = size(M)
        res = zeros(J, J, n_fuels)
        
        for current_mode in 1:J
            for next_mode in 1:J
                # Compare each fuel capacity between modes
                # If different, mark as 1 (change required)
                bit_matrix = M[current_mode,:] .!== M[next_mode,:]
                res[current_mode, next_mode, :] = bit_matrix
            end
        end
        
        return res
    end

    #switch_M = create_matrix(C,J)
    payoff_params=(ZseasMax,ZseasMax,ZseasMaxTrans,ZseasMinTrans,ZseasSum,ZseasDif,Zshift,Dgrowth,n_fuels,h_CO2,h_tech,create_matrix(C,J))

    """
        getAdjX(X, p, time)
    
    Adjust the state vector for seasonal variations and get availability rates.
    
    # Arguments
    - `X`: State vector
    - `p`: Payoff parameters
    - `time`: Current time (0-1 scale)
    
    # Returns
    - Adjusted state vector accounting for seasonality
    """
    function getAdjX(X, p, time)
        # Create a copy of the state to modify
        aX = copy(X)
        
        # Unpack parameters
        ZseasMax, ZseasMax, ZseasMaxTrans, ZseasMinTrans, ZseasSum, ZseasDif, 
        Zshift, Dgrowth, n_fuels, h_CO2, h_tech, switch_M = p
        
        # Calculate seasonal factors with cosine function (annual cycle)
        Zseas = 0.5 .* ZseasSum .+ 0.5 .* ZseasDif .* cos.(2 * π * time .- Zshift)
        
        # Apply seasonality to demand
        aX[1] = X[1] .+ (D0 .+ Dgrowth .* time) .* Zseas[1]
        
        # Apply seasonality to availability factors
        aX[2:(1 + n_fuels)] = X[2:(1 + n_fuels)] .+ 3.0 .* Zseas[2:(1 + n_fuels)]
        
        # Map availability factors to [0,1] range using normal CDF
        aX[2:(1 + n_fuels)] = cdf.(Normal(), aX[2:(1 + n_fuels)])
        
        return aX
    end

    function payoff(x,p,t)::Vector{Float32}
        # Get state vector adjusted for seasonality
        aX = getAdjX(x,p,t)
        
        # Unpack parameters
        ZseasMax, ZseasMax, ZseasMaxTrans, ZseasMinTrans, ZseasSum, ZseasDif, 
        Zshift, Dgrowth, n_fuels, h_CO2, h_tech, switch_M = p
        
        # Extract relevant components from state vector
        @views D = aX[1]                       # Demand
        @views A = aX[2:(1+n_fuels)]           # Availability factors for each fuel
        @views CO2 = aX[(1+n_fuels)+1]         # CO2 price
        @views S = aX[(1+n_fuels)+2:(end-1)]   # Technology prices
        @views price = aX[end]                 # Electricity price
        
        # Calculate profit for each operational mode
        profit = zeros(Float32, J)
        for j in 1:J
            # Adjusted technology prices including CO2 costs
            S2 = h_CO2 .* CO2 + h_tech .* S
            
            # Available capacity based on current availability
            curr_C = C[j,:] .* A
            curr_total_C = sum(curr_C)
            
            # Calculate fixed costs based on technology prices
            fixed_cost = sum(C[j,:] .* S2)
            
            # Revenue is min(capacity, demand) * price
            # This implements the revenue cap where we can't produce more than demand
            rev = min(curr_total_C, D) * price
            
            # Profit is revenue minus fixed costs
            profit[j] = rev - fixed_cost
        end
        
        return profit
    end

    #c = 1.f0
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

    function get_revandcost_closure(payoff,cost,payoff_params)
        rev(x,t) = payoff(x,payoff_params,t) 
        c(x,t) = cost(x,payoff_params,t)
        return rev,c
    end

    rev,c = get_revandcost_closure(payoff,cost,payoff_params)


    payoffmodel = OptSwitch.PayOffModel(payoff_params,rev,c)

    OptSwitch.MLJ_main(new_params, RandomProcess, payoffmodel, x_init, dir=dir, model_types=model_types,save_results=false)

end



# Run the ACLP experiment with 10,000 trajectories and PCA+KNN model
# Reduce K for faster execution, or increase for more accurate results
@benchmark main(10000, 1, model_types=["pca_knn"], dir="ACLP",save_results=false)