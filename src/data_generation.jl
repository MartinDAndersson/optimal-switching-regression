"""
# Data Generation Module

This module implements stochastic process simulation for optimal switching problems.
It supports both pure diffusion processes (SDEs) and jump-diffusion processes.

The mathematical framework follows the general form:
dX_t = μ(X_t, t)dt + σ(X_t, t)dW_t + dJ_t

where:
- μ(X_t, t) is the drift function
- σ(X_t, t) is the dispersion (volatility) function  
- dW_t is a Brownian motion increment
- dJ_t is an optional jump process
"""

################ DEFINING UNDERLYING RANDOM MODEL ################

"""
    StepProcess

Basic container for drift and dispersion functions.

# Fields
- `drift::Function`: Drift function μ(x, p, t)
- `dispersion::Function`: Dispersion function σ(x, p, t)
"""
struct StepProcess
    drift::Function
    dispersion::Function
end

"""
    SDEprocess{FuncType1<:Function, FuncType2<:Function}

Stochastic Differential Equation process of the form:
dX_t = μ(X_t, t)dt + σ(X_t, t)dW_t

# Fields
- `drift::FuncType1`: Drift function μ(x, p, t) returning the deterministic component
- `dispersion::FuncType2`: Dispersion function σ(x, p, t) returning the volatility matrix

# Mathematical Details
The SDE is discretized using the Euler-Maruyama scheme:
X_{n+1} = X_n + μ(X_n, t_n)Δt + σ(X_n, t_n)ΔW_n

where ΔW_n ~ N(0, Δt·I) are independent Gaussian increments.
"""
struct SDEprocess{FuncType1<:Function,FuncType2<:Function}
    drift::FuncType1
    dispersion::FuncType2
end

"""
    JumpProcess{FuncType1<:Function, FuncType2<:Function, FuncType3<:Function}

Jump-diffusion process of the form:
dX_t = μ(X_t, t)dt + σ(X_t, t)dW_t + dJ_t

# Fields
- `drift::FuncType1`: Drift function μ(x, p, t)
- `dispersion::FuncType2`: Dispersion function σ(x, p, t)  
- `jump::FuncType3`: Jump function J(x, p, dt) returning jump increments

# Mathematical Details
Combines continuous diffusion with discrete jumps. The jump component typically follows:
J_t = Σ_{i=1}^{N_t} Y_i

where N_t is a Poisson process with intensity λ and Y_i are i.i.d. jump sizes.
Common implementations include:
- Exponential jumps: Y_i ~ Exponential(λ)
"""
struct JumpProcess{FuncType1<:Function,FuncType2<:Function,FuncType3<:Function}
    drift::FuncType1
    dispersion::FuncType2
    jump::FuncType3
end

################ SIMULATION FUNCTIONS ################

"""
    cpu_euler_maryama_alt(X_prev, time, dt, drift, dispersion, p)

Perform one step of the Euler-Maruyama discretization for SDEs.

# Arguments
- `X_prev::Matrix`: Current state matrix of size (d, M) where d is dimension and M is number of paths
- `time::Real`: Current time
- `dt::Real`: Time step size
- `drift::Function`: Drift function μ(x, p, t) 
- `dispersion::Function`: Dispersion function σ(x, p, t)
- `p`: Parameters passed to drift and dispersion functions

# Returns
- `Matrix`: Next state matrix of size (d, M)

# Mathematical Details
Implements the Euler-Maruyama scheme:
X_{n+1} = X_n + μ(X_n, t_n)Δt + σ(X_n, t_n)√Δt·Z_n

where Z_n ~ N(0, I) are independent standard normal random variables.
"""
function cpu_euler_maryama_alt(X_prev, time, dt, drift, dispersion, p)
    d, M = size(X_prev)
    dX = [
        col .+ drift(col, p, time) .* dt .+
        dispersion(col, p, time) * randn(Float32, d) * sqrt(dt) for
        col in eachcol(X_prev)
    ]
    return reduce(hcat, dX)
end

function cpu_euler_maryama_alt(X_prev, time, dt, StepProc, p)
    d, M = size(X_prev)
    dX = [
        col .+ StepProc.drift(col, p, time) .* dt .+
        StepProc.dispersion(col, p, time) * randn(Float32, d) * sqrt(dt) for
        col in eachcol(X_prev)
    ]
    return reduce(hcat, dX)
end

"""
    euler_jump(X_prev, time, dt, drift, dispersion, jump, p)

Perform one step of Euler-Maruyama discretization with jumps for jump-diffusion processes.

# Arguments
- `X_prev::Matrix`: Current state matrix of size (d, M)
- `time::Real`: Current time
- `dt::Real`: Time step size  
- `drift::Function`: Drift function μ(x, p, t)
- `dispersion::Function`: Dispersion function σ(x, p, t)
- `jump::Function`: Jump function J(x, p, dt) returning jump increments
- `p`: Parameters for the process functions

# Returns
- `Matrix`: Next state matrix including both diffusion and jump components

# Mathematical Details
Implements the jump-diffusion discretization:
X_{n+1} = X_n + μ(X_n, t_n)Δt + σ(X_n, t_n)√Δt·Z_n + J_n

where J_n represents compound Poisson jumps over the interval [t_n, t_n + Δt].
"""
function euler_jump(X_prev, time, dt, drift, dispersion, jump, p)
    d, M = size(X_prev)
    dX = [
        (
            col .+ drift(col, p, time) .* dt .+
            dispersion(col, p, time) * randn(Float32, d) .* sqrt(dt)
        ) .+ jump(col, p, dt) for col in eachcol(X_prev)
    ]
    reduce(hcat, dX)
end

function one_step(X_prev, time, dt, RandomProcess::JumpProcess, p)
    euler_jump(
        X_prev,
        time,
        dt,
        RandomProcess.drift,
        RandomProcess.dispersion,
        RandomProcess.jump,
        p,
    )
end

"""
    one_step(X_prev, time, dt, RandomProcess::SDEprocess, p)

Simulate one time step forward for SDE processes.

# Arguments
- `X_prev::Matrix`: Previous state matrix (d×M)
- `time::Real`: Current time
- `dt::Real`: Time step size
- `RandomProcess::SDEprocess`: SDE process structure
- `p`: Process parameters

# Returns
- `Matrix`: Updated state matrix after one time step
"""
function one_step(X_prev, time, dt, RandomProcess::SDEprocess, p)
    drift, dispersion = RandomProcess.drift, RandomProcess.dispersion
    cpu_euler_maryama_alt(X_prev, time, dt, drift, dispersion, p)
end


"""
    generate_paths(RandomProcess::SDEprocess, x0, t_start, N, dt, p)

Generate multiple trajectories of an SDE process using Euler-Maruyama discretization.

# Arguments
- `RandomProcess::SDEprocess`: SDE process containing drift and dispersion functions
- `x0::Matrix`: Initial states of size (d, K) where d is dimension and K is number of trajectories
- `t_start::Real`: Starting time
- `N::Int`: Number of time steps
- `dt::Real`: Time step size
- `p`: Parameters for the process functions

# Returns
- `Array{Float,3}`: Trajectory array of size (d, N+1, K) containing all simulated paths

# Mathematical Details
Simulates the SDE:
dX_t = μ(X_t, t)dt + σ(X_t, t)dW_t

using the Euler-Maruyama scheme for each trajectory independently.
The resulting trajectories can be used for Monte Carlo estimation of expected values.

# Performance Notes
- Trajectories are generated sequentially for memory efficiency
- For large K, consider using parallel processing for the outer loop
"""
function generate_paths(RandomProcess::SDEprocess, x0, t_start, N, dt, p)
    drift, dispersion = RandomProcess.drift, RandomProcess.dispersion
    d, K = size(x0)
    trajs = zeros(d, N + 1, K)
    trajs[:, 1, :] = x0
    for k = 1:K
        for n = 1:N
            t = t_start + (n - 1) * dt
            x = trajs[:, n, k]
            trajs[:, n+1, k] =
                x .+ drift(x, p, t) .* dt .+ dispersion(x, p, t) * randn(d) .* sqrt(dt)
        end
    end
    return trajs # d x N x K
end


"""
    generate_paths(RandomProcess::JumpProcess, x0, t_start, N, dt, p)

Generate multiple trajectories of a jump-diffusion process.

# Arguments
- `RandomProcess::JumpProcess`: Jump-diffusion process with drift, dispersion, and jump components
- `x0::Matrix`: Initial states of size (d, K)
- `t_start::Real`: Starting time
- `N::Int`: Number of time steps
- `dt::Real`: Time step size
- `p`: Parameters for process functions

# Returns
- `Array{Float,3}`: Trajectory array of size (d, N+1, K) containing all simulated paths

# Mathematical Details
Simulates the jump-diffusion process:
dX_t = μ(X_t, t)dt + σ(X_t, t)dW_t + dJ_t

where J_t is a compound Poisson process. Each time step includes:
1. Continuous diffusion component via Euler-Maruyama
2. Discrete jump component via compound Poisson simulation

# Applications
Common in financial modeling (e.g., Merton jump-diffusion model) and
energy markets where prices exhibit both continuous fluctuations and discrete jumps.
"""
function generate_paths(RandomProcess::JumpProcess, x0, t_start, N, dt, p)
    d, K = size(x0)
    trajs = zeros(d, N + 1, K)
    trajs[:, 1, :] = x0
    for k = 1:K
        for n = 1:N
            t = t_start + (n - 1) * dt
            x = trajs[:, n, k]
            trajs[:, n+1, k] =
                (
                    x .+ RandomProcess.drift(x, p, t) .* dt .+
                    RandomProcess.dispersion(x, p, t) * randn(d) .* sqrt(dt)
                ) .+ RandomProcess.jump(x, p, dt)
        end
    end
    return trajs
end



"""
    define_SDEprob(drift, dispersion, v0, t_start, t_end)

Create an SDE problem structure for use with DifferentialEquations.jl.

# Arguments
- `drift`: Drift function
- `dispersion`: Dispersion function
- `v0`: Initial conditions
- `t_start`: Start time
- `t_end`: End time

# Returns
- SDEProblem object ready for solving
"""
function define_SDEprob(drift, dispersion, v0, t_start, t_end)
    d = size(v0)[1]
    SDEProb = SDEProblem(
        drift,
        dispersion,
        v0,
        (t_start, t_end),
        noise_rate_prototype = zeros(Float32, d, d),
    )
    return SDEProb
end

function get_trajects(RPprob, v0, dt)
    function start_values(prob, i, repeat)
        remake(prob, u0 = v0[:, i])
    end
    d, K = size(v0)
    ensembleRP = EnsembleProblem(RPprob, prob_func = start_values)
    RPsol = solve(ensembleRP, EnsembleThreads(), trajectories = K, dt = dt, saveat = dt)
end


function one_step(RPprob, v0, dt, t)
    function start_values(prob, i, repeat)
        remake(prob, u0 = v0[:, i], tspan = (t, t + dt))
    end
    d, M = size(v0)
    ensembleprob = EnsembleProblem(RPprob, prob_func = start_values)
    sol =
        solve(ensembleprob, EnsembleThreads(), trajectories = M, dt = dt, saveat = [t + dt])

end




