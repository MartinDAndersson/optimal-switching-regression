# Optimal Switching with Regression Methods

A Julia framework for solving optimal switching problems using regression-based machine learning techniques. This project implements the Longstaff-Schwartz algorithm with various regression methods to model and optimize dynamic decision-making processes where a system can switch between different modes of operation over time.

## Overview

OptSwitch provides tools for:

- Modeling stochastic processes (including jump processes)
- Generating simulation paths and trajectories
- Implementing and training neural networks to approximate value functions
- Computing optimal switching strategies
- Evaluating strategy performance
- Visualizing results through various plots

The framework is particularly suited for applications in energy markets, where power plants may switch between different fuel sources or operation modes based on market conditions and prices.

## Components

- **Core Framework** (`src/OptSwitch.jl`): Main module that imports dependencies and exports key functions
- **Stochastic Process Models** (`src/data_generation.jl`): Implementation of SDE and jump processes for simulation
- **Neural Network Training** (`src/Lux.jl`): Infrastructure for training and evaluating neural networks
- **Model Abstractions** (`src/model_abstractions.jl`): Unified interface for different model types
- **Training Utilities** (`src/TrainingLogger.jl`, `src/TrainingPlots.jl`): Tools for logging and visualizing model training

## Model Implementations and Analysis

The codebase includes various implementation strategies for optimal switching problems:

- **Machine Learning Models**: 
  - **k-NN**: k=10, with PCA preprocessing for high dimensions
  - **Tree Methods**: Random Forests (25 trees), LightGBM (200 iterations)
  - **Linear Models**: OLS, Ridge (λ=0.1), LASSO (λ=0.1)
  - **Neural Networks**: 1-2 layers, ReLU activation, dropout regularization
  - **Dimensionality reduction**: PCA+k-NN for high-dimensional problems

- **Strategy Types**:
  - **A Posteriori Optimal**: Perfect hindsight benchmark (ideal but unrealistic)
  - **Dynamic Programming**: Value function approximation via backward induction
  - **Greedy Strategies**: Decisions based on immediate payoffs minus switching costs
  - **Learned Strategies**: Using trained ML models to approximate optimal decisions

Helper functions in `src/plots_common.jl` provide utilities for:
- Calculating optimal value functions and reference benchmarks
- Determining best operational modes based on current state
- Computing accumulated values from different strategies
- Analyzing strategy performance metrics (value, decision quality)
- Visualizing results and model comparisons

## Key Research Findings

The comprehensive experiments reveal important insights about regression methods for optimal switching:

- **k-NN dominance**: k-nearest neighbors consistently achieves near-optimal performance across all experiments
- **Dimension scaling**: PCA+k-NN maintains performance up to 50 dimensions while other methods degrade
- **Neural network instability**: Decision boundaries shift unpredictably across time steps due to contaminated training targets
- **Simple methods advantage**: Noisy targets from backward induction favor robust, stable methods over complex ones
- **Strategic planning**: k-NN excels in problems requiring forward-looking decisions (BSP experiment)

## Use Cases

The package has been used to model problems such as:
- Optimal switching between different energy sources in power production
- Dynamic adjustment of production capacities based on market conditions
- Financial decision-making under uncertainty

## Implemented Experiments

This package contains implementations of several optimal switching problems:

- **CL (Carmona-Ludkovski)**: A 2D jump-diffusion process with 3 operational modes
  - Models electricity and gas price dynamics with mean reversion and jumps
  - Different operating modes with varying payoff profiles and switching costs
  - Benchmark problem from Carmona & Ludkovski article
  - 50,000 trajectories, 180 time steps, T=0.25

- **HCL (High-dimensional Carmona-Ludkovski)**: Extension of CL to higher dimensions (2D to 50D)
  - Tests scalability of different machine learning approaches
  - Demonstrates dimension reduction techniques like PCA
  - Same payoff structure as CL but using first component and mean of others
  - 20,000 trajectories, 180 time steps, T=0.25

- **BSP (Banded Shift Process)**: A 1D state space with 10 operational modes
  - Mean-reverting process with time-varying mean μ(t) = sin(2πt)
  - Banded payoff structure designed to test strategic planning capability
  - Tests ability to learn complex mode-switching boundaries
  - 20,000 trajectories, 36 time steps, T=1.0

- **ACLP (Aïd-Campi-Langrené-Pham)**: A 9D jump-diffusion process with 4 operational modes
  - 4D demand/availability (Z_t) + 5D price processes (S_t)
  - Realistic model of multi-fuel power plant operation
  - Includes demand, availability, CO₂ prices, fuel prices, and electricity price
  - Seasonal effects and jump processes in price components
  - Switching costs based on which fuels change between modes
  - 50,000 trajectories, 90 time steps, T=1.0

## References

This repository implements numerical experiments from the optimal switching literature:

### **Experiment 1: Carmona-Ludkovski (CL)**
*Gas-fired power plant management with electricity and gas price dynamics*

- **Original Paper**: Carmona, R. and Ludkovski, M. (2008). "Pricing asset scheduling flexibility using optimal switching." *Applied Mathematical Finance*, 15(5-6), 405-447.
- **Implementation Based On**: Bayraktar, E., Cohen, A., and Nellis, A. (2023). "A neural network approach to high-dimensional optimal switching problems with jumps in energy markets." *SIAM Journal on Financial Mathematics*, 14(4), 1028-1061.

### **Experiment 2: Aïd-Campi-Langrené-Pham (ACLP)**
*Multi-fuel power plant optimization in high dimensions*

- **Original Paper**: Aïd, R., Campi, L., Langrené, N., and Pham, H. (2014). "A probabilistic numerical method for optimal multiple switching problems in high dimension." *SIAM Journal on Financial Mathematics*, 5, 191-231.
- **Implementation Based On**: Bayraktar, E., Cohen, A., and Nellis, A. (2023). "A neural network approach to high-dimensional optimal switching problems with jumps in energy markets." *SIAM Journal on Financial Mathematics*, 14(4), 1028-1061.

### **Experiment 3: Banded Shift Process (BSP)**
*Artificial multi-timescale planning problem*

- **Source**: Original problem designed to test forward-planning vs. myopic strategies
- **Purpose**: Evaluates methods on scenarios requiring multi-timescale decision making

### **Experiment 4: High-dimensional Carmona-Ludkovski (HCL)**
*Dimensional scaling test of the CL model*

- **Based On**: Extension of Bayraktar, E., Cohen, A., and Nellis, A. (2023). "A neural network approach to high-dimensional optimal switching problems with jumps in energy markets." *SIAM Journal on Financial Mathematics*, 14(4), 1028-1061.
- **Purpose**: Tests method performance as state dimensions scale from 2 to 50

## Directory Structure

All experiments follow a common structure:

```
scripts/
├── experiments/                 # Core model implementations
│   ├── CL.jl                    # Carmona-Ludkovski model
│   ├── HCL.jl                   # High-dimensional extension
│   ├── BSP.jl                   # Basic Switching Problem
│   └── ACLP.jl                  # Advanced energy model
├── analysis/                    # Analysis scripts for publication-quality plots
│   ├── CL_article_plots.jl      # Analysis and plots for CL model
│   ├── HCL_article_plots.jl     # Analysis and plots for HCL model
│   ├── BSP_article_plots.jl     # Analysis and plots for BSP model
│   └── ACLP_article_plots.jl    # Analysis and plots for ACLP model
├── plotting/                    # Plotting utilities for each model
│   ├── cl_plots.jl              # Plotting functions for CL model
│   ├── hcl_plots.jl             # Plotting functions for HCL model
│   ├── bsp_plots.jl             # Plotting functions for BSP model
│   └── aclp_plots.jl            # Plotting functions for ACLP model
└── benchmarks/                  # Validation and timing scripts
    ├── complexity_validation.jl # Benchmark script for validating theoretical complexity
    └── benchmark_training_times.jl # Training time benchmarks
```

## Running Experiments

To run an experiment, follow these two steps:

```julia
# Step 1: Train the models (may take significant time)
include("scripts/experiments/CL.jl")

# Step 2: Generate publication-quality plots and analysis
include("scripts/analysis/CL_article_plots.jl")
```

### Complexity Validation

To validate the theoretical complexity estimates with real-world benchmarks:

```julia
# Run complexity validation benchmark
include("scripts/benchmarks/complexity_validation.jl")
```

This script benchmarks different machine learning models (Linear, Random Forest, LightGBM, k-NN, PCA-k-NN, Neural Networks) on various problem sizes to provide concrete timing estimates and validate theoretical complexity predictions. Results are saved to `data/complexity_validation.csv`.

## Customizing Models and Networks

The OptSwitch framework provides a flexible system for comparing different machine learning approaches. You can easily customize models by modifying the `model_types` parameter when running experiments.

### Available Model Types

**Machine Learning Models (MLJ-based):**
- `"knn"` - k-Nearest Neighbors (k=10) with standardization
- `"pca_knn"` - PCA (6 components) + k-NN pipeline for high dimensions
- `"weighted_knn"` - Weighted k-NN with PCA preprocessing and adaptive Gaussian weights
- `"forest"` - Random Forest (25 trees, max depth 3)
- `"lgbm"` - LightGBM (200 iterations, learning rate 0.05)
- `"ridge"` - Ridge regression (λ=0.1) with standardization
- `"linear"` - Linear regression with polynomial features (degree 6 for d<5)
- `"lasso"` - LASSO regression (λ=0.1) with standardization
- `"hybrid"` - Ensemble model combining linear regression and k-NN with linear metalearner

**Neural Networks (Lux/SimpleChains):**
- `"network"` or `"neural"` - Creates optimized architectures:
  - Low-dim (d<5): 32→1 (shallow) or 16→16→16→1 (deep)
  - High-dim (d≥5): 128→1 (shallow) or 128→64→1 (deep)

**Special:**
- `"all"` - Train all available models for comprehensive comparison

### Usage Examples

```julia
# Run experiment with specific model types
include("scripts/experiments/CL.jl")
main(model_types=["knn", "forest"])              # Compare k-NN vs Random Forest
main(model_types=["network", "pca_knn"])         # Compare neural networks vs PCA+k-NN
main(model_types=["all"])                        # Train all available models

# For high-dimensional problems
include("scripts/experiments/HCL.jl") 
main(50, model_types=["pca_knn", "lgbm"])        # Recommended for d=50
```

### Adding Custom Models

To add a new model type, modify the `create_models_by_type()` function in `src/OptSwitch.jl`:

```julia
elseif model_type == "my_custom_model"
    # Define your MLJ model
    custom_model = Standardizer() |> MyCustomRegressor(param1=value1)
    push!(selected_models, model_to_learningmodel(custom_model, "my_custom_model", "MLJ", N, J))
```

### Neural Network Customization

Neural network architectures are defined in `src/OptSwitch.jl` functions:
- `create_model_lowdim()` - For d < 5 dimensions
- `create_model_dim()` - For d ≥ 5 dimensions  

To modify architectures, edit these functions to change:
- Hidden layer sizes
- Number of layers  
- Activation functions
- Dropout rates
- Learning rates

### Model Configuration

Each model is wrapped in a `LearningModel` struct that provides:
- Unified training interface across MLJ and neural network models
- Automatic handling of multiple time steps (N) and modes (J)
- Built-in logging and performance tracking
- Support for warm-starting from previous experiments

Experiment results are saved to:
- `/data/[experiment_name]/machines/` - Trained machine learning models
- `/plots/[experiment_name]/` - Generated plots and visualizations
- `/data/[experiment_name]/` - Performance metrics and analysis results

## Creating New Experiments

The framework is designed to be extensible. To create a new experiment:

1. Define a stochastic process with these components:
   - `drift(u, p, t)`: Drift dynamics
   - `dispersion(u, p, t)`: Volatility structure
   - `jump(u, p, dt)` (optional): For Poisson jump processes

2. Create payoff and cost functions:
   - `payoff(x, t)`: Returns vector of rewards for each mode at state x, time t
   - `cost(x, t)`: Returns matrix of switching costs between modes

3. Run machine learning training:
   ```julia
   # Configure parameters
   params = Dict("d" => dim, "J" => modes, "N" => timesteps, ...)
   
   # Initialize stochastic process and payoff model
   process = OptSwitch.JumpProcess(drift, dispersion, jump)
   payoffmodel = OptSwitch.PayOffModel(payoff_params, payoff, cost)
   # Run machine learning training
   OptSwitch.MLJ_main(params,process,payoffmodel,initial_states;old_models=[],dir="default",model_types=["forest"])
   ```

## Installation

This code base is using the [Julia Language](https://julialang.org/) and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project.

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

You may notice that most scripts start with the commands:
```julia
using DrWatson
@quickactivate "OptSwitch"
```
which auto-activate the project and enable local path handling from DrWatson.

## Performance Considerations


### Performance and Computational Requirements

**Expected Runtime** (varies by experiment and method):
- **Small experiments** (CL, BSP): Minutes to hours
- **Large experiments** (ACLP, HCL d=50): Hours depending on method and configuration

**Optimization Tips**:
- Neural network training can be computationally intensive; adjust network size and training iterations based on available resources
- To reduce computation time, reduce the number of trajectories K (e.g., from 50000 to 10000)
- k-NN and tree methods generally faster than neural networks for training

## Package Dependencies

The main dependencies for this package are:
- **DrWatson**: Project organization and reproducibility
- **Lux**: Neural network models
- **SimpleChains**: Fast CPU-optimized neural networks
- **MLJ**: Machine learning framework with model interfaces
- **NearestNeighborModels**: k-NN regression implementation
- **DecisionTree**: Random Forest implementation
- **LightGBM**: Gradient boosting trees
- **MLJLinearModels**: Ridge, LASSO, and OLS regression
- **MultivariateStats**: PCA for dimensionality reduction
- **Distributions**: Probability distributions for stochastic processes
- **CairoMakie**: Plotting and visualization
- **DataFrames**: Data manipulation
- **StaticArrays**: Performance-optimized fixed-size arrays


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

This project was developed by Martin, with Benny and Marcus as co-authors.