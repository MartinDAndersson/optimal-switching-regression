"""
Complexity Validation Benchmark

Simple benchmark to validate theoretical complexity table with real-world timings.
Provides concrete timing estimates (seconds, minutes, hours) for different problem sizes.
"""

using DrWatson
@quickactivate
push!(LOAD_PATH, srcdir())

using BenchmarkTools
using DataFrames
using Random
using MLJ
using Printf
using Statistics
using CSV
using Serialization

# Load SimpleChains for direct neural network benchmarking
using SimpleChains
using Random, Statistics

# Load MLJ models
KNN = MLJ.@load KNNRegressor
RandomForest = MLJ.@load RandomForestRegressor pkg=DecisionTree
PCA = MLJ.@load PCA pkg=MultivariateStats
LinearRegressor = MLJ.@load LinearRegressor pkg=MLJLinearModels
RidgeRegressor = MLJ.@load RidgeRegressor pkg=MLJLinearModels
LassoRegressor = MLJ.@load LassoRegressor pkg=MLJLinearModels
LGBMReg = MLJ.@load LGBMRegressor pkg=LightGBM

"""
    create_data(M, d)

Create synthetic regression data of size M×d.
"""
function create_data(M, d)
    Random.seed!(42)
    X = randn(Float32, M, d)
    y = sin.(X[:, 1]) .+ 0.5 .* sum(X[:, 1:min(3, d)], dims=2)[:] .+ 0.1 * randn(Float32, M)
    return X, y
end

"""
    create_simplechains_network(d; hidden_size=64)

Create a neural network using SimpleChains directly.
"""
function create_simplechains_network(d; hidden_size=64)
    return SimpleChain(
        static(d),
        TurboDense(tanh, hidden_size),
        TurboDense(tanh, hidden_size),
        TurboDense(identity, 1)
    )
end

"""
    time_simplechains_network(model_name, M, d; epochs=50, hidden_size=64)

Benchmark neural network training using SimpleChains directly.
"""
function time_simplechains_network(model_name, M, d; epochs=50, hidden_size=64)
    println("  Testing $model_name on $(M)×$(d) data...")
    
    try
        # Generate data
        X, y = create_data(M, d)
        X = Float32.(permutedims(X))  # Convert to Float32 d×M format for SimpleChains
        y = Float32.(reshape(y, 1, :))  # Convert to 1×M format for SimpleChains
        
        # Create SimpleChains model with loss
        model = create_simplechains_network(d, hidden_size=hidden_size)
        model_with_loss = SimpleChains.add_loss(model, SimpleChains.SquaredLoss(y))
        
        # Initialize parameters and gradient storage
        params = SimpleChains.init_params(model)
        grad = SimpleChains.alloc_threaded_grad(model)
        
        # Training loop with manual timing
        training_times = Float64[]
        training_memories = Float64[]
        
        for run in 1:3  # Run 3 samples for timing
            # Reset parameters for each run
            local_params = SimpleChains.init_params(model)
            
            # Time the training
            start_time = time()
            start_memory = Base.gc_live_bytes()
            
            # Use SimpleChains' built-in training with ADAM optimizer
            SimpleChains.train_unbatched!(
                grad, local_params, model_with_loss, X, 
                SimpleChains.ADAM(0.001f0), epochs
            )
            
            end_time = time()
            end_memory = Base.gc_live_bytes()
            
            push!(training_times, end_time - start_time)
            push!(training_memories, max(0, (end_memory - start_memory) / 1024^2))  # Convert to MB
        end
        
        training_time = median(training_times)
        training_memory = median(training_memories)
        
        # Train final model for prediction benchmarking
        final_params = SimpleChains.init_params(model)
        SimpleChains.train_unbatched!(
            grad, final_params, model_with_loss, X, 
            SimpleChains.ADAM(0.001f0), epochs
        )
        
        # Benchmark prediction
        pred_size = min(1000, M)
        X_pred = X[:, 1:pred_size]
        
        prediction_result = @benchmark $model($X_pred, $final_params) seconds=15 samples=5
        
        prediction_time = median(prediction_result.times) / 1e9
        prediction_memory = median(prediction_result.memory) / 1024^2
        prediction_time_per_sample = prediction_time / pred_size
        
        # Estimate model size (parameters count × 4 bytes for Float32)
        param_count = length(final_params)
        model_size_mb = (param_count * 4) / (1024^2)  # Approximate size in MB
        
        println("    ✓ Training: $(format_time(training_time)) ($(round(training_memory, digits=1))MB)")
        println("      Prediction: $(format_time(prediction_time_per_sample))/sample on $(pred_size) samples ($(round(prediction_memory, digits=1))MB)")
        println("      Model size: $(round(model_size_mb, digits=1))MB (~$(param_count) parameters)")
        
        return (
            training_time=training_time,
            training_memory=training_memory,
            prediction_time=prediction_time_per_sample,
            prediction_memory=prediction_memory,
            model_size=model_size_mb,
            pred_samples=pred_size,
            status="success"
        )
        
    catch e
        println("    ERROR: $e")
        return (
            training_time=NaN,
            training_memory=NaN,
            prediction_time=NaN,
            prediction_memory=NaN,
            model_size=NaN,
            pred_samples=0,
            status="error"
        )
    end
end

"""
    measure_model_size(mach)

Measure the size of a trained model when serialized (saved to disk).
"""
function measure_model_size(mach)
    # Create a temporary buffer to serialize the machine
    buffer = IOBuffer()
    
    try
        # Serialize the fitted machine to measure its size
        serialize(buffer, mach)
        size_bytes = position(buffer)
        size_mb = size_bytes / (1024^2)  # Convert to MB
        
        close(buffer)
        return size_mb
        
    catch e
        # If serialization fails, try just the fitted parameters
        try
            close(buffer)
            buffer = IOBuffer()
            serialize(buffer, fitted_params(mach))
            size_bytes = position(buffer)
            size_mb = size_bytes / (1024^2)
            close(buffer)
            return size_mb
        catch
            close(buffer)
            return NaN
        end
    end
end

"""
    time_model(model_name, model, M, d)

Time a single model on M×d data using @benchmark for accurate timing and memory measurement.
"""
function time_model(model_name, model, M, d)
    # Generate data
    X, y = create_data(M, d)
    X_table = MLJ.table(X)
    
    println("  Testing $model_name on $(M)×$(d) data...")
    
    try
        # Benchmark training
        training_result = @benchmark begin
            mach = machine($model, $X_table, $y)
            fit!(mach, verbosity=0)
            mach
        end seconds=30 samples=3
        
        training_time = median(training_result.times) / 1e9  # Convert to seconds
        training_memory = median(training_result.memory) / 1024^2  # Convert to MB
        
        # Create trained model for prediction benchmark
        mach = machine(model, X_table, y)
        fit!(mach, verbosity=0)
        
        # Benchmark prediction on 1000 samples (or all if M < 1000)
        pred_size = min(1000, M)
        X_pred = MLJ.table(X[1:pred_size, :])
        
        prediction_result = @benchmark predict($mach, $X_pred) seconds=15 samples=5
        
        prediction_time = median(prediction_result.times) / 1e9  # Convert to seconds
        prediction_memory = median(prediction_result.memory) / 1024^2  # Convert to MB
        
        # Calculate per-sample prediction time
        prediction_time_per_sample = prediction_time / pred_size
        
        # Measure model size when saved
        model_size_mb = measure_model_size(mach)
        
        println("    ✓ Training: $(format_time(training_time)) ($(round(training_memory, digits=1))MB)")
        println("      Prediction: $(format_time(prediction_time_per_sample))/sample on $(pred_size) samples ($(round(prediction_memory, digits=1))MB)")
        println("      Model size: $(round(model_size_mb, digits=1))MB when saved")
        
        return (
            training_time=training_time, 
            training_memory=training_memory,
            prediction_time=prediction_time_per_sample, 
            prediction_memory=prediction_memory,
            model_size=model_size_mb,
            pred_samples=pred_size,
            status="success"
        )
        
    catch e
        println("    ERROR: $e")
        return (
            training_time=NaN, 
            training_memory=NaN,
            prediction_time=NaN, 
            prediction_memory=NaN,
            model_size=NaN,
            pred_samples=0,
            status="error"
        )
    end
end

"""
    format_time(seconds)

Format time in human-readable form.
"""
function format_time(seconds)
    if seconds < 0.001
        return "$(round(seconds*1e6, digits=1))μs"
    elseif seconds < 1.0
        return "$(round(seconds*1000, digits=1))ms"
    elseif seconds < 60.0
        return "$(round(seconds, digits=2))s"
    elseif seconds < 3600.0
        return "$(round(seconds/60, digits=1))min"
    else
        return "$(round(seconds/3600, digits=1))hr"
    end
end

"""
    run_complexity_validation()

Run benchmark on different problem sizes to validate complexity table.
"""
function run_complexity_validation()
    println("=== Complexity Validation Benchmark ===")
    println("Measuring real-world timings for theoretical complexity table")
    
    # Define models matching your table
    models = Dict(
        "Linear (OLS)" =>Standardizer() |> LinearRegressor(),
        #"Ridge" =>Standardizer() |> RidgeRegressor(lambda=0.1),
        #"LASSO" => LassoRegressor(lambda=0.1),
        "Random Forest" => RandomForest(n_trees=25, max_depth=3),  # T=25
        "LightGBM" => LGBMReg(num_iterations=200, learning_rate=0.1),  # T=100
        "k-NN" => Standardizer() |> KNN(K=10),
        "PCA-k-NN" => Standardizer() |> PCA(maxoutdim=min(6, 10)) |> KNN(K=10)
    )
    
    # Neural network models (handled separately)
    neural_models = Dict(
        "Neural Network (SC)" => "sc_neural"  # SimpleChains neural network
    )
    
    # Problem sizes - start small and scale up
    # Define separate ranges for M and d to create cartesian product
    M_values = [1000, 5000, 10000, 20000]
    d_values = [10, 50, 100, 200]
    
    # Create cartesian product of problem sizes
    problem_sizes = [(M=M, d=d) for M in M_values, d in d_values][:]
    
    results = DataFrame(
        model = String[],
        M = Int[],
        d = Int[],
        training_time = Float64[],
        training_memory = Float64[],
        prediction_time = Float64[],
        prediction_memory = Float64[],
        model_size = Float64[],
        pred_samples = Int[],
        status = String[]
    )
    
    println("\nTesting $(length(models)) models on $(length(problem_sizes)) problem sizes...\n")
    
    for (M, d) in problem_sizes
        println("Problem Size: $(M) samples × $(d) dimensions")
        println("="^50)
        
        # Test MLJ models
        for (model_name, model) in models
            # Skip PCA-kNN for low dimensions
            if model_name == "PCA-k-NN" && d <= 6
                println("  SKIPPED: $model_name (d=$d ≤ 6, PCA not beneficial)")
                push!(results, (
                    model=model_name, M=M, d=d, 
                    training_time=NaN, training_memory=NaN,
                    prediction_time=NaN, prediction_memory=NaN,
                    model_size=NaN, pred_samples=0, status="skipped_dim"
                ))
                continue
            end
            
            # Adjust PCA dimensions for current problem
            if model_name == "PCA-k-NN"
                pca_dims = min(6, d÷2)  # Use at most half the dimensions or 6
                model = Standardizer() |> PCA(maxoutdim=pca_dims) |> KNN(K=10)
            end
            
            result = time_model(model_name, model, M, d)
            
            push!(results, (
                model=model_name, M=M, d=d,
                training_time=result.training_time,
                training_memory=result.training_memory,
                prediction_time=result.prediction_time,
                prediction_memory=result.prediction_memory,
                model_size=result.model_size,
                pred_samples=result.pred_samples,
                status=result.status
            ))
        end
        
        # Test neural network models
        for (model_name, model_type) in neural_models
            if model_type == "sc_neural"
                # Use fixed, safe values for testing
                epochs = 30  # Fixed reasonable value
                hidden_size = 64  # Fixed reasonable value
                
                result = time_simplechains_network(model_name, M, d, epochs=epochs, hidden_size=hidden_size)
                
                push!(results, (
                    model=model_name, M=M, d=d,
                    training_time=result.training_time,
                    training_memory=result.training_memory,
                    prediction_time=result.prediction_time,
                    prediction_memory=result.prediction_memory,
                    model_size=result.model_size,
                    pred_samples=result.pred_samples,
                    status=result.status
                ))
            end
        end
        println()
    end
    
    return results
end

"""
    print_summary_table(results)

Print a summary table with timing estimates.
"""
function print_summary_table(results)
    println("\n" * "="^80)
    println("COMPLEXITY VALIDATION SUMMARY")
    println("="^80)
    
    successful_results = filter(row -> row.status == "success", results)
    
    if nrow(successful_results) == 0
        println("No successful benchmarks completed!")
        return
    end
    
    println("Training & Prediction Summary:")
    println("-"^70)
    
    for model in unique(successful_results.model)
        model_data = filter(row -> row.model == model, successful_results)
        
        if nrow(model_data) > 0
            println("\n$model:")
            for row in eachrow(model_data)
                size_str = "$(row.M)×$(row.d)"
                train_str = format_time(row.training_time)
                train_mem = "$(round(row.training_memory, digits=1))MB"
                pred_str = format_time(row.prediction_time)
                pred_mem = "$(round(row.prediction_memory, digits=1))MB"
                model_size = "$(round(row.model_size, digits=1))MB"
                println("  $(size_str): Train=$(train_str) ($(train_mem)), Predict=$(pred_str)/sample ($(pred_mem)), Size=$(model_size)")
            end
        end
    end
    
    println("\n" * "="^80)
    println("EXTRAPOLATED ESTIMATES for larger problems:")
    println("="^80)
    
    # Simple extrapolation for common scenarios
    println("\nOptSwitch scenarios (rough estimates):")
    println("• Small problem (1K samples, 10D): seconds to minutes")
    println("• Medium problem (10K samples, 50D): minutes to hours") 
    println("• Large problem (100K samples, 100D): hours to days")
    println("• Note: Actual times depend heavily on model choice!")
end

# Main execution
function main()
    results = run_complexity_validation()
    
    # Save results
    save_path = datadir("complexity_validation.csv")
    CSV.write(save_path, results)
    println("Results saved to $save_path")
    
    # Print summary
    print_summary_table(results)
    
    return results
end

# Uncomment to run
results = main()