module TrainingLogger

using Printf
using Statistics
using DataStructures
using Dates
using CSV
using DataFrames

export ModelLogger, log_losses!, get_previous_losses, get_average_losses, print_summary
export save_model_state, save_training_metrics, export_to_csv

"""
    ModelLogger

A comprehensive logger for tracking model training across multiple time steps and modes.

# Fields
- `name`: The model name
- `current_step`: Current time step being processed
- `total_steps`: Total number of time steps in the simulation
- `modes`: Number of different modes being trained
- `loss_history`: Nested dictionary storing training and validation losses
- `metadata`: Additional context information about the model and training
- `log_dir`: Directory where logs and plots are saved
"""
mutable struct ModelLogger
    name::String
    current_step::Int
    total_steps::Int
    modes::Int
    loss_history::Dict{Int, Dict{Int, Vector{Tuple{Float64, Float64}}}}  # n -> j -> [(train_loss, test_loss)]
    metadata::Dict{String, Any}
    log_dir::String
    
    function ModelLogger(name::String, total_steps::Int, modes::Int; log_dir="logs", metadata=Dict{String, Any}())
        # Create loss history structure
        loss_history = Dict{Int, Dict{Int, Vector{Tuple{Float64, Float64}}}}()
        for n in 1:total_steps
            loss_history[n] = Dict{Int, Vector{Tuple{Float64, Float64}}}()
            for j in 1:modes
                loss_history[n][j] = Vector{Tuple{Float64, Float64}}()
            end
        end
        
        # Create timestamped logger directory
        timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
        model_log_dir = joinpath(log_dir, "$(name)_$(timestamp)")
        mkpath(model_log_dir)
        
        # Initialize CSV log file with headers
        log_file = joinpath(model_log_dir, "training_metrics.csv")
        CSV.write(log_file, DataFrame(
            timestamp = String[],
            time_step = Int[],
            mode = Int[],
            train_loss = Float64[],
            val_loss = Float64[],
            improvement = Float64[]
        ))
        
        # Store basic metadata
        metadata["start_time"] = timestamp
        metadata["model_name"] = name
        metadata["total_steps"] = total_steps
        metadata["modes"] = modes
        
        new(name, 0, total_steps, modes, loss_history, metadata, model_log_dir)
    end
end

"""
    log_losses!(logger, n, j, train_loss, test_loss; verbose=true)

Log training and validation losses for a specific time step and mode.

# Arguments
- `logger`: The ModelLogger instance
- `n`: The time step
- `j`: The mode
- `train_loss`: Training loss value
- `test_loss`: Validation/test loss value
- `verbose`: Whether to print progress information
"""
function log_losses!(logger::ModelLogger, n::Int, j::Int, train_loss::Float64, test_loss::Float64; verbose=true)
    push!(logger.loss_history[n][j], (train_loss, test_loss))
    logger.current_step = max(logger.current_step, n)
    
    # Calculate loss changes if we have previous values
    prev_losses = get_previous_losses(logger, n, j)
    train_change = isnothing(prev_losses) ? 0.0 : train_loss - prev_losses[1]
    test_change = isnothing(prev_losses) ? 0.0 : test_loss - prev_losses[2]
    rel_improvement = isnothing(prev_losses) ? 0.0 : 
                     (prev_losses[2] - test_loss) / (abs(prev_losses[2]) + 1e-10) * 100.0
    
    # Append to CSV log file
    log_file = joinpath(logger.log_dir, "training_metrics.csv")
    df = DataFrame(
        timestamp = [Dates.format(now(), "yyyy-mm-dd HH:MM:SS")],
        time_step = [n],
        mode = [j],
        train_loss = [train_loss],
        val_loss = [test_loss],
        improvement = [rel_improvement]
    )
    # Use flush=true to ensure immediate writing to disk
    CSV.write(log_file, df, append=true)
    # Force file flush to disk
    open(log_file, "a") do io
        flush(io)
    end
    
    if verbose
        # Progress bar for time steps
        progress = "["
        prog_width = 20
        filled = round(Int, prog_width * n / logger.total_steps)
        progress *= "=" ^ filled
        progress *= " " ^ (prog_width - filled)
        progress *= "]"
        
        # Format loss changes with plain text indicators
        train_indicator = train_change <= 0 ? "decreased" : "increased"
        test_indicator = test_change <= 0 ? "decreased" : "increased"
        
        # Only print detailed logs periodically (e.g., every 5th mode or step change)
        if j % 5 == 0 || j == 1 || j == logger.modes || abs(rel_improvement) > 10.0
            println("\n=== Training Model: $(logger.name) ===")
            println("Time step $n/$(logger.total_steps) $progress $(round(Int, 100 * n / logger.total_steps))%")
            println("Mode $j/$(logger.modes):")
            @printf("  Training Loss: %.4f (%+.4f)\n", train_loss, train_change)
            @printf("  Validation Loss: %.4f (%+.4f)\n", test_loss, test_change)
           #@printf("  Improvement: %.2f%%\n", rel_improvement)
            
            # Force terminal flush
            flush(stdout)
        end
    end
end

"""
    get_previous_losses(logger, n, j)

Get the most recent losses from the previous time step for a specific mode.
Returns `nothing` if no previous losses exist.
"""
function get_previous_losses(logger::ModelLogger, n::Int, j::Int)
    if n > 1 && !isempty(logger.loss_history[n-1][j])
        return last(logger.loss_history[n-1][j])
    end
    return nothing
end

"""
    get_average_losses(logger, n)

Calculate the average training and validation losses across all modes for a specific time step.
"""
function get_average_losses(logger::ModelLogger, n::Int)
    train_losses = Float64[]
    test_losses = Float64[]
    
    for j in 1:logger.modes
        if !isempty(logger.loss_history[n][j])
            train_loss, test_loss = last(logger.loss_history[n][j])
            push!(train_losses, train_loss)
            push!(test_losses, test_loss)
        end
    end
    
    return isempty(train_losses) ? (0.0, 0.0) : (mean(train_losses), mean(test_losses))
end

"""
    print_summary(logger)

Print a summary of the training process including final losses and improvements.
"""
function print_summary(logger::ModelLogger)
    println("\n=== Training Summary for $(logger.name) ===")
    println("Final Average Losses:")
    
    # Calculate final averages
    avg_train, avg_test = get_average_losses(logger, logger.current_step)
    
    # Calculate overall improvement
    first_step_avg = get_average_losses(logger, 1)
    if first_step_avg != (0.0, 0.0)
        overall_train_improvement = (first_step_avg[1] - avg_train) / (abs(first_step_avg[1]) + 1e-10) * 100
        overall_val_improvement = (first_step_avg[2] - avg_test) / (abs(first_step_avg[2]) + 1e-10) * 100
        
        @printf("Training Loss: %.4f (%.2f%% improvement)\n", avg_train, overall_train_improvement)
        @printf("Validation Loss: %.4f (%.2f%% improvement)\n", avg_test, overall_val_improvement)
    else
        @printf("Training Loss: %.4f\n", avg_train)
        @printf("Validation Loss: %.4f\n", avg_test)
    end
    
    # Print log file location
    println("Detailed logs saved to: $(logger.log_dir)")
    
    # Save final summary to file
    save_model_state(logger)
end

"""
    save_model_state(logger)

Save the current state of the model and training progress to a summary file.
"""
function save_model_state(logger::ModelLogger)
    summary_file = joinpath(logger.log_dir, "summary.txt")
    
    open(summary_file, "w") do io
        println(io, "=== $(logger.name) Training Summary ===")
        println(io, "Completed at: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
        println(io, "Total time steps: $(logger.total_steps)")
        println(io, "Total modes: $(logger.modes)")
        
        # Add metadata
        println(io, "\n=== Metadata ===")
        for (key, value) in logger.metadata
            println(io, "$key: $value")
        end
        
        # Add final losses
        println(io, "\n=== Final Performance ===")
        avg_train, avg_test = get_average_losses(logger, logger.current_step)
        println(io, "Average Training Loss: $avg_train")
        println(io, "Average Validation Loss: $avg_test")
        
        # Add best performance per mode
        println(io, "\n=== Best Performance by Mode ===")
        for j in 1:logger.modes
            best_val_loss = Inf
            best_step = 0
            
            for n in 1:logger.total_steps
                if !isempty(logger.loss_history[n][j])
                    _, val_loss = last(logger.loss_history[n][j])
                    if val_loss < best_val_loss
                        best_val_loss = val_loss
                        best_step = n
                    end
                end
            end
            
            if best_step > 0
                println(io, "Mode $j: Best val loss = $best_val_loss at time step $best_step")
            end
        end
    end
end

"""
    save_training_metrics(logger, file_path=nothing)

Save detailed training metrics to a CSV file.
If file_path is not provided, saves to logger.log_dir/full_training_metrics.csv
"""
function save_training_metrics(logger::ModelLogger, file_path=nothing)
    if isnothing(file_path)
        file_path = joinpath(logger.log_dir, "full_training_metrics.csv")
    end
    
    # Prepare data
    rows = []
    
    for n in 1:logger.total_steps
        for j in 1:logger.modes
            if !isempty(logger.loss_history[n][j])
                for (i, (train_loss, val_loss)) in enumerate(logger.loss_history[n][j])
                    push!(rows, (
                        time_step = n,
                        mode = j,
                        iteration = i,
                        train_loss = train_loss,
                        val_loss = val_loss
                    ))
                end
            end
        end
    end
    
    # Save to CSV
    if !isempty(rows)
        df = DataFrame(rows)
        CSV.write(file_path, df)
        println("Detailed metrics saved to: $file_path")
    else
        @warn "No training metrics available to save"
    end
end

"""
    export_to_csv(logger, metrics=:all)

Export specified metrics to CSV files for further analysis.
- metrics can be :all, :loss_curve, :improvement, :best_values
"""
function export_to_csv(logger::ModelLogger, metrics=:all)
    if metrics ∈ [:all, :loss_curve]
        # Export average losses across time steps
        df = DataFrame(
            time_step = Int[],
            avg_train_loss = Float64[],
            avg_val_loss = Float64[]
        )
        
        for n in 1:logger.total_steps
            avg_train, avg_val = get_average_losses(logger, n)
            if avg_train != 0.0 || avg_val != 0.0
                push!(df, (n, avg_train, avg_val))
            end
        end
        
        if !isempty(df)
            CSV.write(joinpath(logger.log_dir, "loss_curve.csv"), df)
        end
    end
    
    if metrics ∈ [:all, :improvement]
        # Export improvement rates
        df = DataFrame(
            time_step = Int[],
            mode = Int[],
            rel_improvement = Float64[]
        )
        
        for n in 2:logger.total_steps
            for j in 1:logger.modes
                if !isempty(logger.loss_history[n][j]) && !isempty(logger.loss_history[n-1][j])
                    _, current_val = last(logger.loss_history[n][j])
                    _, prev_val = last(logger.loss_history[n-1][j])
                    rel_imp = (prev_val - current_val) / (abs(prev_val) + 1e-10) * 100
                    push!(df, (n, j, rel_imp))
                end
            end
        end
        
        if !isempty(df)
            CSV.write(joinpath(logger.log_dir, "improvement_rates.csv"), df)
        end
    end
    
    if metrics ∈ [:all, :best_values]
        # Export best values per mode
        df = DataFrame(
            mode = Int[],
            best_val_loss = Float64[],
            at_time_step = Int[]
        )
        
        for j in 1:logger.modes
            best_val_loss = Inf
            best_step = 0
            
            for n in 1:logger.total_steps
                if !isempty(logger.loss_history[n][j])
                    _, val_loss = last(logger.loss_history[n][j])
                    if val_loss < best_val_loss
                        best_val_loss = val_loss
                        best_step = n
                    end
                end
            end
            
            if best_step > 0
                push!(df, (j, best_val_loss, best_step))
            end
        end
        
        if !isempty(df)
            CSV.write(joinpath(logger.log_dir, "best_performances.csv"), df)
        end
    end
    
    println("Exported metrics to: $(logger.log_dir)")
end

end # module
