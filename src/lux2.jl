using Lux, ADTypes, MLUtils, Optimisers, Zygote, Random, Statistics, Printf
using ProgressMeter
import SimpleChains: static
# Add packages for plotting and file operations
using CairoMakie, Dates

function loss_function(model, ps, st, data)
    y_pred, st = model(data[1], ps, st)
    loss = MSE_loss(y_pred, data[2])
    return loss, st, ()
end

function MSE_loss(y_pred, y_true)
    return mean(abs2, y_pred .- y_true)
end

function huber_loss(y_pred, y_true; delta=1.0)
    error = y_pred .- y_true
    condition = abs.(error) .<= delta
    return mean(ifelse.(condition, 0.5 .* error.^2, delta .* (abs.(error) .- 0.5 * delta)))
end

function compute_validation_loss(model, ps, st, val_data; batch_size=1024)
    if length(val_data[1]) <= batch_size
        loss, _, _ = loss_function(model, ps, st, val_data)
        return loss
    end
    
    val_loader = DataLoader(val_data; batchsize=batch_size, shuffle=false)
    total_loss = 0.0
    count = 0
    
    for batch in val_loader
        loss, _, _ = loss_function(model, ps, st, batch)
        total_loss += loss
        count += 1
    end
    
    return total_loss / count
end

function copy_parameters!(dest_state::Lux.Training.TrainState, source_state::Lux.Training.TrainState)
    function copy_nested!(dest, source)
        for k in keys(source)
            if source[k] isa NamedTuple
                copy_nested!(dest[k], source[k])
            else
                dest[k] .= source[k]
            end
        end
    end

    copy_nested!(dest_state.parameters, source_state.parameters)
    return dest_state
end

function create_log_directory(n, j; base_dir="logs")
    # First ensure the base directory exists
    if !isdir(base_dir)
        @info "Creating base directory: $base_dir"
        mkpath(base_dir)
    end
    
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    log_dir = joinpath(base_dir, "model_$(n)_$(j)_$timestamp")
    
    # Create the specific log directory with error handling
    try
        mkpath(log_dir)
        @info "Created log directory: $log_dir"
    catch e
        @warn "Failed to create directory: $log_dir"
        @warn "Error: $e"
        # Try to create an alternative directory in the current working directory
        alt_dir = joinpath(".", "logs_$(timestamp)")
        @info "Attempting to create alternative directory: $alt_dir"
        mkpath(alt_dir)
        log_dir = alt_dir
    end
    
    return log_dir
end

function update_plots(log_history, log_dir)
    # Ensure the directory exists
    if !isdir(log_dir)
        mkpath(log_dir)
    end

    # Create and save loss plot
    fig1 = Figure(size= (800, 600))
    ax1 = Axis(fig1[1, 1], 
               title = "Loss Evolution", 
               xlabel = "Epoch", 
               ylabel = "Loss")
    
    epochs = 1:length(log_history[:train_loss])
    
    lines!(ax1, epochs, log_history[:train_loss], 
           color = :blue, linewidth = 2, 
           label = "Training Loss")
    lines!(ax1, epochs, log_history[:val_loss], 
           color = :red, linewidth = 2, 
           label = "Validation Loss")
    
    axislegend(ax1)
    
    # Save current plot to log directory
    try
        save(joinpath(log_dir, "loss_plot.png"), fig1)
    catch e
        @warn "Failed to save loss plot: $e"
    end
    
    # Create and save relative improvement plot if we have at least 2 epochs
    if length(log_history[:train_loss]) > 1
        rel_train_imp = [100 * (log_history[:train_loss][i-1] - log_history[:train_loss][i]) / max(1e-10, log_history[:train_loss][i-1]) 
                        for i in 2:length(log_history[:train_loss])]
        rel_val_imp = [100 * (log_history[:val_loss][i-1] - log_history[:val_loss][i]) / max(1e-10, log_history[:val_loss][i-1]) 
                      for i in 2:length(log_history[:val_loss])]
        
        fig2 = Figure(size= (800, 600))
        ax2 = Axis(fig2[1, 1], 
                   title = "Relative Improvement", 
                   xlabel = "Epoch", 
                   ylabel = "Improvement %")
        
        imp_epochs = 2:length(log_history[:train_loss])
        
        lines!(ax2, imp_epochs, rel_train_imp, 
               color = :blue, linewidth = 2, 
               label = "Train Improvement %")
        lines!(ax2, imp_epochs, rel_val_imp, 
               color = :red, linewidth = 2, 
               label = "Val Improvement %")
        
        axislegend(ax2)
        
        try
            save(joinpath(log_dir, "improvement_plot.png"), fig2)
        catch e
            @warn "Failed to save improvement plot: $e"
        end
    end
    
    return nothing
end

function switching_constrained_loss(model, ps, st, data, payoffmodel, n, j, learning_model, exp_params)

    return total_loss, st_out, ()
end

function lux_main!(tstate, vjp, data, epochs; 
    patience=50, 
    batchsize=1024,
    verbose=true,
    improvement_threshold=0.99,
    log_dir=nothing,
    model_name="neural_network",
    time_step=0,
    mode=0,
    best_val = Inf)

    epochs_since_improvement = 0
    
    # Create a ModelLogger if we have TrainingLogger available
    logger = nothing
    if isdefined(Main, :TrainingLogger) && time_step > 0 && mode > 0
        logger = Main.TrainingLogger.ModelLogger(
            "$(model_name)_ts$(time_step)_mode$(mode)", 
            epochs, 
            1, 
            log_dir=log_dir,
            metadata=Dict(
                "time_step" => time_step,
                "mode" => mode,
                "batchsize" => batchsize,
                "patience" => patience,
                "gradient_enhancement" => use_gradient_enhancement
            )
        )
    end
    
    train_data, val_data = splitobs(data, at=0.8)
    batch_data = DataLoader(train_data; batchsize=batchsize)
    prog = verbose ? Progress(epochs, desc="Training: ", dt=1.0) : nothing
    
    # Create containers for logging history
    log_history = Dict(
        :train_loss => Float64[],
        :val_loss => Float64[],
        :best_val => Float64[],
        :gradient_norm => Float64[]
    )
    val_loss = Inf
    
    # Create log summary file if we're not using ModelLogger
    if log_dir !== nothing && logger === nothing
        log_file = joinpath(log_dir, "training_log.csv")
        try
            open(log_file, "w") do io
                println(io, "epoch,train_loss,val_loss,best_val,patience_left,gradient_norm")
            end
        catch e
            @warn "Failed to create log file: $log_file"
            @warn "Error: $e"
        end
    end
    
    for epoch in 1:epochs
        epoch_loss = 0.0
        batch_count = 0
        total_grad_norm = 0.0

        for batch in batch_data
            loss_fn = loss_function
            grads, loss, stats, tstate = Lux.Training.compute_gradients(
                vjp, loss_fn, batch, tstate)
            
            # Gradient norm calculation disabled for performance
            
            epoch_loss += loss
            batch_count += 1
            
            tstate = Lux.Training.apply_gradients(tstate, grads)
        end
        
        avg_loss = epoch_loss / batch_count
        avg_grad_norm = 0.0  # Gradient norm calculation disabled
        val_loss = compute_validation_loss(tstate.model, tstate.parameters, 
                                          tstate.states, val_data)
        
        # Update logging history
        push!(log_history[:train_loss], avg_loss)
        push!(log_history[:val_loss], val_loss)
        push!(log_history[:best_val], best_val)
        push!(log_history[:gradient_norm], 0.0)
        
        # Early stopping logic
        if val_loss < best_val * improvement_threshold
            best_val = val_loss
            epochs_since_improvement = 0
        else 
            epochs_since_improvement += 1
            if epochs_since_improvement > patience
                verbose && println("Early stopping at epoch $epoch")
                break
            end
        end
        
        # Log with ModelLogger if available
        if logger !== nothing
            Main.TrainingLogger.log_losses!(logger, epoch, 1, avg_loss, val_loss, verbose=verbose)
        end
        
        # Update traditional log file if not using ModelLogger
        if log_dir !== nothing && logger === nothing
            try
                open(joinpath(log_dir, "training_log.csv"), "a") do io
                    println(io, "$epoch,$avg_loss,$val_loss,$best_val,$(patience - epochs_since_improvement),$avg_grad_norm")
                end
            catch e
                @warn "Failed to update log file: $e"
            end
            
            # Update and save plots every few epochs
            if epoch % 5 == 0 || epoch == 1
                update_plots(log_history, log_dir)
            end
        end
        
        if verbose && prog !== nothing
            next!(prog, showvalues=[
                (:epoch, epoch),
                (:train_loss, round(avg_loss, digits=6)),
                (:val_loss, round(val_loss, digits=6)),
                (:best_val, round(best_val, digits=6)),
                (:patience_left, patience - epochs_since_improvement),
                (:grad_norm, round(avg_grad_norm, digits=4))
            ])
        end
    end
    
    # Final log actions
    if log_dir !== nothing
        if logger !== nothing
            # Generate comprehensive report if using ModelLogger
            if isdefined(Main, :TrainingPlots)
                Main.TrainingPlots.generate_training_report(logger)
            else
                Main.TrainingLogger.print_summary(logger)
                Main.TrainingLogger.save_model_state(logger)
            end
        else
            # Traditional plot update
            update_plots(log_history, log_dir)
            
            # Create a basic summary file
            try
                open(joinpath(log_dir, "summary.txt"), "w") do io
                    println(io, "Training completed at: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
                    println(io, "Final validation loss: $val_loss")
                    println(io, "Best validation loss: $best_val")
                    println(io, "Total epochs run: $epochs")
                    println(io, "Time step: $time_step, Mode: $mode")
                end
            catch e
                @warn "Failed to save summary: $e"
            end
        end
    end
    
    return tstate, val_loss
end

function normalize_data(X; mean_val=nothing, std_val=nothing)
    if isnothing(mean_val) || isnothing(std_val)
        mean_val = Statistics.mean(X, dims=2)
        std_val = Statistics.std(X, dims=2)
        std_val[std_val .== 0] .= 1
    end
    mean_val = 0
    std_val = 1
    
    X_norm = (X .- mean_val) ./ std_val
    return X_norm, mean_val, std_val
end

function better_normalize_data(X)
    # Calculate robust stats
    mean_val = mean(X, dims=2)
    std_val = std(X, dims=2)
    
    # Prevent division by zero
    std_val[std_val .< 1e-5] .= 1.0
    mean_val = 0
    std_val = 1
    # Normalize
    X_norm = (X .- mean_val) ./ std_val
    
    # Clip extreme values
    X_norm = clamp.(X_norm, -5.0, 5.0)
    
    return X_norm, mean_val, std_val
end

function lux_fit!(learning_model, X_train, Y_train, n, j; 
    seed=123,
    epochs=400,
    batchsize=512,
    patience=50,
    verbose=true,
    log_dir_base="logs",
    epochs_ratio=1.0)
    
    # Set seed for reproducibility
    Random.seed!(seed)
    
    Y_train = reshape(Y_train, 1, :)
    tstate = learning_model.model[n, j][1]
    N = size(learning_model.model, 1) - 1

    # Copy parameters from next state if not at the final time step
    if n < N
        next_tstate, _ = learning_model.model[n+1, j]
        previous_params = next_tstate.parameters.params
        tstate.parameters.params .= previous_params .+ 0.005 * randn(Float32,size(previous_params))
    end
    
    # Increase epochs for critical time steps closer to the end
    epoch_factor = 1.0 + 10.0 * (1.0 - n/N)^2
    epochs = Int(round(epochs * epoch_factor))
    patience = 50 #+ Int(round(30 * (1.0 - n/N)^2))
    # Normalize input data
    X_norm, x_mean_val, x_std_val = normalize_data(X_train)
    data = (X_norm, Y_train)

    vjp = AutoZygote()
    
    # Create a log directory for this training run
    log_dir = create_log_directory(n, j, base_dir=log_dir_base)
    verbose && @info "Logs will be saved to: $log_dir"
    
    # Get model name for logging
    model_name = learning_model.name
    
    improvement_threshold = 0.999
    # Pass additional information to the training function
    tstate, final_val_loss = lux_main!(
        tstate,
        vjp,
        data,
        epochs;
        patience=patience,
        batchsize=batchsize,
        verbose=verbose,
        improvement_threshold=improvement_threshold,
        log_dir=log_dir,
        model_name=model_name,
        time_step=n,
        mode=j
    )

    # Store updated state and normalization parameters
    learning_model.model[n, j] = (tstate, (x_mean_val, x_std_val))
    
    return nothing
end