module TrainingPlots

using CairoMakie
using Statistics
using DataFrames
using CSV
using Dates

export plot_training_history, plot_loss_curves, plot_improvement_curve
export generate_training_report, plot_mode_performance, plot_all_metrics

"""
    plot_training_history(logger, save_path=nothing)

Plot the training and validation loss history for a model.
If save_path is provided, saves the plot to that location.
"""
function plot_training_history(logger, save_path=nothing)
    n_steps = logger.total_steps
    n_modes = logger.modes
    
    # Prepare data for plotting
    time_steps = collect(1:n_steps)
    train_losses = zeros(n_steps, n_modes)
    test_losses = zeros(n_steps, n_modes)
    
    for n in 1:n_steps
        for j in 1:n_modes
            if !isempty(logger.loss_history[n][j])
                train_loss, test_loss = last(logger.loss_history[n][j])
                train_losses[n, j] = train_loss
                test_losses[n, j] = test_loss
            end
        end
    end
    
    # Calculate mean losses across modes
    mean_train = vec(mean(train_losses, dims=2))
    mean_test = vec(mean(test_losses, dims=2))
    
    # Create the plot
    fig = Figure(size=(800, 600))
    
    # Plot showing mean losses only
    ax = Axis(fig[1, 1],
        title="Training Progress - $(logger.name)",
        xlabel="Time Step",
        ylabel="Loss")
    
    lines!(ax, time_steps, mean_train, label="Mean Training Loss", color=:blue)
    lines!(ax, time_steps, mean_test, label="Mean Validation Loss", color=:red)
    axislegend(ax)
    
    # Save if path provided
    if !isnothing(save_path)
        save(save_path, fig)
    end
    
    return fig
end

"""
    plot_loss_curves(logger, modes=:all; save_path=nothing)

Plot detailed loss curves for specific modes or all modes.
If modes is :all, plots for all modes. Otherwise, provide an array of mode indices.
"""
function plot_loss_curves(logger, modes=:all; save_path=nothing)
    n_steps = logger.total_steps
    
    # Determine which modes to plot
    plot_modes = modes == :all ? (1:logger.modes) : modes
    
    # Create figure with a grid of subplots
    n_plots = length(plot_modes)
    n_cols = min(3, n_plots)
    n_rows = ceil(Int, n_plots / n_cols)
    
    fig = Figure(size=(800 * n_cols, 600 * n_rows))
    
    for (i, j) in enumerate(plot_modes)
        row = ceil(Int, i / n_cols)
        col = ((i - 1) % n_cols) + 1
        
        ax = Axis(fig[row, col],
            title="Mode $j Loss Curve",
            xlabel="Time Step",
            ylabel="Loss")
        
        # Collect losses for this mode
        time_steps = Int[]
        train_losses = Float64[]
        test_losses = Float64[]
        
        for n in 1:n_steps
            if !isempty(logger.loss_history[n][j])
                push!(time_steps, n)
                train_loss, test_loss = last(logger.loss_history[n][j])
                push!(train_losses, train_loss)
                push!(test_losses, test_loss)
            end
        end
        
        if !isempty(time_steps)
            # Plot loss curves
            lines!(ax, time_steps, train_losses, label="Training", color=:blue)
            lines!(ax, time_steps, test_losses, label="Validation", color=:red)
            axislegend(ax)
        else
            text!(ax, 0.5, 0.5, text="No data for mode $j", 
                  align=(:center, :center), color=:black)
        end
    end
    
    # Save if path provided
    if !isnothing(save_path)
        save(save_path, fig)
    end
    
    return fig
end

"""
    plot_improvement_curve(logger; save_path=nothing)

Plot relative improvement rates across time steps.
"""
function plot_improvement_curve(logger; save_path=nothing)
    # Prepare data
    time_steps = Int[]
    improvements = Float64[]
    
    # Calculate improvement for each time step (averaged across modes)
    for n in 2:logger.total_steps
        avg_prev = get_average_losses(logger, n-1)[2]  # Using validation loss
        avg_curr = get_average_losses(logger, n)[2]
        
        if avg_prev != 0.0 && avg_curr != 0.0
            rel_imp = (avg_prev - avg_curr) / (abs(avg_prev) + 1e-10) * 100
            push!(time_steps, n)
            push!(improvements, rel_imp)
        end
    end
    
    # Create the plot
    fig = Figure(size=(800, 600))
    
    ax = Axis(fig[1, 1],
        title="Relative Improvement Rate - $(logger.name)",
        xlabel="Time Step",
        ylabel="Improvement (%)")
    
    # Add zero line for reference
    hlines!(ax, 0, color=:gray, linestyle=:dash)
    
    # Plot improvement curve
    if !isempty(time_steps)
        stem!(ax, time_steps, improvements, color=:blue)
        scatter!(ax, time_steps, improvements, color=:blue, markersize=6)
    end
    
    # Save if path provided
    if !isnothing(save_path)
        save(save_path, fig)
    end
    
    return fig
end

"""
    plot_mode_performance(logger; save_path=nothing)

Plot a bar chart showing the best validation loss for each mode.
"""
function plot_mode_performance(logger; save_path=nothing)
    # Collect best loss for each mode
    modes = collect(1:logger.modes)
    best_losses = Float64[]
    
    for j in modes
        best_loss = Inf
        for n in 1:logger.total_steps
            if !isempty(logger.loss_history[n][j])
                _, val_loss = last(logger.loss_history[n][j])
                best_loss = min(best_loss, val_loss)
            end
        end
        
        push!(best_losses, best_loss == Inf ? 0.0 : best_loss)
    end
    
    # Create the plot
    fig = Figure(size=(800, 600))
    
    ax = Axis(fig[1, 1],
        title="Best Performance by Mode - $(logger.name)",
        xlabel="Mode",
        ylabel="Best Validation Loss",
        xticks=(modes, string.(modes)))
    
    barplot!(ax, modes, best_losses, color=:blue)
    
    # Add data labels
    for (i, val) in enumerate(best_losses)
        if val > 0
            text!(ax, i, val + maximum(best_losses) * 0.02, 
                  text="$(round(val, digits=4))", 
                  align=(:center, :bottom), 
                  rotation=90, 
                  fontsize=10)
        end
    end
    
    # Save if path provided
    if !isnothing(save_path)
        save(save_path, fig)
    end
    
    return fig
end

"""
    plot_all_metrics(logger, output_dir=nothing)

Generate all available plots and save them to the specified directory.
If output_dir is not provided, uses logger.log_dir.
"""
function plot_all_metrics(logger, output_dir=nothing)
    # Use logger's directory if none provided
    plot_dir = isnothing(output_dir) ? logger.log_dir : output_dir
    if !isdir(plot_dir)
        mkpath(plot_dir)
    end
    
    # Generate overview plot
    plot_path = joinpath(plot_dir, "loss_overview.pdf")
    plot_training_history(logger, plot_path)
    
    # Generate detailed mode plots for a subset
    if logger.modes <= 6
        # Plot all modes if there are few
        modes_to_plot = collect(1:logger.modes)
    else
        # Sample a few modes for larger models
        modes_to_plot = [1, 2, div(logger.modes, 2), logger.modes-1, logger.modes]
    end
    
    plot_path = joinpath(plot_dir, "mode_details.pdf")
    plot_loss_curves(logger, modes_to_plot, save_path=plot_path)
    
    # Generate improvement rate plot
    plot_path = joinpath(plot_dir, "improvement_rate.pdf")
    plot_improvement_curve(logger, save_path=plot_path)
    
    # Generate mode performance comparison
    plot_path = joinpath(plot_dir, "mode_performance.pdf")
    plot_mode_performance(logger, save_path=plot_path)
    
    # Return path to all generated plots
    return plot_dir
end

"""
    generate_training_report(logger, output_dir=nothing)

Generate a comprehensive training report with all metrics and plots.
If output_dir is not provided, uses logger.log_dir.
"""
function generate_training_report(logger, output_dir=nothing)
    # Use logger's directory if none provided
    report_dir = isnothing(output_dir) ? logger.log_dir : output_dir
    if !isdir(report_dir)
        mkpath(report_dir)
    end
    
    # Create subdirectory for plots
    plots_dir = joinpath(report_dir, "plots")
    if !isdir(plots_dir)
        mkpath(plots_dir)
    end
    
    # Generate all plots
    plot_all_metrics(logger, plots_dir)
    
    # Generate CSV exports
    csv_path = joinpath(report_dir, "metrics")
    if !isdir(csv_path)
        mkpath(csv_path)
    end
    
    # Export CSV data
    avg_losses = DataFrame(
        time_step = Int[],
        avg_train_loss = Float64[],
        avg_val_loss = Float64[]
    )
    
    for n in 1:logger.total_steps
        avg_train, avg_val = get_average_losses(logger, n)
        if avg_train != 0.0 || avg_val != 0.0
            push!(avg_losses, (n, avg_train, avg_val))
        end
    end
    
    if !isempty(avg_losses)
        CSV.write(joinpath(csv_path, "average_losses.csv"), avg_losses)
    end
    
    # Create summary report HTML
    report_path = joinpath(report_dir, "training_report.html")
    
    open(report_path, "w") do io
        write(io, """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Training Report: $(logger.name)</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2, h3 { color: #333366; }
                .container { max-width: 1200px; margin: 0 auto; }
                .metrics { margin: 20px 0; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .plot { margin: 20px 0; text-align: center; }
                .plot img { max-width: 100%; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Training Report: $(logger.name)</h1>
                <p>Generated on: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))</p>
                
                <div class="metrics">
                    <h2>Training Summary</h2>
                    <table>
                        <tr><th>Parameter</th><th>Value</th></tr>
                        <tr><td>Model Name</td><td>$(logger.name)</td></tr>
                        <tr><td>Time Steps</td><td>$(logger.total_steps)</td></tr>
                        <tr><td>Modes</td><td>$(logger.modes)</td></tr>
                    </table>
                </div>
                
                <div class="metrics">
                    <h2>Final Performance</h2>
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
        """)
        
        # Add final metrics
        avg_train, avg_test = get_average_losses(logger, logger.current_step)
        first_step_avg = get_average_losses(logger, 1)
        
        if first_step_avg != (0.0, 0.0)
            overall_train_improvement = (first_step_avg[1] - avg_train) / (abs(first_step_avg[1]) + 1e-10) * 100
            overall_val_improvement = (first_step_avg[2] - avg_test) / (abs(first_step_avg[2]) + 1e-10) * 100
            
            write(io, """
                        <tr><td>Final Training Loss</td><td>$(round(avg_train, digits=6))</td></tr>
                        <tr><td>Final Validation Loss</td><td>$(round(avg_test, digits=6))</td></tr>
                        <tr><td>Overall Training Improvement</td><td>$(round(overall_train_improvement, digits=2))%</td></tr>
                        <tr><td>Overall Validation Improvement</td><td>$(round(overall_val_improvement, digits=2))%</td></tr>
            """)
        else
            write(io, """
                        <tr><td>Final Training Loss</td><td>$(round(avg_train, digits=6))</td></tr>
                        <tr><td>Final Validation Loss</td><td>$(round(avg_test, digits=6))</td></tr>
            """)
        end
        
        write(io, """
                    </table>
                </div>
                
                <div class="plots">
                    <h2>Training Plots</h2>
                    
                    <div class="plot">
                        <h3>Loss Overview</h3>
                        <img src="plots/loss_overview.pdf" alt="Loss Overview">
                    </div>
                    
                    <div class="plot">
                        <h3>Improvement Rate</h3>
                        <img src="plots/improvement_rate.pdf" alt="Improvement Rate">
                    </div>
                    
                    <div class="plot">
                        <h3>Mode Performance</h3>
                        <img src="plots/mode_performance.pdf" alt="Mode Performance">
                    </div>
                    
                    <div class="plot">
                        <h3>Mode Details</h3>
                        <img src="plots/mode_details.pdf" alt="Mode Details">
                    </div>
                </div>
            </div>
        </body>
        </html>
        """)
    end
    
    return report_path
end

end # module
