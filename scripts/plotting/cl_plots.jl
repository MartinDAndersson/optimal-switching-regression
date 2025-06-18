module cl_plots
#current_dir = pwd()*"/scripts/"
using DrWatson                           # Project organization and reproducibility
@quickactivate  
using CairoMakie
using StatsBase: quantile
using Makie
using Random
using Statistics
import .Threads
#using Threads
using DataFrames
using StatsBase: quantile
using Makie
using ColorSchemes
using KernelDensity
using CairoMakie
using Statistics
using LinearAlgebra
using Colors: RGB, gray
using ColorSchemes
using StatsBase

export calculate_value_function
export calculate_greedy_value_matrix
export determine_optimal_strategy
export compute_strategy_for_value
export plot_switching_strategies_and_values
export plot_strategy_analysis
export prepare_strategy_analysis
export plot_value_prediction_error
export plot_value_functions_surface_2d
export plot_switching_boundaries_evolution
export plot_switching_boundaries_comparison
export apost_grid
export plot_value_differences
export create_value_grid_timepoint
export create_full_value_grid

include(srcdir("plots_common.jl"))

function create_strategy_overview_figure(strategies, times, strategy_names;
                                       figsize=(350, 550))
    fig = Figure(size=figsize, fontsize=10, backgroundcolor=:white)
    
    # Create a professional color palette using ColorSchemes
    mode_colors = [
        ColorSchemes.Paired_3[1],  # Mode 1 (Off)
        ColorSchemes.Paired_3[2],  # Mode 2 (Half)
        ColorSchemes.Paired_3[3]   # Mode 3 (Full)
    ]
    
    # Define secondary colors for UI elements using proper attribute names
    text_color = RGB(0.2, 0.2, 0.2)        # Near black for maximum contrast
    grid_color = RGB(0.8, 0.8, 0.8)        # Light gray for subtle gridlines
    spine_color = RGB(0.3, 0.3, 0.3)       # Darker gray for clear boundaries
    
    mode_labels = ["Off", "Half", "Full"]
    n_strategies = length(strategies)
    plot_height = 35
    bands = []
    
    gl = GridLayout(fig[1, 1], 
                   margins=(40, 20, 25, 25),
                   halign=:left,
                   valign=:top)
    
    Label(gl[1, 1:2], "Operational Mode Distribution",
          font=:bold, fontsize=12, padding=(0, 0, 15, 0),
          color=text_color)
    
    for (i, (strategy, name)) in enumerate(zip(strategies, strategy_names))
        Label(gl[i+1, 1], name,
              halign=:right,
              valign=:center,
              fontsize=10,
              padding=(0, 15, 0, 0),
              color=text_color)
        
        ax = Axis(gl[i+1, 2],
                 height=plot_height,
                 xlabelvisible=i == n_strategies,
                 xticklabelsize=9,
                 xticks=LinearTicks(4),
                 yticks=LinearTicks(1),
                 spinewidth=0.5,
                 xgridvisible=false,
                 ygridvisible=false,
                 yaxisposition=:right,
                 leftspinecolor=spine_color,
                 rightspinecolor=spine_color,
                 bottomspinecolor=spine_color,
                 topspinecolor=spine_color,
                 xgridcolor=grid_color,
                 ygridcolor=grid_color)
        
        i != n_strategies && hidexdecorations!(ax)
        
        mode_props = compute_mode_proportions(strategy, length(times), 
                                            size(strategy, 2), 4)
        stack_bottom = zeros(length(times))
        
        for mode in 1:3
            stack_top = stack_bottom .+ mode_props[:, mode]
            b = band!(ax, times, stack_bottom, stack_top,
                     color=(mode_colors[mode], 0.9))
            i == 1 && push!(bands, b)
            stack_bottom = stack_top
        end
        
        limits!(ax, extrema(times)..., 0, 1)
    end
    
    Legend(gl[n_strategies+2, 1:2],
           bands, mode_labels,
           orientation=:horizontal,
           nbanks=2,
           framecolor=(spine_color, 0.2),
           framevisible=true,
           titleposition=:left,
           title="Modes:",
           labelsize=9,
           margin=(15, 15, 10, 10),
           patchsize=(25, 10),
           rowgap=8,
           colgap=15)
    
    colsize!(gl, 1, Auto(1.2))
    colsize!(gl, 2, Relative(0.75))
    rowgap!(gl, 10)
    
    return fig
end
# function plot_value_functions_surface_2d(Vs::Vector, time, mode, samples)
#     @assert size(samples, 1) == 2 "Samples should have 2 dimensions"
#     @assert size(samples, 3) > 0 "Samples should have non-zero size in the third dimension"

#     # Determine range from all sample points
#     x1_samples, x2_samples = samples[1, :, :], samples[2, :, :]
#     x1_min, x1_max = minimum(x1_samples), maximum(x1_samples)
#     x2_min, x2_max = minimum(x2_samples), maximum(x2_samples)
    
#     x1_range = range(x1_min, x1_max, length=100)
#     x2_range = range(x2_min, x2_max, length=100)
    
#     # Calculate global z-limits
#     z_min = Inf
#     z_max = -Inf
#     for V in Vs
#         z_values = [V([x1; x2], time, mode)[1] for x1 in x1_range, x2 in x2_range]
#         z_min = min(z_min, minimum(z_values))
#         z_max = max(z_max, maximum(z_values))
#     end
    
#     fig = Figure(size = (1200, 800))
    
#     for i in 1:6
#         row = div(i-1, 3) + 1
#         col = (i-1) % 3 + 1
        
#         ax = Axis3(fig[row, col], 
#                    xlabel = "Electricity Price", 
#                    ylabel = "Gas Price", 
#                    zlabel = "Value",
#                    title = i <= length(Vs) ? "Model $(Vs[i].name)" : "")
        
#         if i <= length(Vs)
#             V = Vs[i]
#             surface!(ax, x1_range, x2_range, 
#                      (x1, x2) -> V([x1; x2], time, mode)[1], 
#                      color=:viridis, transparency=true, alpha=0.6)
            
#             # Plot sample points
#             for j in 1:size(samples, 3)
#                 scatter!(ax, samples[1, :, j], samples[2, :, j], 
#                          [V([samples[1, k, j]; samples[2, k, j]], time, mode)[1] for k in 1:size(samples, 2)],
#                          color = :black, markersize = 2, alpha = 0.5, marker = :circle)
#             end
            
#             # Set consistent z-axis limits
#             zlims!(ax, z_min, z_max)
#         end
#     end
    
#     Label(fig[0, :], "Value Functions for Mode $mode at t = $time", fontsize = 20)
    
#     fig
# end

# function plot_value_function_evolution_2d(V, time_points, mode, samples)
#     @assert size(samples, 1) == 2 "Samples should have 2 dimensions"
#     @assert size(samples, 2) == length(time_points) "Second dimension of samples should match the number of time points"

#     # Determine range from all sample points
#     x1_samples, x2_samples = samples[1, :, :], samples[2, :, :]
#     x1_min, x1_max = minimum(x1_samples), maximum(x1_samples)
#     x2_min, x2_max = minimum(x2_samples), maximum(x2_samples)
    
#     x1_range = range(x1_min, x1_max, length=100)
#     x2_range = range(x2_min, x2_max, length=100)
    
#     # Calculate global z-limits
#     z_min = Inf
#     z_max = -Inf
#     for time in time_points
#         z_values = [V([x1; x2], time, mode)[1] for x1 in x1_range, x2 in x2_range]
#         z_min = min(z_min, minimum(z_values))
#         z_max = max(z_max, maximum(z_values))
#     end
    
#     fig = Figure(size = (1200, 800))
    
#     for (i, time) in enumerate(time_points)
#         row = div(i-1, 3) + 1
#         col = (i-1) % 3 + 1
        
#         ax = Axis3(fig[row, col], 
#                    xlabel = "Electricity Price", 
#                    ylabel = "Gas Price", 
#                    zlabel = "Value",
#                    title = "t = $time")
        
#         surface!(ax, x1_range, x2_range, 
#                  (x1, x2) -> V([x1; x2], time, mode)[1], 
#                  color=:viridis, transparency=true, alpha=0.6)
        
#         # Plot sample points
#         scatter!(ax, samples[1, i, :], samples[2, i, :], 
#                  [V([samples[1, i, k]; samples[2, i, k]], time, mode)[1] for k in 1:size(samples, 3)],
#                  color = :black, markersize = 2, alpha = 0.5, marker = :circle)
        
#         # Set consistent z-axis limits
#         zlims!(ax, z_min, z_max)
#     end
    
#     Label(fig[0, :], "Evolution of Value Function for Model $(V.name), Mode $mode", fontsize = 20)
    
#     fig
# end

# function plot_comprehensive_switching_strategy(V, payoff, cost, sample_paths::Array{Float64,3}, times, dt)
#     @assert size(sample_paths, 1) == 2 "Sample paths should be a 2×N×M array"
#     @assert size(sample_paths, 2) == length(times) "Number of time points should match the second dimension of sample_paths"

#     num_paths = size(sample_paths, 3)
    
#     fig = Figure(size = (1200, 800))
    
#     # Prices plot
#     ax1 = Axis(fig[1, 1], xlabel = "Time", ylabel = "Log of Electricity Price")
#     ax1_right = Axis(fig[1, 1], ylabel = "Log of Gas Price", yaxisposition = :right)
#     hidexdecorations!(ax1_right, grid = false)
#     linkyaxes!(ax1, ax1_right)

#     # Actual strategy plot
#     ax2 = Axis(fig[2, 1], xlabel = "Time", ylabel = "Mode of strategy")

#     # Running payoff plot
#     ax3 = Axis(fig[3, 1], xlabel = "Time", ylabel = "Accumulated value")

#     colors = cgrad(:rainbow, num_paths, categorical=true)
#     Label(fig[0, 1:2], "Strategies using $(V.name)", fontsize = 20, font = :bold)

#     for path in 1:num_paths
#         # Plot prices
#         lines!(ax1, times, log.(sample_paths[1, :, path]), color = colors[path], label = path == 1 ? "Electricity Paths" : "")
#         lines!(ax1_right, times, log.(sample_paths[2, :, path]), color = colors[path], linestyle = :dash, label = path == 1 ? "Gas Paths" : "")

#         # Calculate and plot actual strategy and running payoff
#         actual_modes = zeros(Int, length(times))
#         running_payoff = zeros(length(times))
#         current_mode = 1
#         for (i, t) in enumerate(times)
#             state = sample_paths[:, i, path]
            
#             # Record the current mode
#             actual_modes[i] = current_mode
            
#             # Decide whether to switch based on value function
#             value_current = V(state, t, current_mode)[1]
#             best_mode = current_mode
#             best_value = value_current
#             for j in 1:3
#                 if j != current_mode
#                     switch_cost = cost(state, t)[current_mode, j]
#                     value_after_switch = V(state, t, j)[1] - switch_cost
#                     if value_after_switch > best_value
#                         best_value = value_after_switch
#                         best_mode = j
#                     end
#                 end
#             end
            
#             # Calculate payoff and update running payoff
#             current_payoff = payoff(state, t)[best_mode]*dt
#             if best_mode != current_mode
#                 switch_cost = cost(state, t)[current_mode, best_mode]
#                 running_payoff[i] = (i > 1 ? running_payoff[i-1] : 0) + current_payoff - switch_cost
#                 current_mode = best_mode
#             else
#                 running_payoff[i] = (i > 1 ? running_payoff[i-1] : 0) + current_payoff
#             end
#         end
        
#         # Plot actual strategy
#         stairs!(ax2, times, actual_modes, step = :post, color = colors[path], label = path == 1 ? "Strategies" : "")
        
#         # Plot running payoff
#         lines!(ax3, times, running_payoff, color = colors[path], label = path == 1 ? "Accumulated Values" : "")
#     end

#     # Adjust strategy plot y-axis
#     ax2.yticks = (1:3, string.(1:3))
#     ylims!(ax2, 0.5, 3.5)

#     # Add legends
#     Legend(fig[1, 2], [LineElement(color = :black), LineElement(color = :black, linestyle = :dash)],
#            ["Electricity Paths", "Gas Paths"], "Prices", framevisible = false)
#     Legend(fig[2, 2], [LineElement(color = :black)], ["Strategies"], "Modes", framevisible = false)
#     Legend(fig[3, 2], [LineElement(color = :black)], ["Accumulated Values"], "Payoffs", framevisible = false)
    
#     fig
# end


# function plot_switching_boundaries_2d_trajs(
#     V, 
#     cost::Function, 
#     payoff::Function, 
#     time::Real, 
#     samples::Matrix{Float64},
#     dt::Real
# )
#     @assert size(samples, 1) == 2 "Samples should have 2 dimensions"

#     # Determine the plot range (use percentiles to exclude outliers)
#     x_min, x_max = percentile(samples[1, :], [1, 99])
#     y_min, y_max = percentile(samples[2, :], [1, 99])
#     x_range = x_max - x_min
#     y_range = y_max - y_min
    
#     # Extend the range slightly beyond the sample range
#     electricity_price_range = range(x_min - 0.05 * x_range, x_max + 0.05 * x_range, length=100)
#     gas_price_range = range(y_min - 0.05 * y_range, y_max + 0.05 * y_range, length=100)
    
#     function optimal_mode_with_cost(electricity_price, gas_price, current_mode)
#         state = [electricity_price; gas_price]
#         current_payoff = payoff(state, time)[current_mode] * dt
#         value_current = V(state, time, current_mode)[1] #+ current_payoff
#         best_mode = current_mode
#         best_value = value_current
        
#         for new_mode in 1:3
#             if new_mode != current_mode
#                 switch_cost = cost(state, time)[current_mode, new_mode]
#                 new_payoff = payoff(state, time)[new_mode] * dt
#                 value_after_switch = V(state, time, new_mode)[1] - switch_cost #+ new_payoff
#                 if value_after_switch - best_value > 1e-3
#                     best_value = value_after_switch
#                     best_mode = new_mode
#                 end
#             end
#         end
        
#         return best_mode
#     end
    
#     optimal_mode = [optimal_mode_with_cost(x1, x2, 1) for x1 in electricity_price_range, x2 in gas_price_range]
    
#     fig = Figure(size = (1000, 600))
#     ax = Axis(fig[1, 1], 
#                xlabel = "Electricity Price", 
#                ylabel = "Gas Price",
#                title = "Optimal Switching Boundaries at t = $time")
    
#     colors = [:royalblue, :gold, :firebrick]
#     cmap = cgrad(colors, 3, categorical=true)
    
#     # Plot optimal mode boundaries
#     optimal_mode_plot = heatmap!(ax, electricity_price_range, gas_price_range, optimal_mode, 
#                                  colormap=cmap, colorrange=(0.5, 3.5))
    
#     if length(unique(optimal_mode)) > 1
#         contour!(ax, electricity_price_range, gas_price_range, optimal_mode, 
#                  levels=2:3, color=:black, linewidth=2)
#     end
    
#     # Plot scatter points
#     scatter!(ax, samples[1, :], samples[2, :], 
#              color = :black, markersize = 4, alpha = 0.5, label = "Samples")
    
#     ax.xgridvisible = true
#     ax.ygridvisible = true
    
#     # Add colorbar for optimal modes
#     Colorbar(fig[1, 3], optimal_mode_plot, label = "Optimal Mode", ticks = (1:3, string.(1:3)))
    
#     # Add legend for optimal modes
#     legend_labels = ["Mode 1", "Mode 2", "Mode 3", "Samples"]
#     elements = [MarkerElement(color = c, marker = :rect) for c in colors]
#     push!(elements, MarkerElement(color = :black, marker = :circle))
#     Legend(fig[1, 2], elements, legend_labels, "Modes and Samples")
    
#     # Set consistent limits for the plot
#     xlims!(ax, x_min, x_max)
#     ylims!(ax, y_min, y_max)
    
#     fig
# end



function plot_switching_boundaries_evolution(
    V, 
    cost::Function, 
    payoff::Function, 
    time_points, 
    samples::Array{Float64, 3},  # 2 x time_points x num_samples
    dt::Real,
    J::Int
)
    @assert size(samples, 1) == 2 "First dimension of samples should be 2"
    @assert size(samples, 2) == length(time_points) "Second dimension of samples should match the number of time points"

    # Determine the overall plot range across all time points
    x_min, x_max = percentile(vec(samples[1, :, :]), [1, 99])
    y_min, y_max = percentile(vec(samples[2, :, :]), [1, 99])
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Extend the range slightly beyond the sample range
    electricity_price_range = range(x_min - 0.05 * x_range, x_max + 0.05 * x_range, length=100)
    gas_price_range = range(y_min - 0.05 * y_range, y_max + 0.05 * y_range, length=100)

    function optimal_mode(electricity_price, gas_price, time)
        state = [electricity_price; gas_price]
        current_mode = 1  # Start with mode 1 as the current mode
        best_mode = choose_best_mode(V, state, time, current_mode, cost, dt,J)
        return best_mode
    end

    # Determine the layout
    n_plots = length(time_points)
    n_cols = 2#ceil(Int, sqrt(n_plots))
    n_rows = 3#ceil(Int, n_plots / n_cols)
    title_text = "Switching Boundaries Evolution \n $(V.name)"
    fig = Figure(size = (200 * n_cols, 60 + 150 * n_rows),title=title_text)
    
    Label(fig[0, 1:3], title_text,font=:bold,fontsize=22)
    colors = ColorSchemes.Paired_3#[:royalblue, :gold, :firebrick]
    cmap = cgrad(colors, 3, categorical=true)

    for (idx, time) in enumerate(time_points)
                row, col = fldmod1(idx, n_cols)

        ax = Axis(fig[fldmod1(idx, n_cols)...], 
                  xlabel = "Electricity Price", 
                  ylabel = "Gas Price",
                  title = "t = $time")

        optimal_mode_grid = [optimal_mode(x1, x2, time) for x1 in electricity_price_range, x2 in gas_price_range]
        # Only show x-labels for bottom row
        if row == n_rows
            ax.xlabel = "Electricity Price"
        else
            hidexdecorations!(ax, grid = true)  # Keep grid but hide labels
        end
                # Only show y-labels for leftmost column
        if col == 1
            ax.ylabel = "Gas Price"
        else
            hideydecorations!(ax, grid = true)  # Keep grid but hide labels
        end
        # Plot optimal mode boundaries
        optimal_mode_plot = heatmap!(ax, electricity_price_range, gas_price_range, optimal_mode_grid, 
                                     colormap=cmap, colorrange=(0.5, 3.5))
        
        if length(unique(optimal_mode_grid)) > 1
            contour!(ax, electricity_price_range, gas_price_range, optimal_mode_grid, 
                     levels=2:3, color=:black, linewidth=0.02)
        end
        
        # Plot scatter points for the current time point
        # scatter!(ax, samples[1, idx, :], samples[2, idx, :], 
        #          color = :black, markersize = 4, alpha = 0.3)
        
        ax.xgridvisible = true
        ax.ygridvisible = true
        
        # Set consistent limits for the plot
        xlims!(ax, x_min, x_max)
        ylims!(ax, y_min, y_max)
    end

    # Add colorbar for optimal modes
    Colorbar(fig[1:n_rows, n_cols+1], colormap=cmap, limits=(0.5, 3.5),
             ticks = (1:3, string.(1:3)), label = "Optimal Mode")

    # Add legend for optimal modes
    legend_labels = ["Off", "Half", "Full"]
    elements = [MarkerElement(color = c, marker = :rect) for c in colors]
    #push!(elements, MarkerElement(color = :black, marker = :circle))
    Legend(fig[n_rows+1, :], elements, legend_labels, "Modes and Samples", orientation = :horizontal)

    fig
end


function plot_and_save_switching_boundaries_for_models(mods, cost, payoff, time_points, trajs, dt)
    for mod in mods
        # Generate the plot
        fig = plot_switching_boundaries_evolution(mod, cost, payoff, time_points, trajs[:, time_points, 1:1000], dt)
        
        # Create a filename using the model's name
        filename = "Switching_boundaries_$(mod.name).png"
        
        # Save the figure
        save(filename, fig)
        
        println("Saved plot for $(mod.name) as $filename")
    end
end

function plot_switching_boundaries_comparison(
    Vs::Vector,
    cost::Function, 
    payoff::Function, 
    time_point::Real,
    samples::Array{Float64, 3},
    dt::Real,
    J::Int,
    initial_mode::Int
)
    # Data preprocessing 
    time_idx = time_point
    x_min, x_max = percentile(vec(samples[1, :, :]), [1, 99])
    y_min, y_max = percentile(vec(samples[2, :, :]), [1, 99])
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Grid setup
    electricity_price_range = range(x_min - 0.05 * x_range, x_max + 0.05 * x_range, length=100)
    gas_price_range = range(y_min - 0.05 * y_range, y_max + 0.05 * y_range, length=100)

    optimal_mode(V, electricity_price, gas_price) = begin
        state = [electricity_price; gas_price]
        choose_best_mode(V, state, time_point, initial_mode, cost, dt, J)
    end

    # Layout configuration
    n_cols = 2
    n_rows = 3
    fig = Figure(size=(200 * n_cols, 60 + 200 * n_rows))
    
    Label(fig[0, 1:2], "Switching Boundaries Comparison\nat t = $time_point", font=:bold)
    
    # Color configuration
    colors = ColorSchemes.Paired_3
    cmap = cgrad(colors, 3, categorical=true)

    # Create grid of plots
    for (idx, V) in enumerate(Vs)
        row, col = fldmod1(idx, n_cols)
        
        ax = Axis(fig[row, col],xticks=LinearTicks(4))
        
        # Handle axis decorations
        if row != n_rows
            hidexdecorations!(ax)
        else
            ax.xlabel = "Electricity Price"
        end
        
        if col != 1
            hideydecorations!(ax)
        else
            ax.ylabel = "Gas Price"
        end
        
        ax.title = V.name

        # Generate optimal mode grid
        optimal_mode_grid = [optimal_mode(V, x1, x2) 
                           for x1 in electricity_price_range, x2 in gas_price_range]

        # Core visualization
        heatmap!(ax, electricity_price_range, gas_price_range, optimal_mode_grid, 
                colormap=cmap, colorrange=(0.5, 3.5))
        
        if length(unique(optimal_mode_grid)) > 1
            contour!(ax, electricity_price_range, gas_price_range, optimal_mode_grid, 
                    levels=1:3, color=:black, linewidth=0.5)
        end
        
        # Add scatter for first panel only
        idx == 1 && scatter!(ax, samples[1, time_idx, :], samples[2, time_idx, :], 
                           color=:black, markersize=4, alpha=0.3)
        
        # Set consistent limits
        xlims!(ax, x_min, x_max)
        ylims!(ax, y_min, y_max)
    end

    # Add colorbar and legend
    Colorbar(fig[1:n_rows, n_cols+1], colormap=cmap, limits=(0.5, 3.5),
             ticks=(1:3, ["Off", "Half", "Full"]))

    elements = [MarkerElement(color=c, marker=:rect) for c in colors]
    push!(elements, MarkerElement(color=:black, marker=:circle))
    
    Legend(fig[n_rows+1, 1:n_cols], elements, 
           ["Off", "Half", "Full", "Samples"], 
           "Modes and Samples", 
           orientation=:horizontal)

    return fig
end


function plot_switching_frequency(Vs::Vector, sample_paths::Array{Float64,3}, times, payoff, cost)
    d, num_times, num_paths = size(sample_paths)
    num_strategies = length(Vs)
    
    switching_freq = zeros(Int, num_strategies, num_times-1)
    
    for (strategy_index, V) in enumerate(Vs)
        for path in 1:num_paths
            current_mode = 1
            for t in 1:num_times-1
                state = sample_paths[:, t, path]
                best_mode = current_mode
                best_value = V(state, times[t], current_mode)[1]
                
                for j in 1:3
                    if j != current_mode
                        switch_cost = cost(state, times[t])[current_mode, j]
                        value_after_switch = V(state, times[t], j)[1] - switch_cost
                        if value_after_switch > best_value
                            best_value = value_after_switch
                            best_mode = j
                        end
                    end
                end
                
                if best_mode != current_mode
                    switching_freq[strategy_index, t] += 1
                    current_mode = best_mode
                end
            end
        end
    end
    
    switching_freq = switching_freq ./ num_paths
    
    fig = Figure(size=(1200, 800))
    ax = Axis(fig[1,1], xlabel="Time", ylabel="Strategy", title="Switching Frequency Heatmap")
    heatmap!(ax, times[1:end-1], 1:num_strategies, switching_freq, colormap=:viridis)
    Colorbar(fig[1,2], limits=(0, maximum(switching_freq)), colormap=:viridis, label="Switching Frequency")
    
    ax.yticks = (1:num_strategies, [V.name for V in Vs])
    
    fig
end


function create_grid_value_function(RandomProcess, grid_points, t_start, dt, p, num_paths=5000)
    # Create a grid of initial points
    x_range = range(grid_points[1, 1], grid_points[1, 2], length=100)
    y_range = range(grid_points[2, 1], grid_points[2, 2], length=100)
    
    # Create initial states grid
    x_init = zeros(2, length(x_range) * length(y_range))
    idx = 1
    for x in x_range, y in y_range
        x_init[:, idx] = [x, y]
        idx += 1
    end
    
    # Generate paths starting from grid points
    trajs = OptSwitch.generate_paths(RandomProcess, x_init, t_start, N, dt, p)[:, :, 1:num_paths]
    
    # Calculate value function on these trajectories
    V = calculate_value_function(trajs, cost, payoff, times, dt, J)
    
    return trajs, V
end

struct apost_grid
    x_range::StepRangeLen
    y_range::StepRangeLen
    j_range::StepRangeLen
    t_range::Union{StepRangeLen, Vector{Int64}}
    values::Array{Float64,4}  # Now 4D array
    
    function apost_grid(x_range, y_range, j_range, t_range, values)
        @assert size(values) == (length(x_range), length(y_range), length(j_range), length(t_range))
        new(x_range, y_range, j_range, t_range, values)
    end
end

function (G::apost_grid)(state, time, j::Int)
    i = clamp(searchsortedfirst(G.x_range, state[1]), 1, length(G.x_range))
    y = clamp(searchsortedfirst(G.y_range, state[2]), 1, length(G.y_range))
    k = clamp(searchsortedfirst(G.j_range, j), 1, length(G.j_range))
    t = clamp(searchsortedfirst(G.t_range, time), 1, length(G.t_range))
    
    return G.values[i,y,k,t]
end

function create_value_grid_timepoint(generate_paths,RandomProcess, grid_points, tp,N,dt, t_start, p, cost, payoff, J)
    # Create spatial grids
    x_range = range(grid_points[1, 1], grid_points[1, 2], length=100)
    y_range = range(grid_points[2, 1], grid_points[2, 2], length=100)
    shift=N-tp
    # Initialize flattened state array
    x_init_new = zeros(2, length(x_range) * length(y_range))
    idx = 1
    for x in x_range, y in y_range
        x_init_new[:, idx] = [x, y]
        idx += 1
    end
    
    # Generate trajectories starting at tp
    trajs_new = generate_paths(RandomProcess, x_init_new, t_start + tp*dt, shift, dt, p)
    optimal_value = calculate_value_function(trajs_new, cost, payoff, tp:181, dt, J)
    
    # Reshape values for this timepoint
    reshaped_values = zeros(length(x_range), length(y_range), J)
    for (idx, (y_idx, x_idx)) in enumerate(Iterators.product(1:length(y_range), 1:length(x_range)))
        for j in 1:J
            reshaped_values[x_idx, y_idx, j] = optimal_value[1, idx, j]  # Take first timepoint
        end
    end
    
    return reshaped_values
end

# Function to create full grid for multiple timepoints
function create_full_value_grid(generate_paths,RandomProcess, grid_points, timepoints,N, dt, t_start, p, cost, payoff, J)
    x_range = range(grid_points[1, 1], grid_points[1, 2], length=100)
    y_range = range(grid_points[2, 1], grid_points[2, 2], length=100)
    t_range = timepoints
    
    # Initialize 4D array
    full_values = zeros(length(x_range), length(y_range), J, length(timepoints))
    
    # Compute values for each timepoint
    for (t_idx, tp) in enumerate(timepoints)
        full_values[:,:,:,t_idx] = create_value_grid_timepoint(generate_paths,
            RandomProcess, grid_points, tp,N,  dt, t_start, p, cost, payoff, J)
    end
    
    return apost_grid(x_range, y_range, 1:J, t_range, full_values)
end


 function plot_value_differences(reshaped_values, x_range, y_range)
    fig = Figure(size=(900, 300))
    
    # Calculate differences between modes
    diff_1_2 = reshaped_values[:,:,1] - reshaped_values[:,:,2]
    diff_1_3 = reshaped_values[:,:,1] - reshaped_values[:,:,3]
    diff_2_3 = reshaped_values[:,:,2] - reshaped_values[:,:,3]
    
    # Create subplots
    ax1 = Axis(fig[1,1], title="Mode 1 - Mode 2", xlabel="Electricity Price", ylabel="Gas Price")
    ax2 = Axis(fig[1,2], title="Mode 1 - Mode 3", xlabel="Electricity Price")
    ax3 = Axis(fig[1,3], title="Mode 2 - Mode 3", xlabel="Electricity Price")
    
    # Plot heatmaps with consistent color scaling
    max_abs_diff = maximum(abs, [diff_1_2; diff_1_3; diff_2_3])
    colorrange = (-max_abs_diff, max_abs_diff)
    colormap = :balance  # diverging colormap centered at 0
    
    hm1 = heatmap!(ax1, x_range, y_range, diff_1_2, colormap=colormap, colorrange=colorrange)
    hm2 = heatmap!(ax2, x_range, y_range, diff_1_3, colormap=colormap, colorrange=colorrange)
    hm3 = heatmap!(ax3, x_range, y_range, diff_2_3, colormap=colormap, colorrange=colorrange)
    
    # Add contour lines at zero (switching boundaries)
    contour!(ax1, x_range, y_range, diff_1_2, levels=[0], color=:black)
    contour!(ax2, x_range, y_range, diff_1_3, levels=[0], color=:black)
    contour!(ax3, x_range, y_range, diff_2_3, levels=[0], color=:black)
    
    # Add colorbar
    Colorbar(fig[1,4], colormap=colormap, colorrange=colorrange, label="Value Difference")
    
    fig
end

end