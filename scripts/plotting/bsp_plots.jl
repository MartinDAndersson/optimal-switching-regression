module bsp_plots
#current_dir = pwd()*"/scripts/new_experiments/"
using DrWatson
@quickactivate
using CairoMakie
using StatsBase: quantile
using Makie
using Random
using Statistics
import .Threads
using StaticArrays
using DataFrames
using ColorSchemes
using KernelDensity
using Colors: RGB, gray

export calculate_value_function
export calculate_greedy_value_matrix
export determine_optimal_strategy
export compute_strategy_for_value
export plot_switching_strategies_and_values
export plot_strategy_analysis
export prepare_strategy_analysis
export plot_value_prediction_error
export plot_switching_boundaries_comparison
export visualize_payoff_3d

include(srcdir("plots_common.jl"))

function create_strategy_overview_figure(strategies, times, strategy_names;
                                       figsize=(350, 550))
    fig = Figure(size=figsize, fontsize=10, backgroundcolor=:white)
    
    # Expand color palette for 10 modes
    mode_colors = [
        ColorSchemes.tableau_10[i] for i in 1:10
    ]
    mode_labels = ["Mode $i" for i in 1:10]
    
    n_strategies = length(strategies)
    plot_height = 35
    bands = []
    
    gl = GridLayout(fig[1, 1], 
                   margins=(40, 20, 25, 25),
                   halign=:left,
                   valign=:top)
    
    Label(gl[1, 1:2], "Operational Mode Distribution",
          font=:bold, fontsize=12, padding=(0, 0, 15, 0))
    
    for (i, (strategy, name)) in enumerate(zip(strategies, strategy_names))
        Label(gl[i+1, 1], name,
              halign=:right,
              valign=:center,
              fontsize=10,
              padding=(0, 15, 0, 0))
        
        ax = Axis(gl[i+1, 2],
                 height=plot_height,
                 xlabelvisible=i == n_strategies,
                 xticklabelsize=9,
                 xticks=LinearTicks(4),
                 yticks=LinearTicks(1),
                 spinewidth=0.5,
                 xgridvisible=false,
                 ygridvisible=false,
                 yaxisposition=:right)
        
        i != n_strategies && hidexdecorations!(ax)
        
        # Modified to handle 10 modes
        mode_props = compute_mode_proportions(strategy, length(times), 
                                            size(strategy, 2), 10)
        stack_bottom = zeros(length(times))
        
        for mode in 1:10
            stack_top = stack_bottom .+ mode_props[:, mode]
            b = band!(ax, times, stack_bottom, stack_top,
                     color=(mode_colors[mode], 0.9))
            i == 1 && push!(bands, b)
            stack_bottom = stack_top
        end
        
        limits!(ax, extrema(times)..., 0, 1)
    end
    
    # Modified legend to handle 10 modes in a more compact layout
    Legend(gl[n_strategies+2, 1:2],
           bands, mode_labels,
           orientation=:horizontal,
           nbanks=3,  # Increased banks for more modes
           framecolor=(:gray, 0.2),
           framevisible=true,
           titleposition=:left,
           title="Modes:",
           titlesize=10,
           labelsize=9,
           margin=(15, 15, 10, 10),
           patchsize=(20, 8),  # Slightly smaller patches
           rowgap=6,
           colgap=12)
    
    colsize!(gl, 1, Auto(1.2))
    colsize!(gl, 2, Relative(0.75))
    rowgap!(gl, 10)
    
    return fig
end


function plot_strategy_performance(df)
    fig = Figure(size=(900, 400))
    
    ax1 = Axis(fig[1, 1],
        xlabel = "Final Value",
        ylabel = "Strategy",
        title = "Strategy Performance Overview",
        titlesize = 16)
    
    df_sorted = sort(df[df.Strategy .!= "aposteriori", :], :NormalizedFinalValue, rev=true)
    strategies = df_sorted.Strategy
    n_strategies = length(strategies)
    
    y_positions = 1:n_strategies
    
    scatter!(ax1, df_sorted.NonNormalizedFinalValue, y_positions,
        color = :black,
        marker = :circle,
        markersize = 12,
        label = "Actual")
    
    scatter!(ax1, df_sorted.PredictedValue, y_positions,
        color = :black,
        marker = :xcross,
        markersize = 12,
        label = "Predicted")
    
    for (i, row) in enumerate(eachrow(df_sorted))
        lines!(ax1, [row.NonNormalizedFinalValue, row.PredictedValue], [i, i],
            color = :black,
            linestyle = :dash,
            linewidth = 1)
    end
    
    optimal_value = df[df.Strategy .== "aposteriori", :NonNormalizedFinalValue][1]
    vlines!(ax1, optimal_value, color = :black, linestyle = :dashdot, label = "Optimal")
    
    for (i, row) in enumerate(eachrow(df_sorted))
        gap = round(100 * row.DifferenceFromOptimal, digits=2)
        text!(ax1, row.NonNormalizedFinalValue - 5, i,
            text = "Gap: $(gap)%",
            align = (:right, :center),
            fontsize = 10)
    end
    
    ax1.yticks = (y_positions, strategies)
    axislegend(ax1, position = :rt)
    
    return fig
end

using Statistics
using ColorSchemes
using StatsBase
function plot_switching_boundaries_comparison_1d(
    Vs::Vector,
    cost::Function, 
    payoff::Function, 
    time_point::Real,
    samples::Array{Float64, 3},  # 3D array: [input_dim, time, samples]
    dt::Real,
    J::Int,
    initial_mode::Int
)
    # Data preprocessing 
    time_idx = time_point
    x_min, x_max = percentile(vec(samples[1, time_idx, :]), [1, 99])  # Percentiles for 1D input at the given time
    x_range = x_max - x_min

    # Grid setup
    price_range = range(x_min - 0.05 * x_range, x_max + 0.05 * x_range, length=100)

    optimal_mode(V, price) = begin
        state = [price]  # 1D state
        choose_best_mode(V, state, time_point, initial_mode, cost, dt, J)
    end

    # Layout configuration
    n_cols = 2
    n_rows = 3
    fig = Figure(size=(200 * n_cols, 60 + 200 * n_rows))
    
    Label(fig[0, 1:2], "Switching Boundaries Comparison\nat t = $time_point", font=:bold)
    
    # Color configuration
    colors = ColorSchemes.Paired_10  # Use a color scheme with 10 colors
    cmap = cgrad(colors, 10, categorical=true)

    # Create grid of plots
    for (idx, V) in enumerate(Vs)
        row, col = fldmod1(idx, n_cols)
        
        ax = Axis(fig[row, col], xticks=LinearTicks(4))
        
        # Handle axis decorations
        if row != n_rows
            hidexdecorations!(ax)
        else
            ax.xlabel = "Price"
        end
        
        if col != 1
            hideydecorations!(ax)
        else
            ax.ylabel = "Mode"
        end
        
        ax.title = V.name

        # Generate optimal mode grid
        optimal_mode_grid = [optimal_mode(V, x) for x in price_range]

        # Core visualization
        heatmap!(ax, price_range, 1:10, reshape(optimal_mode_grid, (length(price_range), 1)), 
                colormap=cmap, colorrange=(0.5, 10.5))
        
        if length(unique(optimal_mode_grid)) > 1
            contour!(ax, price_range, 1:10, reshape(optimal_mode_grid, (length(price_range), 1)), 
                    levels=1:10, color=:black, linewidth=0.5)
        end
        
        # Add scatter for first panel only
        if idx == 1
            scatter!(ax, samples[1, time_idx, :], fill(initial_mode, size(samples, 3)), 
                     color=:black, markersize=4, alpha=0.3)
        end
        
        # Set consistent limits
        xlims!(ax, x_min, x_max)
        ylims!(ax, 1, 10)
    end

    # Add colorbar and legend
    Colorbar(fig[1:n_rows, n_cols+1], colormap=cmap, limits=(0.5, 10.5),
             ticks=(1:10, ["Mode $i" for i in 1:10]))

    elements = [MarkerElement(color=c, marker=:rect) for c in colors]
    push!(elements, MarkerElement(color=:black, marker=:circle))
    
    Legend(fig[n_rows+1, 1:n_cols], elements, 
           ["Mode $i" for i in 1:10] + ["Samples"], 
           "Modes and Samples", 
           orientation=:horizontal)

    return fig
end

using StatsBase
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
    # Data preprocessing (1D state)
    time_idx = time_point  # assume time_point indexes the sample time
    x_min, x_max = percentile(vec(samples[1, :, :]), [1, 99])
    x_range = x_max - x_min
    state_range = range(x_min - 0.05*x_range, x_max + 0.05*x_range, length=200)

    # Optimal mode function (state = [x])
    optimal_mode(V, x) = choose_best_mode(V, [x], time_point, initial_mode, cost, dt, J)

    # Layout configuration: increased resolution for a cleaner look
    n_cols = 2
    n_rows = ceil(Int, length(Vs)/n_cols)
    fig = Figure(size = (400*n_cols, 300*n_rows))
    Label(fig[0, 1:n_cols], "Switching Boundaries Comparison at t = $time_point", fontsize=18, halign = :center)

    # Use a vector of 10 discrete colors
    colors = distinguishable_colors(10)

    for (idx, V) in enumerate(Vs)
        row, col = fldmod1(idx, n_cols)
        ax = Axis(fig[row, col],
                  xlabel = "State",
                  ylabel = "Optimal Mode",
                  title = V.name,
                  yticks = (1:10, ["Mode $i" for i in 1:10]))
        
        mode_vals = [optimal_mode(V, x) for x in state_range]

        # Draw dashed line connecting optimal modes (giving a step-like feel)
        lines!(ax, state_range, mode_vals, color = :black, linewidth = 2, linestyle = :dash)
        scatter!(ax, state_range, mode_vals, color = [colors[m] for m in mode_vals],
                 markersize = 12, marker = :rect)
        
        # Overlay sample points on first panel
        if idx == 1
            sample_x = vec(samples[1, time_idx, :])
            sample_modes = [optimal_mode(V, x) for x in sample_x]
            scatter!(ax, sample_x, sample_modes, color = :gray, markersize = 8, marker = :circle, transparency = true)
        end

        xlims!(ax, first(state_range), last(state_range))
        ylims!(ax, 0.5, 10.5)
    end

    # Create a discrete colorbar using a dummy image
    dummy_img = reshape(1:10, 10, 1)
    cb_ax = Axis(fig[1, n_cols+1])
    heatmap!(cb_ax, [1], 1:10, dummy_img, colormap = colors, colorrange = (0.5, 10.5), interpolate = false)
    Colorbar(fig[1:n_rows, n_cols+1], colormap = colors, limits = (0.5, 10.5),
             ticks = (1:10, ["Mode $i" for i in 1:10]), label = "Modes", ticklabelsize=12)

    return fig
end

function visualize_payoff_3d(payoff_func, cost_func)
    # Set figure with a fixed aspect ratio
    fig = Figure(size=(800, 600), fontsize=14)
    
    # Create carefully sized grid layouts
    top_row = fig[1, 1] = GridLayout()
    bottom_row = fig[2, 1] = GridLayout()
    
    # Set row sizes with proper proportions
    rowsize!(fig.layout, 1, Relative(0.6))
    rowsize!(fig.layout, 2, Relative(0.4))
    
    # Calculate aspect ratio for the 3D plot considering state and mode ranges
    state_range_size = 5.0  # -2.5 to 2.5
    mode_range_size = 10.0  # 1 to 10
    aspect_3d = state_range_size / mode_range_size
    
    # Main 3D surface plot - takes most of top row
    ax_3d = Axis3(top_row[1, 1:2], 
        xlabel="State value", 
        ylabel="Mode", 
        zlabel="Payoff",
        title="3D Payoff Structure",
        titlesize=16,
        titlealign=:left)  # Align title left to avoid interference
    
    # Cost matrix - right side of top row
    ax_cost = Axis(top_row[1, 3], 
        xlabel="To Mode",
        ylabel="From Mode",
        title="Switching Cost Matrix",
        titlesize=16,
        titlealign=:left,  # Align title left
        aspect=1)  # Square aspect ratio for cost matrix
    
    # Bottom row - Heatmap (left) and Optimal Mode (right)
    ax_heat = Axis(bottom_row[1, 1:2], 
        xlabel="State value", 
        ylabel="Mode",
        title="Payoff Heatmap",
        titlesize=16,
        titlealign=:left,
        aspect=2.0)  # Wide aspect ratio for heatmap
    
    ax_opt = Axis(bottom_row[1, 3], 
        xlabel="State value", 
        ylabel="Optimal Mode",
        title="Optimal Mode by State",
        titlesize=16,
        titlealign=:left,
        aspect=2.0)  # Match aspect ratio with heatmap
    
    # Generate state values
    x_range = range(-2.5, 2.5, length=200)
    mode_range = 1:10
    
    # Calculate payoffs for every state-mode combination
    z = zeros(Float32, length(x_range), length(mode_range))
    for (i, x) in enumerate(x_range)
        payoff_vector = payoff_func(SVector{1}([x]), 0.0)
        for (j, mode) in enumerate(mode_range)
            z[i, j] = payoff_vector[mode]
        end
    end
    
    # Create the 3D surface
    # Create the 3D surface with adjusted render settings
    surface!(ax_3d, x_range, mode_range, z, 
        colormap=:viridis,
        transparency=false,  # Change to false for PDF export
        alpha=1.0,           # Increase to 1.0 for PDF export
        shading=true,        # Enable shading for better rendering
        overdraw=true)       # Help with overlapping polygons
    

    # Get cost matrix (constant across states)
    cost_matrix = Array(cost_func(SVector{1}([0.0]), 0.0))
    
    # Create heatmap for cost
    hm_cost = heatmap!(ax_cost, 1:10, 1:10, cost_matrix, 
        colormap=:inferno)
    
    # Colorbar for cost - adjust placement to avoid title interference
    cb_cost = Colorbar(top_row[1, 4], hm_cost, 
                      label="Cost value",
                      height=Relative(0.8),
                      valign=:center)
    
    # Create payoff heatmap
    hm = heatmap!(ax_heat, x_range, mode_range, z, 
        colormap=:viridis)
    
    # Colorbar for payoff - adjust placement
    cb_payoff = Colorbar(bottom_row[1, 4], hm, 
                        label="Payoff value",
                        height=Relative(0.8),
                        valign=:center)
    
    # Optimal mode plot
    optimal_modes = [argmax(z[i, :]) for i in 1:size(z, 1)]
    scatter!(ax_opt, x_range, optimal_modes, 
        markersize=4, 
        color=:black)
    
    # Add horizontal lines to show full mode range
    hlines!(ax_opt, 1:10, color=(:gray, 0.3), linestyle=:dash)
    
    # Add mode boundary vertical lines
    for n in 1:9
        boundary = n * 0.4 - 2.0
        vlines!(ax_heat, boundary, linestyle=:dash, color=(:black, 0.5), linewidth=1)
        vlines!(ax_opt, boundary, linestyle=:dash, color=(:black, 0.5), linewidth=1)
    end
    
    # Make axes limits consistent between corresponding plots
    ylims!(ax_heat, 0.5, 10.5)
    ylims!(ax_opt, 0.5, 10.5)
    xlims!(ax_heat, -2.5, 2.5)
    xlims!(ax_opt, -2.5, 2.5)
    
    # Link x-axes for heatmap and optimal mode plot for perfect alignment
    linkxaxes!(ax_heat, ax_opt)
    
    # Adjust xticks and yticks for cost matrix for cleaner presentation
    ax_cost.xticks = 1:2:10
    ax_cost.yticks = 1:2:10
    
    # Ensure consistent ticks and formatting across plots
    ax_heat.xticks = -2:1:2
    ax_opt.xticks = -2:1:2
    
    # Set appropriate spacing between grid elements - reduce gaps
    colgap!(top_row, 5)
    colgap!(bottom_row, 5)
    rowgap!(fig.layout, 5)
    
    # Apply tight layout
    fig.layout.alignmode = Outside(5)  # Reduce outside padding
    
    # Return the figure
    return fig
end




end
