module hcl_plots
using DrWatson
@quickactivate
current_dir = scriptsdir("plotting")

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
export calculate_strategy_distances_gray
export calculate_strategy_distance
export generate_visualizations

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
end