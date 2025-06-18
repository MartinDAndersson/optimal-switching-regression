module aclp_plots
using DrWatson
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

export calculate_value_function
export calculate_greedy_value_matrix
export determine_optimal_strategy
export compute_strategy_for_value
export plot_switching_strategies_and_values
export plot_strategy_analysis
export prepare_strategy_analysis
export plot_value_prediction_error

current_dir = srcdir()

include(srcdir("plots_common.jl"))



function create_strategy_overview_figure(strategies, times, strategy_names;
                                       figsize=(350, 550))
    fig = Figure(size=figsize, fontsize=10, backgroundcolor=:white)
    
    mode_colors = [
        ColorSchemes.Paired_4[1],  # Mode 1 
        ColorSchemes.Paired_4[2],  # Mode 2 
        ColorSchemes.Paired_4[3],   # Mode 3 
        ColorSchemes.Paired_4[4]   # Mode 4 
    ]
    mode_labels = ["Mode 1", "Mode 2", "Mode 3", "Mode 4"]
    
    n_strategies = length(strategies)
    plot_height = 35  # Slightly increased for better readability
    bands = []
    
    # Optimized grid layout with refined margins
    gl = GridLayout(fig[1, 1], 
                   margins=(40, 20, 25, 25),  # Increased left margin for labels
                   halign=:left,
                   valign=:top)
    
    Label(gl[1, 1:2], "Operational Mode Distribution",
          font=:bold, fontsize=12, padding=(0, 0, 15, 0))
    
    for (i, (strategy, name)) in enumerate(zip(strategies, strategy_names))
        Label(gl[i+1, 1], name,
              halign=:right,
              valign=:center,
              fontsize=10,
              padding=(0, 15, 0, 0))  # Increased right padding
        
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
        
        mode_props = compute_mode_proportions(strategy, length(times), 
                                            size(strategy, 2), 4)
        stack_bottom = zeros(length(times))
        
        for mode in 1:4
            stack_top = stack_bottom .+ mode_props[:, mode]
            b = band!(ax, times, stack_bottom, stack_top,
                     color=(mode_colors[mode], 0.9))
            i == 1 && push!(bands, b)
            stack_bottom = stack_top
        end
        
        limits!(ax, extrema(times)..., 0, 1)
    end
    
    # Optimized legend configuration
    Legend(gl[n_strategies+2, 1:2],
           bands, mode_labels,
           orientation=:horizontal,
           nbanks=2,
           framecolor=(:gray, 0.2),
           framevisible=true,
           titleposition=:left,
           title="Modes:",
           titlesize=10,
           labelsize=9,
           margin=(15, 15, 10, 10),  # Increased margins
           patchsize=(25, 10),       # Larger patches
           rowgap=8,                 # Increased row gap
           colgap=15)                # Increased column gap
    
    # Refined layout spacing
    colsize!(gl, 1, Auto(1.2))      # Increased label column width
    colsize!(gl, 2, Relative(0.75))  # Adjusted plot area
    rowgap!(gl, 10)                 # Consistent row spacing
    
    return fig
end




end