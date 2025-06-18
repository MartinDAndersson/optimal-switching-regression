function calculate_value_function(sample_paths::Array{Float64,3}, cost, payoff, times, dt,modes)
    d, num_times, num_paths = size(sample_paths)
    num_modes = modes  # 
    # Initialize value function
    V = zeros(num_times, num_paths, num_modes)

    # Backward induction
    for t in num_times-1:-1:1
        for path in 1:num_paths
            state = sample_paths[:, t, path]
            time = (times[t] - 1) * dt
            
            payoffs = payoff(state, time) .* dt
            costs = cost(state, time)
            
            for current_mode in 1:num_modes
                V[t, path, current_mode] = maximum(
                    payoffs[next_mode] - costs[current_mode, next_mode] + V[t+1, path, next_mode]
                    for next_mode in 1:num_modes
                )
            end
        end
    end

    return V
end


"""
    choose_best_mode(V, state, time, current_mode, cost, dt, J)

Choose the optimal mode to switch to given the current state and value function.

Parameters:
- V: Value function V(state, time, mode)
- state: Current state vector
- time: Current time
- current_mode: Current operating mode
- cost: Switching cost function cost(state, time)[from_mode, to_mode]
- dt: Time step
- J: Number of possible modes

Returns:
- Optimal mode to switch to
"""
function choose_best_mode(V, state, time, current_mode, cost, dt, J)
    # Calculate value-cost pairs for all possible modes
    values = Vector{Float64}(undef, J)
    for j in 1:J
        values[j] = V(state, time, j)[1] - cost(state, time*dt)[current_mode, j]
    end
    
    # Pass both values and cost to select_best_mode
    return select_best_mode(values, current_mode; cost=cost(state, time*dt))
end

"""
    select_best_mode(values, current_mode; cost=nothing, rel_factor=1e-3, abs_thresh=1e-3)

Select the best mode based on values with a threshold-based approach that favors
staying in the current mode when differences are small.

Parameters:
- values: Vector of values for each mode (already including switching costs)
- current_mode: Current operating mode
- cost: Optional cost function or matrix for future use
- rel_factor: Relative threshold factor (default: 1e-3)
- abs_thresh: Absolute minimum threshold (default: 1e-3)

Returns:
- Optimal mode to switch to
"""

function prepare_for_latex(df)
    # Replace any special characters in column names
    rename_dict = Dict(
        n => replace(String(n), "_" => " ", "=" => "-") 
        for n in names(df)
    )
    df_latex = rename(df, rename_dict)
    
    # Convert any problematic values
    for col in names(df_latex)
        # Handle NaN, Inf values
        if eltype(df_latex[!, col]) <: AbstractFloat
            df_latex[!, col] = map(x -> 
                isnan(x) ? "NA" : 
                isinf(x) ? (x > 0 ? "Inf" : "-Inf") : 
                round(x, digits=5), 
                df_latex[!, col])
        end
        
        # Convert any string with LaTeX special characters
        if eltype(df_latex[!, col]) <: AbstractString
            df_latex[!, col] = map(x -> 
                replace(x, 
                    "&" => "\\&", 
                    "%" => "\\%", 
                    "_" => "\\_", 
                    "#" => "\\#", 
                    "~" => "\\textasciitilde{}"),
                df_latex[!, col])
        end
    end
    
    return df_latex
end

export prepare_for_latex

function select_best_mode(values, current_mode; cost=nothing, rel_factor=1e-4, abs_thresh=1e-5)  #abs_thresh=1e-3
    # Find maximum value and its mode
    val_max = maximum(values)
    J = length(values)
    best_mode = argmax(values)
    current_value = values[current_mode]
    non_curr_max = -Inf
    non_curr_mode = rand(1:J)
    for j in 1:J
        if j != current_mode
            if values[j] >= non_curr_max
                non_curr_max = values[j]
                non_curr_mode = j
            end
        end
    end 
    # Define relative threshold based on value scale
    thresh = rel_factor * abs(val_max)

    # Define effective threshold
    #thresh = max(rel_thresh, abs_thresh)
    #thresh = rel_thresh*abs(val_max)
    #@info "thresh = $thresh"
    # If best mode is clearly better, take it
    if non_curr_max - current_value >= -abs_thresh#thresh
        #@info "Switching from $current_mode to $non_curr_mode"
        return non_curr_mode
    else
        return current_mode
        # # Find all modes with values close to the maximum
        # candidates = findall(x -> val_max - x <= thresh, values)
        
        # # If we have multiple candidates, prefer non-current modes
        # if length(candidates) > 1
        #     non_current_candidates = filter(x -> x != current_mode, candidates)
        #     if !isempty(non_current_candidates)
        #         # Among the non-current candidates, select the one with highest value
        #         return non_current_candidates[argmax(values[non_current_candidates])]
        #     end
        # end
        
        # # Default: stay in current mode if it's among candidates,
        # # otherwise take the first candidate (highest value)
        # return current_mode in candidates ? current_mode : candidates[1]
    end
end

# Replace the old simplified version to use the new unified logic
function choose_best_mode(values, current_mode, J)
    return select_best_mode(values, current_mode)
end

function calculate_accumulated_value(strategy::Matrix{Int}, sample_paths::Array{Float64,3}, 
    times, payoff, cost, dt;initial_mode=1)
    d, num_times, num_paths = size(sample_paths)
    accumulated_values = zeros(num_times, num_paths)
    
    for path in 1:num_paths
        total_accumulated = 0.0
        current_mode = initial_mode
        
        for t in 1:num_times-1
            state = sample_paths[:, t, path]
            real_time = (t-1)*dt
            
            # Calculate switch cost if applicable
            if t == 1
                switch_cost = cost(state, real_time)[initial_mode, strategy[t, path]]
            else
                previous_mode = strategy[t-1, path]
                switch_cost = cost(state, real_time)[previous_mode, strategy[t, path]]
            end
            
            # Apply the switch cost
            total_accumulated -= switch_cost
            
            # Update current mode
            current_mode = strategy[t, path]
            
            # Calculate payoff for the current time step
            current_payoff = payoff(state, real_time)[current_mode] * dt
            
            # Update total accumulated value
            total_accumulated += current_payoff
            accumulated_values[t+1, path] = total_accumulated
        end
        accumulated_values[end, path] = total_accumulated
        # Reverse the accumulated values to show accumulation from each time step forward
        #accumulated_values[:, path] = total_accumulated .- accumulated_values[:, path]
    end
    
    return accumulated_values
end

function determine_optimal_strategy(V, sample_paths, cost, times, dt,J, initial_mode)
    num_times, num_paths, num_modes = size(V)
    optimal_strategy = zeros(Int, num_times, num_paths)
    
    # Set initial modes
    optimal_strategy[1, :] .= initial_mode

    for t in 1:num_times-1  # Note: we only need to determine strategy up to the second-to-last time step
        for path in 1:num_paths
            current_mode = t == 1 ? initial_mode : optimal_strategy[t-1, path]
            
            state = sample_paths[:, t, path]
            time = (times[t] - 1) * dt
            
            costs = cost(state, time)
            
            # Calculate values for switching to each mode
            values = [V[t, path, j] - costs[current_mode, j] for j in 1:num_modes]
            #best_mode = argmax(values)
            # Pass both values and cost matrix to select_best_mode
            best_mode = select_best_mode(values, current_mode; cost=costs,rel_factor=0., abs_thresh=0.)
            
            optimal_strategy[t, path] = best_mode
        end
    end

    # For the last time step, keep the previous mode
    optimal_strategy[end, :] .= optimal_strategy[end-1, :]

    return optimal_strategy
end

using CairoMakie, LaTeXStrings, Colors

function plot_value_prediction_error(Vs::Vector, sample_paths::Array{Float64,3}, 
                                   times, payoff, cost, dt,J; initial_mode=1)
    # Create figure with vertically-stacked panels
    fig = Figure(size=(400, 620), fontsize=10)  # Standardized base fontsize
    gl = fig[1,1] = GridLayout(margins=(25, 25, 25, 25))
    
    # Create vertically-stacked axes
    ax1 = Axis(gl[1, 1], 
              xlabel="Time",
              ylabel="Prediction Error",
              title="Instantaneous Error",
              titlesize=16,
              xlabelsize=12,
              ylabelsize=12)
    
    ax2 = Axis(gl[2, 1],
              xlabel="Time",
              ylabel="Cumulative |Error|",
              title="Accumulated Error",
              titlesize=16,
              xlabelsize=12,
              ylabelsize=12)
    
    # Visual parameters remain the same
    n_models = length(Vs)
    colors = ColorSchemes.okabe_ito[1:min(8, n_models)]
    markers = [:circle, :utriangle, :dtriangle, :rect, :star4, :cross]
    line_styles = [:solid, :dash, :dot, :dashdot]
    
    # Data computation and plotting remain identical
    error_data = []
    quality_metrics = Dict{String, Tuple{Float64, Float64}}()
    all_mean_errs = Float64[]
    all_cum_errs = Float64[]
    times_vec = collect(times)
    
    # Compute error data (same as before)
    for (idx, V) in enumerate(Vs)
        predicted_values, strategy = compute_strategy_for_value(
            V, sample_paths, times, cost, dt, initial_mode,J)
        
        accumulated_values = calculate_accumulated_value(
            strategy, sample_paths, times, payoff, cost, dt; 
            initial_mode=initial_mode)
        
        actual_values = [accumulated_values[end, path] .- accumulated_values[:, path] 
                        for path in 1:size(accumulated_values, 2)]
        actual_values = hcat(actual_values...)
        
        error = predicted_values - actual_values
        mean_err = vec(mean(error, dims=2))
        std_err = vec(std(error, dims=2))
        
        cum_err = zeros(length(times))
        for i in 2:length(times)
            cum_err[i] = trapz(times_vec[1:i], abs.(mean_err[1:i]))
        end
        
        quality_metrics[V.name] = (abs(mean_err[1]), cum_err[end])
        push!(error_data, (V.name, mean_err, std_err, cum_err,
                         colors[idx], markers[mod1(idx, 6)], 
                         line_styles[mod1(idx, 4)]))
        append!(all_mean_errs, mean_err)
        append!(all_cum_errs, cum_err)
    end
    
    # Plot data (same as before)
    for (name, mean_err, std_err, cum_err, color, marker, style) in error_data
        scatterlines!(ax1, times[1:5:end], mean_err[1:5:end],
                     color=color, marker=marker, markersize=8,
                     linestyle=style, linewidth=1.2,
                     label=name)
        
        scatterlines!(ax2, times[1:5:end], cum_err[1:5:end],
                     color=color, marker=marker, markersize=8,
                     linestyle=style, linewidth=1.2,
                     label=name)
    end
    
    hlines!(ax1, [0], color=(:black, 0.5), linestyle=:dash, linewidth=0.5)
    
    # Configure axis limits
    err_range = maximum(all_mean_errs) - minimum(all_mean_errs)
    cum_range = maximum(all_cum_errs) - minimum(all_cum_errs)
    
    ylims!(ax1, 
           minimum(all_mean_errs) - 0.1 * err_range,
           maximum(all_mean_errs) + 0.1 * err_range)
    ylims!(ax2, 
           0,
           maximum(all_cum_errs) * 1.1)
    
    # Enhanced legend styling
    legend_theme = (
        framecolor = (:black, 0.1),
        framewidth = 1,
        padding = (8, 8, 8, 8),
        patchsize = (25, 2),
        rowgap = 4,
        labelsize = 12
    )
    
    # Single legend at bottom with enhanced styling
    Legend(gl[3, 1], ax1,
           orientation=:horizontal,
           nbanks=2;
           legend_theme...)
    
    # Configure layout spacing for vertical arrangement
    rowsize!(gl, 1, Relative(0.4))
    rowsize!(gl, 2, Relative(0.4))
    rowsize!(gl, 3, Relative(0.2))
    rowgap!(gl, 20)
    
    return fig, quality_metrics
end

# Helper function for numerical integration
function trapz(x::Vector, y::Vector)
    @assert length(x) == length(y) "x and y must have same length"
    n = length(x)
    r = 0.0
    for i in 2:n
        r += (x[i] - x[i-1]) * (y[i] + y[i-1])/ 2
    end
    return r
end

function create_value_plot!(fig, row, col_start, acc_values, times, strategy_name, reference_values=nothing, diff_limits=nothing)
    # Calculate mean value regardless of whether we're plotting
    mean_val = vec(mean(acc_values, dims=2))
    
    if !isnothing(reference_values)
        ref_mean = vec(mean(reference_values, dims=2))
        ref_scale = mean(ref_mean)
        diff_percent = 100 * (ref_mean .- mean_val) ./ ref_scale
        
        # Ensure we have valid limits
        y_limits = if !isnothing(diff_limits) && !any(isnan.(diff_limits))
            diff_limits
        else
            valid_diffs = filter(!isnan, diff_percent)
            if isempty(valid_diffs)
                (-1.0, 1.0)
            else
                mn, mx = extrema(valid_diffs)
                rng = mx - mn
                # Add slightly more padding since aid case might have larger variations
                (mn - 0.1 * rng, mx + 0.1 * rng)
            end
        end

        time_range = extrema(times)
        tick_count = min(10, length(times))
        tick_values = range(time_range[1], time_range[2], tick_count)
        
        ax = Axis(fig[row, col_start+2], 
            xlabel = "Time",
            ylabel = "% Difference from Optimal",
            title = "Performance Gap ($strategy_name)",
            titlesize = 16,
            limits = (time_range[1], time_range[2], y_limits...),
            xticks = tick_values)
            
        lines!(ax, times, diff_percent, color = :red, linewidth = 2)
        hlines!(ax, [0], color = :gray, linestyle = :dash)
        axislegend(ax, position = :rt)

    end
    
    return mean_val
end

function calculate_greedy_value_matrix(sample_paths::Array{Float64,3}, payoff, cost, times, dt, initial_mode::Int,J)
    d, num_times, num_paths = size(sample_paths)
    num_modes = J  # Assuming 4 modes as in previous examples

    # Initialize value matrix and strategy
    V_greedy = zeros(num_times, num_paths, num_modes)
    greedy_strategy = zeros(Int, num_times, num_paths)

    # Set initial mode for all paths
    greedy_strategy[1, :] .= initial_mode

    # Calculate greedy values
    for t in 1:num_times
        for path in 1:num_paths
            state = sample_paths[:, t, path]
            
            # Get current mode
            current_mode = t == 1 ? initial_mode : greedy_strategy[t-1, path]

            # Calculate immediate payoff minus switching cost for each mode
            for mode in 1:num_modes
                immediate_payoff = payoff(state, (times[t]-1)*dt)[mode] * dt
                switching_cost = mode == current_mode ? 0 : cost(state, (times[t]-1)*dt)[current_mode, mode]
                V_greedy[t, path, mode] = immediate_payoff - switching_cost
            end
            
            # Choose the mode with the highest immediate value
            best_mode = argmax(V_greedy[t, path, :])
            greedy_strategy[t, path] = best_mode
        end
    end

    return V_greedy, greedy_strategy
end

function create_value_comparison_figure(accumulated_values, times, strategy_names, reference_values)
    # Create figure with improved dimensions for readability while staying compact
    fig = Figure(size=(350, 550), backgroundcolor=:white)
    
    # Create grid layout with optimized margins for publication
    fig[1,1] = gl = GridLayout(margin=(40, 20, 20, 20))
    
    # Keep original performance calculation logic
    performances = []
    for (i, (values, name)) in enumerate(zip(accumulated_values, strategy_names))
        name == "a posteriori" && continue
        value_loss = calculate_time_series_loss(values, reference_values, times)
        endpoint_performance = abs(value_loss[end])
        push!(performances, (endpoint_performance, i, values, name, value_loss))
    end
    sort!(performances, by=x->x[1])
    performances = performances[1:min(8, length(performances))]
    mid_point = div(length(performances), 2)
    
    # Enhanced panel layout
    top_panel = GridLayout()
    bottom_panel = GridLayout()
    gl[1:2,1] = [top_panel, bottom_panel]
    
    # Improved axis styling
    axis_theme = Dict(
        :backgroundcolor => (:grey90, 0.05),
        :xgridcolor => (:grey, 0.2),
        :ygridcolor => (:grey, 0.2),
        :xgridwidth => 0.5,
        :ygridwidth => 0.5,
        :xminorgridvisible => false,
        :yminorgridvisible => false
    )
    
    # Create axes with enhanced styling
    ax1 = Axis(top_panel[1,1];
        xlabel = "",
        ylabel = "Performance Loss",
        title = "Top Performing Strategies",
        xlabelsize = 12,
        ylabelsize = 12,
        xticklabelsize = 10,
        yticklabelsize = 10,
        axis_theme...)
    
    ax2 = Axis(bottom_panel[1,1];
        xlabel = "Time",
        ylabel = "Performance Loss",
        title = "Other Strategies",
        xlabelsize = 12,
        ylabelsize = 12,
        xticklabelsize = 10,
        yticklabelsize = 10,
        axis_theme...)
    
    # Enhanced line styles for better contrast
    styles = [:solid, :dash, :dot, :dashdot]
    strategy_color = ColorSchemes.Dark2_4  # Professional blue-grey
    
    # Plot top performers with enhanced styling
    top_elements = []
    top_labels = String[]
    for (j, (perf, i, values, name, value_loss)) in enumerate(performances[1:mid_point])
        line = lines!(ax1, times[10:end], value_loss[10:end],
            color = strategy_color[j],
            linestyle = styles[mod1(j, length(styles))],
            linewidth = 1.5)
        push!(top_elements, line)
        push!(top_labels, "$(name) ($(round(perf, digits=3)))")
    end
    opt_line1 = lines!(ax1, times[10:end], zeros(length(times[10:end])),
        color = :black,
        linestyle = :dashdot,
        linewidth = 1.0)
    push!(top_elements, opt_line1)
    push!(top_labels, "a posteriori")
    
    # Plot bottom performers with enhanced styling
    bottom_elements = []
    bottom_labels = String[]
    for (j, (perf, i, values, name, value_loss)) in enumerate(performances[mid_point+1:end])
        line = lines!(ax2, times[10:end], value_loss[10:end],
            color = strategy_color[j],
            linestyle = styles[mod1(j, length(styles))],
            linewidth = 1.5)
        push!(bottom_elements, line)
        push!(bottom_labels, "$(name) ($(round(perf, digits=3)))")
    end
    opt_line2 = lines!(ax2, times[10:end], zeros(length(times[10:end])),
        color = :black,
        linestyle = :dashdot,
        linewidth = 1.0)
    push!(bottom_elements, opt_line2)
    push!(bottom_labels, "a posteriori")
    
    # Enhanced legend styling
    legend_theme = (
        framecolor = (:black, 0.1),
        framewidth = 1,
        padding = (8, 8, 8, 8),
        patchsize = (25, 2),
        rowgap = 4
    )
    
    Legend(top_panel[1,2], top_elements, top_labels; legend_theme...)
    Legend(bottom_panel[1,2], bottom_elements, bottom_labels; legend_theme...)
    
    # Maintain axis linkage
    linkxaxes!(ax1, ax2)
    
    # Improved panel spacing
    rowgap!(gl, 15)
    colgap!(gl, 10)
    
    # Maintain panel size ratio
    rowsize!(gl, 1, Relative(0.5))
    rowsize!(gl, 2, Relative(0.5))
    
    return fig
end


function prepare_strategy_analysis(Vs, payoff, cost, sample_paths, times, dt, J;
                                 other_strategies=[], initial_mode=1)
    d, num_times, num_paths = size(sample_paths)
    @assert num_times == length(times) "Number of time points should match sample paths"

    # Compute core data
    strategies, accumulated_values, mean_values, strategy_names = 
        compute_strategy_data(Vs, other_strategies, sample_paths, times, payoff,
                            cost, dt, initial_mode,J)
    
    # Calculate reference metrics
    reference_values, reference_index, diff_limits = 
        compute_reference_metrics(accumulated_values, strategy_names)
    
    # Compute additional statistics
    normalized_mean_values = mean_values ./ mean(reference_values,dims=2)
    original_mean_values = copy(mean_values)
    predictions = compute_predictions(Vs, other_strategies, sample_paths, times, 
                                   initial_mode, original_mean_values)
    differences = reference_index === nothing ? nothing :
        compute_differences(strategies, strategies[reference_index], num_times, num_paths)

    distances = calculate_strategy_distances_gray(
        strategies, 
        strategy_names, 
        sample_paths, 
        payoff, 
        cost, 
        dt,
        predictions,  # Use predictions instead of mean_values
        reference_index;  # Add reference_index
        initial_mode=initial_mode
    )
    
    # Create summary
    summary_df = get_summary_df(strategy_names, mean_values, normalized_mean_values, 
                              predictions, differences,original_mean_values)
    
    return (
        strategies = strategies,
        accumulated_values = accumulated_values,
        mean_values = mean_values,
        strategy_names = strategy_names,
        reference_values = reference_values,
        reference_index = reference_index,
        diff_limits = diff_limits,
        predictions = predictions,
        differences = differences,
        summary_df = summary_df,
        times = times,
        num_paths = num_paths,
        strat_dist = distances
    )
end

function plot_strategy_analysis(analysis_data)
    # Create strategy overview
    fig_overview = create_strategy_overview_figure(
        analysis_data.strategies,
        analysis_data.times,
        analysis_data.strategy_names
    )
    
    # Create value comparison
    fig_comparison = create_value_comparison_figure(
        analysis_data.accumulated_values,
        analysis_data.times,
        analysis_data.strategy_names,
        analysis_data.reference_values
    )
    
    return fig_overview, fig_comparison
end


function calculate_time_series_loss(strategy_values, reference_values, times)
    # Calculate scale factors for each time step
    σ_t = [std(reference_values[t, :]) for t in 1:size(reference_values,1)]
    μ_t = [mean(reference_values[t, :]) for t in 1:size(reference_values,1)]
    
    # Compute normalized loss for each time step
    loss = zeros(length(times))
    for t in 1:length(times)
        ref_val = μ_t[t]
        strategy_val = mean(strategy_values[t, :])
        scale = max(σ_t[t], abs(ref_val)/10) # Prevent division by very small values
        
        # Normalized difference scaled by local variation
        loss[t] = (ref_val - strategy_val) / scale
    end
    
    return loss
end


function get_summary_df(strategy_names, mean_values, normalized_mean_values, predictions, differences,original_mean_values)
    comprehensive_summary_df = DataFrame(
        Strategy = strategy_names,
        NormalizedFinalValue = normalized_mean_values[end, :],
        NonNormalizedFinalValue = original_mean_values[end, :],
        PredictedValue = predictions,
        DifferenceFromOptimal = differences
    )
    sort!(comprehensive_summary_df, :NormalizedFinalValue, rev=true)
    comprehensive_summary_df.Rank = 1:length(strategy_names)
    return comprehensive_summary_df
end

function compute_reference_metrics(accumulated_values, strategy_names)
    reference_index = findfirst(name -> name == "a posteriori", strategy_names)
    isnothing(reference_index) && return nothing, nothing, nothing
    
    reference_values = accumulated_values[reference_index]
    all_diffs = Float64[]
    
    for (i, values) in enumerate(accumulated_values)
        if i != reference_index
            mean_val = vec(mean(values, dims=2))
            ref_mean = vec(mean(reference_values, dims=2))
            diff_percent = 100 * (mean_val .- ref_mean) ./ ref_mean
            append!(all_diffs, diff_percent)
        end
    end
    
    diff_min, diff_max = extrema(all_diffs)
    diff_range = diff_max - diff_min
    diff_limits = (diff_min - 0.05 * diff_range, diff_max + 0.05 * diff_range)
    
    return reference_values, reference_index, diff_limits
end

function compute_strategy_data(Vs, other_strategies, sample_paths, times, payoff,
    cost, dt, initial_mode,J)
   num_strategies = length(Vs) + length(other_strategies)
   strategies = Vector{Matrix{Int}}(undef, num_strategies)
   accumulated_values = Vector{Matrix{Float64}}(undef, num_strategies)
   mean_values = Matrix{Float64}(undef, length(times), num_strategies)
   strategy_names = Vector{String}(undef, num_strategies)

   for (strategy_index, V) in enumerate(Vs)
       predicted_values, actual_strategy = compute_strategy_for_value(
           V, sample_paths, times, cost, dt, initial_mode,J)
       
       strategies[strategy_index] = actual_strategy
       strategy_names[strategy_index] = V.name
       
       acc_values = calculate_accumulated_value(
           actual_strategy, sample_paths, times, payoff, cost, dt;
           initial_mode=initial_mode)
       accumulated_values[strategy_index] = acc_values  # Remove reversal
       mean_values[:, strategy_index] = vec(mean(acc_values, dims=2))
   end

   for (i, (strategy, name)) in enumerate(other_strategies)
       index = length(Vs) + i
       strategies[index] = strategy
       strategy_names[index] = name
       
       acc_values = calculate_accumulated_value(
           strategy, sample_paths, times, payoff, cost, dt;
           initial_mode=initial_mode)
       accumulated_values[index] = acc_values  # Remove reversal
       mean_values[:, index] = vec(mean(acc_values, dims=2))
   end

   return strategies, accumulated_values, mean_values, strategy_names
end

function compute_differences(strategies, optimal_strategies, num_times, num_paths)
    num_strategies = length(strategies)
    differences = Vector{Float64}(undef, num_strategies)

    for i in 1:num_strategies
        diff_mode = zeros(10)  # Changed to 4 modes
        total_steps = 0

        for t in 1:num_times, path in 1:num_paths
            optimal_mode = optimal_strategies[t, path]
            strategy_mode = strategies[i][t, path]
            if optimal_mode != strategy_mode
                diff_mode[optimal_mode] += 1
            end
            total_steps += 1
        end

        # Normalize by total steps to get proportions
        diff_mode ./= total_steps
        
        # Calculate Euclidean distance from perfect match
        differences[i] = sqrt(sum(diff_mode.^2))
    end

    return differences
end

function compute_predictions(Vs, other_strategies, sample_paths, times, initial_mode, original_mean_values)
    num_strategies = length(Vs) + length(other_strategies)
    predictions = Vector{Float64}(undef, num_strategies)

    for (i, V) in enumerate(Vs)
        if V.name ∉ ["greedy", "a posteriori"]
            initial_state = sample_paths[:, 1, :]
            predictions[i] = mean([V(initial_state[:, j], times[1], initial_mode)[1] 
                                 for j in 1:size(sample_paths, 3)])
        else
            predictions[i] = original_mean_values[end, i]
        end
    end

    for (i, (strategy, name)) in enumerate(other_strategies)
        index = length(Vs) + i
        if name ∉ ["greedy", "a posteriori"]
            initial_state = sample_paths[:, 1, :]
            predictions[index] = mean([V(initial_state[:, j], times[1], initial_mode)[1] 
                                     for j in 1:size(sample_paths, 3)])
        else
            predictions[index] = original_mean_values[end, index]
        end
    end

    return predictions
end

function compute_strategy_for_value(V, sample_paths, times, cost, dt, initial_mode,J)
    num_times, num_paths = size(sample_paths, 2), size(sample_paths, 3)
    predicted_values = zeros(num_times, num_paths)
    actual_strategy = zeros(Int, num_times, num_paths)

    Threads.@threads for path in 1:num_paths
        current_mode = initial_mode
        for t in 1:num_times
            state = sample_paths[:, t, path]
            if t < num_times
                predicted_values[t, path] = V(state, times[t], current_mode)[1]
                best_mode = choose_best_mode(V, state, times[t], current_mode, cost, dt,J)
                actual_strategy[t, path] = best_mode
                current_mode = best_mode
            else
                predicted_values[t, path] = 0
                actual_strategy[t, path] = current_mode
            end
        end
    end
    
    return predicted_values, actual_strategy
end


"""
    compute_mode_proportions(strategy_matrix, num_times, num_paths, num_modes)

Compute the proportion of paths in each mode at each time point.

# Arguments
- `strategy_matrix`: Matrix of mode indices [time × paths]
- `num_times`: Number of time points
- `num_paths`: Number of paths
- `num_modes`: Number of possible modes

# Returns 
- `mode_props`: Matrix of proportions [time × modes]
"""

function compute_mode_proportions(strategy_matrix, num_times, num_paths,num_modes)
    mode_props = zeros(num_times, num_modes)  # Changed to 4 modes
    for t in 1:num_times
        for mode in 1:num_modes  # Changed to 4 modes
            mode_props[t, mode] = count(==(mode), view(strategy_matrix, t, :)) / num_paths
        end
    end
    return mode_props
end



function calculate_strategy_distances_gray(strategies, strategy_names, sample_paths,
     payoff, cost, dt, mean_values, reference_index; initial_mode=1)
    n_strategies = length(strategies)
    
    # 1. Decision Distance Matrix (disagreement percentage)
    decision_distance = zeros(n_strategies, n_strategies)
    
    # 2. Value Distance (normalized difference in final accumulated value)
    value_distance = zeros(n_strategies, n_strategies)
    
    # Calculate accumulated values for each strategy
    accumulated_values = []
    for (i, strategy) in enumerate(strategies)
        values = calculate_accumulated_value(
            strategy, sample_paths, 1:size(strategy, 1), 
            payoff, cost, dt; initial_mode=initial_mode)
        
        if all(isfinite.(values))
            push!(accumulated_values, values)
        else
            @warn "Strategy $(strategy_names[i]) has non-finite values"
            push!(accumulated_values, zeros(size(values)))
        end
    end
    
    # Calculate final mean values for each strategy (last timepoint)
    final_mean_values = [mean(values[end, :]) for values in accumulated_values]
    
    # Calculate prediction accuracy for each strategy at the final timepoint
    prediction_error = zeros(n_strategies)
    for i in 1:n_strategies
        actual_value = final_mean_values[i]  # Actual value at final timepoint
        predicted_value = mean_values[i]     # Predicted value from value function
        
        if isfinite(actual_value) && isfinite(predicted_value) && abs(actual_value) > 1e-10
            # Calculate relative error (smaller is better)
            prediction_error[i] = abs(predicted_value - actual_value) #/ abs(actual_value)
        else
            prediction_error[i] = NaN
        end
    end

    # Create dataframe with strategy names and performance metrics
    
    # Calculate pairwise distances (same as before)
    for i in 1:n_strategies
        for j in i:n_strategies
            if i == j
                continue
            end
            
            # Calculate decision distance (disagreement percentage)
            disagreements = sum(strategies[i] .!= strategies[j])
            total_decisions = length(strategies[i])
            decision_distance[i, j] = disagreements / total_decisions
            decision_distance[j, i] = decision_distance[i, j]
            
            # Calculate value distance with sign
            value_i = final_mean_values[i]
            value_j = final_mean_values[j]
            
            if isfinite(value_i) && isfinite(value_j) && abs(value_i) > 1e-10 && abs(value_j) > 1e-10
                relative_diff = abs(value_i - value_j) / max(abs(value_i), abs(value_j))
                value_distance[i, j] = value_i > value_j ? relative_diff : -relative_diff
                value_distance[j, i] = -value_distance[i, j]
            else
                value_distance[i, j] = NaN
                value_distance[j, i] = NaN
            end
        end
    end
    

    # Create a single dataframe with strategy names and distances to reference
    distances_df = DataFrame(
        Strategy = String[], 
        Decision_Distance_To_Reference = Float64[], 
        Prediction_Error = Float64[],
        Prediction_Accuracy = Float64[]
    )

    # Only proceed if we have a valid reference index
    if reference_index !== nothing
        ref_value = final_mean_values[reference_index]
        
        # Calculate distance values for each strategy relative to reference
        for i in 1:n_strategies
            # Skip the reference itself
            if i == reference_index
                continue
            end
            
            # Decision distance directly to reference
            decision_dist_to_ref = decision_distance[i, reference_index]
            
            # For greedy strategy, set prediction error to 0
            pred_error = strategy_names[i] == "greedy" ? 0.0 : 
                (abs(prediction_error[i]) / abs(final_mean_values[i]))
            pred_accuracy = 1/(1+pred_error)
            
            # Add row to combined dataframe
            push!(distances_df, (strategy_names[i], decision_dist_to_ref, pred_error,pred_accuracy))
        end
    else
        # If no reference index, just create empty dataframe with a note
        push!(distances_df, ("No reference strategy found", NaN, NaN))
    end


    # Create visualization with paper-friendly dimensions
    # Paper-friendly dimensions with aspect ratio closer to typical journal column width
    fig = Figure(size=(800, 1000), backgroundcolor=:white)
    
    # Create a grid layout with 3 rows and 2 columns for better paper layout
    gl = fig[1, 1] = GridLayout()
    
    # Use colorblind-friendly colormaps
    decision_cmap = :viridis
    value_cmap = :RdBu
    
    # Helper function to determine text color based on background luminance
    function get_text_color(val, colormap)
        rgb = get(colormap, val)
        luminance = 0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b
        return luminance < 0.55 ? :white : :black
    end
    
    # Title row
    Label(gl[1, 1:2], "Strategy Analysis", 
          fontsize=18, font=:bold)
    
    # REORGANIZED LAYOUT:
    # First heatmap - Decision Distance (now in row 2, spanning both columns)
    ax1 = Axis(gl[2, 1], 
              title="Decision Distance (% Different Choices)",
              xticks=(1:n_strategies, strategy_names),
              yticks=(1:n_strategies, strategy_names),
              titlefont=:bold,
              titlegap=8,
              xticklabelrotation=45,
              xticklabelalign=(:right, :center))
    
    ax1.bottomspinevisible = true
    ax1.spinewidth = 1.5
    
    colorrange = (0, maximum(filter(x -> !isnan(x) && x != 0, decision_distance)))
    hm1 = heatmap!(ax1, decision_distance, colormap=decision_cmap, colorrange=colorrange)
    Colorbar(gl[2, 2], hm1, width=15, ticklabelsize=10, labelpadding=5)
    
    # Add text labels
    for i in 1:n_strategies, j in 1:n_strategies
        if i == j
            poly!(ax1, [Point(j-0.5, i-0.5), Point(j+0.5, i-0.5), 
                       Point(j+0.5, i+0.5), Point(j-0.5, i+0.5)], 
                 color=(:lightgray, 0.3), strokewidth=1)
            text!(ax1, j, i, text="—",
                 color=:darkgray, align=(:center, :center), 
                 fontsize=12, font=:bold)
        else
            val = decision_distance[i,j]
            if maximum(filter(x -> !isnan(x) && x != 0, decision_distance)) > 0
                norm_val = val / maximum(filter(x -> !isnan(x) && x != 0, decision_distance))
            else
                norm_val = 0.0
            end
            text_color = get_text_color(norm_val, cgrad(decision_cmap))
            text!(ax1, j, i, text="$(round(Int, val*100))%",
                 color=text_color, align=(:center, :center), 
                 fontsize=10, font=:bold)
        end
    end
    
    # Second heatmap - Value Distance (now in row 3, spanning both columns)
    ax2 = Axis(gl[3, 1], 
              title="Value Distance (% Performance Difference)",
              xticks=(1:n_strategies, strategy_names),
              yticks=(1:n_strategies, strategy_names),
              titlefont=:bold,
              titlegap=8,
              xticklabelrotation=45,
              xticklabelalign=(:right, :center))
    
    ax2.bottomspinevisible = true
    ax2.spinewidth = 1.5
    
    value_distance_plot = copy(value_distance)
    value_distance_plot[isnan.(value_distance_plot)] .= 0
    
    max_abs_val = maximum(abs.(filter(x -> !isnan(x) && x != 0, value_distance)))
    colorrange_val = (-max_abs_val, max_abs_val)
    
    hm2 = heatmap!(ax2, value_distance_plot, colormap=value_cmap, colorrange=colorrange_val)
    Colorbar(gl[3, 2], hm2, width=15, ticklabelsize=10, labelpadding=5)
    
    for i in 1:n_strategies, j in 1:n_strategies
        if i == j
            poly!(ax2, [Point(j-0.5, i-0.5), Point(j+0.5, i-0.5), 
                       Point(j+0.5, i+0.5), Point(j-0.5, i+0.5)], 
                 color=(:lightgray, 0.3), strokewidth=1)
            text!(ax2, j, i, text="—",
                 color=:darkgray, align=(:center, :center), 
                 fontsize=12, font=:bold)
        else
            val = value_distance[i,j]
            if isnan(val)
                text!(ax2, j, i, text="N/A",
                     color=:black, align=(:center, :center), 
                     fontsize=10, font=:bold)
            else
                max_abs_val = maximum(abs.(filter(x -> !isnan(x) && x != 0, value_distance)))
                if max_abs_val > 0
                    norm_val = (val + max_abs_val) / (2 * max_abs_val)
                else
                    norm_val = 0.5
                end
                
                text_color = get_text_color(norm_val, cgrad(value_cmap))
                
                sign_prefix = val > 0 ? "+" : ""
                text!(ax2, j, i, text="$(sign_prefix)$(round(Int, val*100))%",
                     color=text_color, align=(:center, :center), 
                     fontsize=10, font=:bold)
            end
        end
    end
    
    # Bottom row: Side by side bar chart and ranking table
    # Create a new sublayout for the bottom row
    bottom_layout = gl[4, 1:2] = GridLayout()
    
    # Third visualization - Prediction Error Bar Chart (now in bottom left)
    ax3 = Axis(bottom_layout[1, 1], 
              title="Prediction Error (%)",
              xlabel="Strategy",
              ylabel="Relative Error (%)",
              titlefont=:bold,
              titlegap=8)
    
    # Sort strategies by prediction accuracy
    sorted_indices = sortperm(prediction_error)
    sorted_strategies = strategy_names[sorted_indices]
    sorted_errors = prediction_error[sorted_indices]
    
    # Create bar plot
    bars = barplot!(ax3, 1:n_strategies, sorted_errors .* 100, 
            color=:steelblue, strokecolor=:black, strokewidth=1)
    
    # Add value labels on top of each bar with reduced distance
    for (i, val) in enumerate(sorted_errors)
        if !isnan(val)
            text!(ax3, i, val * 100 + 0.5, 
                 text="$(round(val * 100, digits=1))%", 
                 fontsize=11, align=(:center, :bottom))
        else
            text!(ax3, i, 0.5, 
                 text="N/A", 
                 fontsize=11, align=(:center, :bottom))
        end
    end
    
    # Set xticks to show strategy names
    ax3.xticks = (1:n_strategies, sorted_strategies)
    ax3.xticklabelrotation = 45
    ax3.xticklabelalign = (:right, :center)
    ax3.xticklabelsize = 11
    
    # Filter out "a posteriori" and "greedy" strategies for the ranking table
    exclude_patterns = ["a posteriori", "greedy"]
    filtered_indices = findall(idx -> !any(contains.(lowercase.(strategy_names[idx]), exclude_patterns)), 1:n_strategies)
    
    filtered_strategy_names = strategy_names[filtered_indices]
    n_filtered = length(filtered_indices)
    
    # Calculate filtered rankings
    filtered_decision_ranks = zeros(Int, n_filtered)
    filtered_value_ranks = zeros(Int, n_filtered)
    filtered_prediction_ranks = zeros(Int, n_filtered)
    
    # Calculate rankings using the same logic as before
    decision_distances = zeros(n_strategies)
    if reference_index !== nothing
        for i in 1:n_strategies
            decision_distances[i] = i == reference_index ? 0 : decision_distance[i, reference_index]
        end
    else
        for i in 1:n_strategies
            decision_distances[i] = mean(filter(x -> !isnan(x) && x != 0, decision_distance[i, :]))
        end
    end
    decision_ranks = sortperm(decision_distances)
    value_ranks = sortperm(final_mean_values, rev=true)
    prediction_ranks = sortperm(prediction_error)
    
    for (i, orig_idx) in enumerate(filtered_indices)
        filtered_temp_dec = sortperm(decision_distances[filtered_indices])
        filtered_temp_val = sortperm(final_mean_values[filtered_indices], rev=true)
        filtered_temp_pred = sortperm(prediction_error[filtered_indices])
        
        filtered_decision_ranks[i] = findfirst(x -> x == i, filtered_temp_dec)
        filtered_value_ranks[i] = findfirst(x -> x == i, filtered_temp_val)
        filtered_prediction_ranks[i] = findfirst(x -> x == i, filtered_temp_pred)
    end
    
    # Calculate average ranks for filtered strategies
    filtered_avg_ranks = zeros(Float64, n_filtered)
    for i in 1:n_filtered
        filtered_avg_ranks[i] = (filtered_decision_ranks[i] + filtered_value_ranks[i] + filtered_prediction_ranks[i]) / 3.0
    end
    
    # Create a ranking table (now in bottom right)
    ax4 = Axis(bottom_layout[1, 2], 
              title="Strategy Rankings",
              titlefont=:bold,
              titlegap=8)
    
    hidedecorations!(ax4)
    hidespines!(ax4)
    
    # Create a more compact table-like visualization for rankings
    cell_height = 0.7
    cell_width = 0.7
    
    # Define colors for ranking
    ranking_colors = cgrad(:viridis, n_strategies, categorical=true)
    
    # Draw table header with smaller font
    text!(ax4, 1, n_filtered + 1, text="Strategy", 
         fontsize=10, font=:bold, align=(:center, :center))
    text!(ax4, 2, n_filtered + 1, text="Decision", 
         fontsize=10, font=:bold, align=(:center, :center))
    text!(ax4, 3, n_filtered + 1, text="Value", 
         fontsize=10, font=:bold, align=(:center, :center))
    text!(ax4, 4, n_filtered + 1, text="Predict", 
         fontsize=10, font=:bold, align=(:center, :center))
    text!(ax4, 5, n_filtered + 1, text="Avg", 
         fontsize=10, font=:bold, align=(:center, :center))
    
    # Sort filtered strategies by average rank
    filtered_avg_rank_sorted = sortperm(filtered_avg_ranks)
    
    # Draw table cells with colored rankings (more compact)
    for (row, i) in enumerate(filtered_avg_rank_sorted)
        orig_idx = filtered_indices[i]
        
        # Strategy name cell - increased cell size and text size
        poly!(ax4, [Point(0.5, n_filtered-row+0.5), Point(1.5, n_filtered-row+0.5), 
                   Point(1.5, n_filtered-row+1.5), Point(0.5, n_filtered-row+1.5)], 
             color=:white, strokecolor=:black, strokewidth=1)
        text!(ax4, 1, n_filtered - row + 1, text=filtered_strategy_names[i], 
             fontsize=11, align=(:center, :center))
        
        # Decision rank cell
        decision_rank = filtered_decision_ranks[i]
        cell_color = ranking_colors[decision_rank]
        poly!(ax4, [Point(1.5, n_filtered-row+0.5), Point(2.5, n_filtered-row+0.5), 
                   Point(2.5, n_filtered-row+1.5), Point(1.5, n_filtered-row+1.5)], 
             color=cell_color, strokecolor=:black, strokewidth=1)
        text!(ax4, 2, n_filtered - row + 1, text="$(decision_rank)", 
             fontsize=11, align=(:center, :center), color=get_text_color(decision_rank/n_strategies, ranking_colors))
        
        # Value rank cell
        value_rank = filtered_value_ranks[i]
        cell_color = ranking_colors[value_rank]
        poly!(ax4, [Point(2.5, n_filtered-row+0.5), Point(3.5, n_filtered-row+0.5), 
                   Point(3.5, n_filtered-row+1.5), Point(2.5, n_filtered-row+1.5)], 
             color=cell_color, strokecolor=:black, strokewidth=1)
        text!(ax4, 3, n_filtered - row + 1, text="$(value_rank)", 
             fontsize=11, align=(:center, :center), color=get_text_color(value_rank/n_strategies, ranking_colors))
        
        # Prediction rank cell
        prediction_rank = filtered_prediction_ranks[i]
        cell_color = ranking_colors[prediction_rank]
        poly!(ax4, [Point(3.5, n_filtered-row+0.5), Point(4.5, n_filtered-row+0.5), 
                   Point(4.5, n_filtered-row+1.5), Point(3.5, n_filtered-row+1.5)], 
             color=cell_color, strokecolor=:black, strokewidth=1)
        text!(ax4, 4, n_filtered - row + 1, text="$(prediction_rank)", 
             fontsize=11, align=(:center, :center), color=get_text_color(prediction_rank/n_strategies, ranking_colors))
        
        # Average rank cell
        avg_rank = filtered_avg_ranks[i]
        cell_color = ranking_colors[round(Int, avg_rank)]
        poly!(ax4, [Point(4.5, n_filtered-row+0.5), Point(5.5, n_filtered-row+0.5), 
                   Point(5.5, n_filtered-row+1.5), Point(4.5, n_filtered-row+1.5)], 
             color=cell_color, strokecolor=:black, strokewidth=1)
        text!(ax4, 5, n_filtered - row + 1, text="$(round(avg_rank, digits=1))", 
             fontsize=11, align=(:center, :center), color=get_text_color(avg_rank/n_strategies, ranking_colors))
    end
    
    # Set axis limits to fit the table
    xlims!(ax4, 0, 6)
    ylims!(ax4, 0, n_filtered + 2)
    
    # Remove the bottom text as requested
    
    # Set relative sizes to give more space to matrices and bottom row
    rowsize!(gl, 1, Auto(0.05))  # Title row
    rowsize!(gl, 2, Auto(0.3))   # First matrix
    rowsize!(gl, 3, Auto(0.3))   # Second matrix
    rowsize!(gl, 4, Auto(0.35))  # Bottom row with bar chart and table - increased space
    
    colsize!(bottom_layout, 1, Auto(0.5))  # Bar chart
    colsize!(bottom_layout, 2, Auto(0.5))  # Ranking table
    
    # Add narrow gaps between elements to save space
    rowgap!(gl, 10)
    colgap!(gl, 10)
    
    return (
        decision = decision_distance, 
        value = value_distance,
        prediction_error = prediction_error,
        figure = fig,
        rankings = (
            decision = decision_ranks,
            value = value_ranks,
            prediction = prediction_ranks,
            average = sortperm((1:n_strategies) ./ decision_ranks + (1:n_strategies) ./ value_ranks + (1:n_strategies) ./ prediction_ranks)
        ),
        distances = distances_df
    )
end
# Main function - orchestrates the process but delegates to smaller functions
# function calculate_strategy_distances_gray(strategies, strategy_names, sample_paths, payoff, cost, dt, mean_values, reference_index; initial_mode=1)
#     # Calculate all metrics
#     accumulated_values = calculate_accumulated_values(strategies, strategy_names, sample_paths, payoff, cost, dt, initial_mode)
#     final_mean_values = [mean(values[end, :]) for values in accumulated_values]
    
#     # Calculate distances and errors
#     decision_distance = calculate_decision_distances(strategies)
#     value_distance = calculate_value_distances(final_mean_values)
#     prediction_error = calculate_prediction_errors(final_mean_values, mean_values)
    
#     # Calculate rankings
#     rankings = calculate_rankings(decision_distance, final_mean_values, prediction_error, reference_index)
    
#     # Create visualization
#     fig = create_strategy_visualization(
#         strategies, strategy_names, decision_distance, value_distance, 
#         prediction_error, rankings, reference_index)
    
#     # Return results
#     return (
#         decision = decision_distance, 
#         value = value_distance,
#         prediction_error = prediction_error,
#         figure = fig,
#         rankings = rankings
#     )
# end

# Calculate accumulated values for each strategy
function calculate_accumulated_values(strategies, strategy_names, sample_paths, payoff, cost, dt, initial_mode)
    accumulated_values = []
    
    for (i, strategy) in enumerate(strategies)
        values = calculate_accumulated_value(
            strategy, sample_paths, 1:size(strategy, 1), 
            payoff, cost, dt; initial_mode=initial_mode)
        
        if all(isfinite.(values))
            push!(accumulated_values, values)
        else
            @warn "Strategy $(strategy_names[i]) has non-finite values"
            push!(accumulated_values, zeros(size(values)))
        end
    end
    
    return accumulated_values
end

# Calculate decision distances (disagreement percentage)
function calculate_decision_distances(strategies)
    n_strategies = length(strategies)
    decision_distance = zeros(n_strategies, n_strategies)
    
    for i in 1:n_strategies
        for j in i:n_strategies
            if i == j
                continue
            end
            
            disagreements = sum(strategies[i] .!= strategies[j])
            total_decisions = length(strategies[i])
            decision_distance[i, j] = disagreements / total_decisions
            decision_distance[j, i] = decision_distance[i, j]
        end
    end
    
    return decision_distance
end

# Calculate value distances (normalized difference in final values)
function calculate_value_distances(final_mean_values)
    n_strategies = length(final_mean_values)
    value_distance = zeros(n_strategies, n_strategies)
    
    for i in 1:n_strategies
        for j in i:n_strategies
            if i == j
                continue
            end
            
            value_i = final_mean_values[i]
            value_j = final_mean_values[j]
            
            if isfinite(value_i) && isfinite(value_j) && abs(value_i) > 1e-10 && abs(value_j) > 1e-10
                relative_diff = abs(value_i - value_j) / max(abs(value_i), abs(value_j))
                value_distance[i, j] = value_i > value_j ? relative_diff : -relative_diff
                value_distance[j, i] = -value_distance[i, j]
            else
                value_distance[i, j] = NaN
                value_distance[j, i] = NaN
            end
        end
    end
    
    return value_distance
end

# Calculate prediction errors
function calculate_prediction_errors(final_mean_values, mean_values)
    n_strategies = length(final_mean_values)
    prediction_error = zeros(n_strategies)
    
    for i in 1:n_strategies
        actual_value = final_mean_values[i]
        predicted_value = mean_values[i]
        
        if isfinite(actual_value) && isfinite(predicted_value) && abs(actual_value) > 1e-10
            prediction_error[i] = abs(predicted_value - actual_value) / abs(actual_value)
        else
            prediction_error[i] = NaN
        end
    end
    
    return prediction_error
end

# Calculate rankings
function calculate_rankings(decision_distance, final_mean_values, prediction_error, reference_index)
    n_strategies = length(final_mean_values)
    
    # Calculate decision distance from reference (or average)
    decision_distances = zeros(n_strategies)
    if reference_index !== nothing
        for i in 1:n_strategies
            decision_distances[i] = i == reference_index ? 0 : decision_distance[i, reference_index]
        end
    else
        for i in 1:n_strategies
            decision_distances[i] = mean(filter(x -> !isnan(x) && x != 0, decision_distance[i, :]))
        end
    end
    
    # Calculate rankings
    decision_ranks = sortperm(decision_distances)
    value_ranks = sortperm(final_mean_values, rev=true)
    prediction_ranks = sortperm(prediction_error)
    
    # Calculate average rankings
    avg_scores = (1:n_strategies) ./ decision_ranks + (1:n_strategies) ./ value_ranks + (1:n_strategies) ./ prediction_ranks
    average_ranks = sortperm(avg_scores)
    
    return (
        decision = decision_ranks,
        value = value_ranks,
        prediction = prediction_ranks,
        average = average_ranks
    )
end

# Create filtered rankings
function create_filtered_rankings(filtered_indices, rankings, decision_distance, 
                               strategy_names, prediction_error, reference_index)
    n_filtered = length(filtered_indices)
    
    filtered_decision_ranks = zeros(Int, n_filtered)
    filtered_value_ranks = zeros(Int, n_filtered)
    filtered_prediction_ranks = zeros(Int, n_filtered)
    
    # Extract original indices for each rank
    for (i, orig_idx) in enumerate(filtered_indices)
        # Find position in each ranking
        filtered_decision_ranks[i] = findfirst(x -> x == orig_idx, rankings.decision)
        filtered_value_ranks[i] = findfirst(x -> x == orig_idx, rankings.value)
        filtered_prediction_ranks[i] = findfirst(x -> x == orig_idx, rankings.prediction)
    end
    
    # Calculate average ranks
    filtered_avg_ranks = zeros(Float64, n_filtered)
    for i in 1:n_filtered
        filtered_avg_ranks[i] = (filtered_decision_ranks[i] + filtered_value_ranks[i] + filtered_prediction_ranks[i]) / 3.0
    end
    
    return (
        decision = filtered_decision_ranks,
        value = filtered_value_ranks,
        prediction = filtered_prediction_ranks,
        average = filtered_avg_ranks,
        avg_sorted = sortperm(filtered_avg_ranks)
    )
end

# Determine text color based on background luminance
function get_text_color(val, colormap)
    rgb = get(colormap, val)
    luminance = 0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b
    return luminance < 0.55 ? :white : :black
end

# Create the complete visualization with grayscale-friendly design
function create_strategy_visualization(strategies, strategy_names, decision_distance, value_distance, 
                                       prediction_error, rankings, reference_index)
    n_strategies = length(strategies)
    
    # Create figure with paper-friendly dimensions
    fig = Figure(size=(800, 1000), backgroundcolor=:white)
    gl = fig[1, 1] = GridLayout()
    
    # Add title
    Label(gl[1, 1:2], "Strategy Analysis", fontsize=18, font=:bold)
    
    # Create decision distance heatmap (grayscale)
    create_decision_heatmap!(gl[2, 1:2], decision_distance, strategy_names)
    
    # Create value distance heatmap (pattern + grayscale)
    create_value_heatmap!(gl[3, 1:2], value_distance, strategy_names)
    
    # Create bottom row with prediction error chart and rankings table
    bottom_layout = gl[4, 1:2] = GridLayout()
    create_prediction_error_chart!(bottom_layout[1, 1], prediction_error, strategy_names)
    create_rankings_table!(bottom_layout[1, 2], strategy_names, rankings, decision_distance, 
                          prediction_error, reference_index)
    
    # Set layout properties
    set_layout_properties!(gl, bottom_layout)
    
    return fig
end

# Create grayscale-friendly decision distance heatmap
function create_decision_heatmap!(pos, decision_distance, strategy_names)
    n_strategies = length(strategy_names)
    decision_cmap = :grays  # Use grayscale colormap instead of viridis
    
    ax = Axis(pos, 
        title="Decision Distance (% Different Choices)",
        xticks=(1:n_strategies, strategy_names),
        yticks=(1:n_strategies, strategy_names),
        titlefont=:bold,
        titlegap=8,
        xticklabelrotation=45,
        xticklabelalign=(:right, :center))
    
    ax.bottomspinevisible = true
    ax.spinewidth = 1.5
    
    colorrange = (0, maximum(filter(x -> !isnan(x) && x != 0, decision_distance)))
    hm = heatmap!(ax, decision_distance, colormap=decision_cmap, colorrange=colorrange)
    Colorbar(pos[1, 2], hm, width=15, ticklabelsize=10, labelpadding=5)
    
    # Add text labels and patterns for better distinction
    for i in 1:n_strategies, j in 1:n_strategies
        if i == j
            poly!(ax, [Point(j-0.5, i-0.5), Point(j+0.5, i-0.5), 
                      Point(j+0.5, i+0.5), Point(j-0.5, i+0.5)], 
                 color=(:white, 0.8), strokewidth=1, stroke=:black)
            text!(ax, j, i, text="—",
                 color=:black, align=(:center, :center), 
                 fontsize=12, font=:bold)
        else
            val = decision_distance[i,j]
            # Add diagonal hatching pattern based on value intensity
            if val > 0.001  # Only add pattern if there's a meaningful value
                # Add diagonal line pattern (density based on value)
                line_spacing = max(3, Int(round(20 * (1 - val))))
                for k in 1:line_spacing:20
                    lines!(ax, [j-0.5+k/20, j-0.5], [i-0.5, i-0.5+k/20], 
                          color=:black, linewidth=0.5, transparency=0.4)
                end
            end
            
            # Always use black text for maximum contrast
            text!(ax, j, i, text="$(round(Int, val*100))%",
                 color=:black, align=(:center, :center), 
                 fontsize=10, font=:bold, bgcolor=(:white, 0.6))
        end
    end
    
    return ax
end

# Create grayscale-friendly value distance heatmap with patterns
function create_value_heatmap!(pos, value_distance, strategy_names)
    n_strategies = length(strategy_names)
    value_cmap = :grays  # Use grayscale instead of RdBu
    
    ax = Axis(pos, 
        title="Value Distance (% Performance Difference)",
        xticks=(1:n_strategies, strategy_names),
        yticks=(1:n_strategies, strategy_names),
        titlefont=:bold,
        titlegap=8,
        xticklabelrotation=45,
        xticklabelalign=(:right, :center))
    
    ax.bottomspinevisible = true
    ax.spinewidth = 1.5
    
    value_distance_plot = copy(value_distance)
    value_distance_plot[isnan.(value_distance_plot)] .= 0
    
    max_abs_val = maximum(abs.(filter(x -> !isnan(x) && x != 0, value_distance)))
    colorrange_val = (0, max_abs_val)  # Only use intensity, not diverging colors
    
    hm = heatmap!(ax, abs.(value_distance_plot), colormap=value_cmap, colorrange=colorrange_val)
    Colorbar(pos[1, 2], hm, width=15, ticklabelsize=10, labelpadding=5)
    
    # Add text labels and directional patterns
    for i in 1:n_strategies, j in 1:n_strategies
        if i == j
            poly!(ax, [Point(j-0.5, i-0.5), Point(j+0.5, i-0.5), 
                      Point(j+0.5, i+0.5), Point(j-0.5, i+0.5)], 
                 color=(:white, 0.8), strokewidth=1, stroke=:black)
            text!(ax, j, i, text="—",
                 color=:black, align=(:center, :center), 
                 fontsize=12, font=:bold)
        else
            val = value_distance[i,j]
            if isnan(val)
                text!(ax, j, i, text="N/A",
                     color=:black, align=(:center, :center), 
                     fontsize=10, font=:bold, bgcolor=(:white, 0.8))
            else
                # Add pattern based on sign:
                # - Horizontal lines for positive values
                # - Vertical lines for negative values
                if abs(val) > 0.001
                    line_spacing = max(3, Int(round(20 * (1 - abs(val)/max_abs_val))))
                    if val > 0
                        # Horizontal lines for positive values
                        for k in 1:line_spacing:20
                            lines!(ax, [j-0.5, j+0.5], [i-0.5+k/20, i-0.5+k/20], 
                                  color=:black, linewidth=0.5, transparency=0.4)
                        end
                    else
                        # Vertical lines for negative values
                        for k in 1:line_spacing:20
                            lines!(ax, [j-0.5+k/20, j-0.5+k/20], [i-0.5, i+0.5], 
                                  color=:black, linewidth=0.5, transparency=0.4)
                        end
                    end
                end
                
                # Add sign indicator and value
                sign_prefix = val > 0 ? "+" : ""
                text!(ax, j, i, text="$(sign_prefix)$(round(Int, val*100))%",
                     color=:black, align=(:center, :center), 
                     fontsize=10, font=:bold, bgcolor=(:white, 0.6))
            end
        end
    end
    
    return ax
end

# Create grayscale-friendly prediction error bar chart with patterns
function create_prediction_error_chart!(pos, prediction_error, strategy_names)
    n_strategies = length(strategy_names)
    
    ax = Axis(pos, 
        title="Prediction Error (%)",
        xlabel="Strategy",
        ylabel="Relative Error (%)",
        titlefont=:bold,
        titlegap=8)
    
    # Sort strategies by prediction accuracy
    sorted_indices = sortperm(prediction_error)
    sorted_strategies = strategy_names[sorted_indices]
    sorted_errors = prediction_error[sorted_indices]
    
    # Create patterned bar plot
    for (i, val) in enumerate(sorted_errors)
        if !isnan(val)
            # Create the base bar with light gray
            barplot!(ax, [i], [val * 100], 
                    color=(:gray, 0.5), strokecolor=:black, strokewidth=1)
            
            # Add hatch pattern (diagonal lines)
            bar_height = val * 100
            for j in 1:3:30
                lines!(ax, [i-0.4+j/30, i-0.4], [0, bar_height*j/30], 
                      color=:black, linewidth=0.5, transparency=0.7)
                lines!(ax, [i+0.4-j/30, i+0.4], [0, bar_height*j/30], 
                      color=:black, linewidth=0.5, transparency=0.7)
            end
            
            # Add horizontal lines for additional pattern
            for j in 1:5:Int(ceil(bar_height))
                lines!(ax, [i-0.4, i+0.4], [j, j], 
                      color=:black, linewidth=0.5, transparency=0.5)
            end
            
            # Add value label with high contrast
            text!(ax, i, val * 100 + 0.5, 
                 text="$(round(val * 100, digits=1))%", 
                 fontsize=11, align=(:center, :bottom), color=:black)
        else
            # Empty bar with just a border for N/A values
            poly!(ax, [Point(i-0.4, 0), Point(i+0.4, 0), 
                      Point(i+0.4, 0.5), Point(i-0.4, 0.5)], 
                 color=(:white, 0.8), strokecolor=:black, strokewidth=1)
            
            text!(ax, i, 0.75, 
                 text="N/A", 
                 fontsize=11, align=(:center, :bottom), color=:black)
        end
    end
    
    # Set xticks to show strategy names
    ax.xticks = (1:n_strategies, sorted_strategies)
    ax.xticklabelrotation = 45
    ax.xticklabelalign = (:right, :center)
    ax.xticklabelsize = 11
    
    return ax
end

# Create grayscale-friendly rankings table
function create_rankings_table!(pos, strategy_names, rankings, decision_distance, prediction_error, reference_index)
    n_strategies = length(strategy_names)
    
    # Filter out specific strategies for the ranking table
    exclude_patterns = ["a posteriori", "greedy"]
    filtered_indices = findall(idx -> !any(contains.(lowercase.(strategy_names[idx]), exclude_patterns)), 1:n_strategies)
    
    filtered_strategy_names = strategy_names[filtered_indices]
    n_filtered = length(filtered_indices)
    
    # Create filtered rankings
    filtered_rankings = create_filtered_rankings(filtered_indices, rankings, decision_distance, 
                                              strategy_names, prediction_error, reference_index)
    
    # Create table visualization
    ax = Axis(pos, title="Strategy Rankings", titlefont=:bold, titlegap=8)
    hidedecorations!(ax)
    hidespines!(ax)
    
    # Draw table header with border
    poly!(ax, [Point(0.5, n_filtered+0.5), Point(5.5, n_filtered+0.5), 
               Point(5.5, n_filtered+1.5), Point(0.5, n_filtered+1.5)], 
         color=(:black, 0.1), strokecolor=:black, strokewidth=1.5)
    
    text!(ax, 1, n_filtered + 1, text="Strategy", fontsize=10, font=:bold, align=(:center, :center))
    text!(ax, 2, n_filtered + 1, text="Decision", fontsize=10, font=:bold, align=(:center, :center))
    text!(ax, 3, n_filtered + 1, text="Value", fontsize=10, font=:bold, align=(:center, :center))
    text!(ax, 4, n_filtered + 1, text="Predict", fontsize=10, font=:bold, align=(:center, :center))
    text!(ax, 5, n_filtered + 1, text="Avg", fontsize=10, font=:bold, align=(:center, :center))
    
    # Draw table cells with patterns
    draw_ranking_table_cells!(ax, filtered_rankings, filtered_strategy_names, n_filtered, n_strategies)
    
    # Set axis limits
    xlims!(ax, 0, 6)
    ylims!(ax, 0, n_filtered + 2)
    
    return ax
end

# Create filtered rankings 
# (using the original function, no changes neede

# Draw ranking table cells with grayscale patterns
function draw_ranking_table_cells!(ax, filtered_rankings, filtered_strategy_names, n_filtered, n_strategies)
    # Draw cells for each strategy, sorted by average rank
    for (row, i) in enumerate(filtered_rankings.avg_sorted)
        # Strategy name cell
        poly!(ax, [Point(0.5, n_filtered-row+0.5), Point(1.5, n_filtered-row+0.5), 
                  Point(1.5, n_filtered-row+1.5), Point(0.5, n_filtered-row+1.5)], 
             color=:white, strokecolor=:black, strokewidth=1)
        text!(ax, 1, n_filtered - row + 1, text=filtered_strategy_names[i], 
             fontsize=11, align=(:center, :center))
        
        # Decision rank cell
        decision_rank = filtered_rankings.decision[i]
        draw_grayscale_rank_cell!(ax, 2, n_filtered - row + 1, decision_rank, n_strategies)
        
        # Value rank cell
        value_rank = filtered_rankings.value[i]
        draw_grayscale_rank_cell!(ax, 3, n_filtered - row + 1, value_rank, n_strategies)
        
        # Prediction rank cell
        prediction_rank = filtered_rankings.prediction[i]
        draw_grayscale_rank_cell!(ax, 4, n_filtered - row + 1, prediction_rank, n_strategies)
        
        # Average rank cell
        avg_rank = filtered_rankings.average[i]
        draw_grayscale_rank_cell!(ax, 5, n_filtered - row + 1, round(Int, avg_rank), n_strategies, 
                      text="$(round(avg_rank, digits=1))")
    end
end

# Draw an individual rank cell with grayscale pattern
function draw_grayscale_rank_cell!(ax, x, y, rank, n_strategies; text=nothing)
    # Calculate shade of gray and pattern density based on rank
    # Lower rank (better) = darker shade + denser pattern
    gray_intensity = 0.9 - 0.7 * (n_strategies - rank) / n_strategies
    pattern_spacing = max(2, Int(round(10 * rank / n_strategies)))
    
    # Draw cell background
    poly!(ax, [Point(x-0.5, y-0.5), Point(x+0.5, y-0.5), 
              Point(x+0.5, y+0.5), Point(x-0.5, y+0.5)], 
         color=(gray_intensity, gray_intensity, gray_intensity), 
         strokecolor=:black, strokewidth=1)
    
    # Add pattern based on rank (better ranks get denser patterns)
    if rank <= n_strategies ÷ 3  # Top ranks - crosshatch pattern
        # Horizontal lines
        for j in -10:pattern_spacing:10
            lines!(ax, [x-0.5, x+0.5], [y-0.5+j/20, y-0.5+j/20], 
                  color=:black, linewidth=0.5, transparency=0.5)
        end
        # Vertical lines
        for j in -10:pattern_spacing:10
            lines!(ax, [x-0.5+j/20, x-0.5+j/20], [y-0.5, y+0.5], 
                  color=:black, linewidth=0.5, transparency=0.5)
        end
    elseif rank <= 2 * n_strategies ÷ 3  # Middle ranks - diagonal lines
        for j in -15:pattern_spacing:15
            lines!(ax, [x-0.5+j/15, x-0.5], [y-0.5, y-0.5+j/15], 
                  color=:black, linewidth=0.5, transparency=0.5)
            lines!(ax, [x+0.5-j/15, x+0.5], [y-0.5, y-0.5+j/15], 
                  color=:black, linewidth=0.5, transparency=0.5)
        end
    else  # Bottom ranks - dots or sparse lines
        for j in -10:pattern_spacing*2:10
            for k in -10:pattern_spacing*2:10
                scatter!(ax, [x-0.5+j/20], [y-0.5+k/20], 
                       color=:black, markersize=1, transparency=0.5)
            end
        end
    end
    
    # Display text with background for better readability
    text_to_display = isnothing(text) ? "$(rank)" : text
    text!(ax, x, y, text=text_to_display, 
         fontsize=11, align=(:center, :center), color=:black,
         bgcolor=(:white, 0.7))
end

# Set layout properties for the figure
function set_layout_properties!(gl, bottom_layout)
    # Set row sizes
    rowsize!(gl, 1, Auto(0.05))  # Title row
    rowsize!(gl, 2, Auto(0.3))   # First matrix
    rowsize!(gl, 3, Auto(0.3))   # Second matrix
    rowsize!(gl, 4, Auto(0.35))  # Bottom row
    
    # Set column sizes for bottom layout
    colsize!(bottom_layout, 1, Auto(0.5))  # Bar chart
    colsize!(bottom_layout, 2, Auto(0.5))  # Ranking table
    
    # Add gaps
    rowgap!(gl, 10)
    colgap!(gl, 10)
end

#include("blackwhite.jl")