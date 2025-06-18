
"""
    Model Abstractions Module

This module provides unified interfaces for different types of machine learning models
used in optimal switching problems, including neural networks (Lux), MLJ models,
and SimpleChains. It handles model creation, training, and prediction across
different time steps and operational modes.
"""

using Lux, Optimisers, Zygote, MLUtils
using FileIO
using Random
using JLD2: load
using DataFrames
using Printf


"""
    arrayoflux(model, rng, N, J, d; sc=true, base_lr=0.001f0, min_lr=1f-5, decay_type="step", decay_rate=3.0f0)

Create an array of Lux models for different time steps and modes, with learning rate scheduling.

# Arguments
- `model`: Base model architecture to replicate
- `rng`: Random number generator for initialization
- `N`: Number of time steps
- `J`: Number of modes
- `d`: Input dimension
- `sc`: Whether to use SimpleChains adaptor
- `base_lr`: Base learning rate
- `min_lr`: Minimum learning rate
- `decay_type`: Learning rate decay type ("step" or "exp")
- `decay_rate`: Factor controlling decay speed

# Returns
- 2D array of model training states and normalization parameters
"""
function arrayoflux(model, rng, N, J, d; 
                    sc=true, 
                    base_lr=0.001f0,
                    min_lr=1f-5,
                    decay_type="step",  # Changed default to "step"
                    decay_rate=3.0f0)
    
    gdev = gpu_device()
    
    # Create an array to hold all models
    model_array = Array{Any}(undef, N, J)
    
    # Calculate dimension-adjusted base learning rate
    dim_adjusted_lr = base_lr * min(1.0f0, max(0.3f0, 10.0f0/d))
    
    for n in 1:N
        # Calculate time-dependent factor (0.01 at n=1, 1.0 at n=N)
        progress = (n - 1) / (N - 1)  # 0 at n=1, 1 at n=N
        factor = 0.01f0 + 0.99f0 * progress
        
        # Apply the factor to the base learning rate
        # More conservative initial learning rate
        dim_adjusted_lr = base_lr * min(1.0f0, max(0.5f0, 5.0f0/d))

        # Or consider exponential decay instead of linear scaling
        opt = Optimisers.ADAM(base_lr * exp(-3.0 * (1.0 - progress)))        
        # Create optimizer with the calculated learning rate
        #opt = Optimisers.ADAM(current_lr)
        
        for j in 1:J
            if sc
                adaptor = ToSimpleChainsAdaptor(static(d))
                sc_model = adaptor(model)
                ps, st = Lux.setup(rng, sc_model)
                model_array[n, j] = (Lux.Training.TrainState(sc_model, ps, st, opt), 
                                   (zeros(Float32, d, 1), ones(Float32, d, 1)))
            else
                ps, st = Lux.setup(rng, model)
                model_array[n, j] = (Lux.Training.TrainState(model, ps, st, opt), 
                                   (zeros(Float32, d, 1), ones(Float32, d, 1)))
            end
        end
    end
    
    return model_array
end

function arrayofevos(model,N,J)
    [nothing for n = 1:N, j = 1:J]
end

function arrayofMLJ(model,N,J)
    Array{Any}(undef, N, J)
end


# function lux_create_optimiser(ps)
#     opt = Optimisers.ADAM(0.003f0)
#     return  Optimisers.setup(opt, ps)
# end


function lux_loss(model,ps,st,data)
    y_pred,st = Lux.apply(model,data[1],ps,st)
    mse_loss = mean(abs2, data[2] .- y_pred)
    return mse_loss, y_pred, st
end

"""
    LearningModel{Model,Name,PkgName}

A unified interface for different types of machine learning models.

# Fields
- `model`: The underlying model object or container
- `name`: The name of the model (used for logging and saving)
- `pkg`: The package/framework of the model (e.g., "sc", "lux", "MLJ")
"""
@with_kw struct LearningModel{Model,Name,PkgName}
    model::Model
    name::Name
    pkg::PkgName
end

"""
    (f::LearningModel)(x, time, mode)

Predict using a learning model at a specific time step and mode.

# Arguments
- `x`: Input features (state vector)
- `time`: Time step index 
- `mode`: Mode index

# Returns
- Vector of predictions
"""
function (f::LearningModel)(x, time, mode)
    if (size(x) |> length) == 1
        x = reshape(x,(size(x)[1],1))
    end
    d,K = size(x)
    if f.pkg == "sklearn"
        x=permutedims(x)
        return ScikitLearn.predict(f.model[time, mode], x)
    elseif f.pkg == "sc"
        tstate, (mean_val, std_val) = f.model[time,mode]
        model = tstate.model
        ps = tstate.parameters
        st = tstate.states
        dev_gpu = Lux.gpu_device()
        dev_cpu = Lux.cpu_device()
        x_norm = (x .- mean_val) ./ std_val
        return model(x_norm,ps,st)[1] |> Array |> vec
    elseif f.pkg == "lux"
        tstate, (mean_val, std_val) = f.model[time,mode]
        model = tstate.model
        ps = tstate.parameters
        st = tstate.states
        gdev = Lux.gpu_device()
        cdev = Lux.cpu_device()
        x_norm = (x .- mean_val) ./ std_val
        return model(x_norm |> gdev,ps,st)[1] |> cdev |> Array |> vec
    elseif f.pkg == "MLJ"
        #println("what")
        x = DataFrame(permutedims(x),:auto)
        return MLJ.predict(f.model[2][time,mode],x)
    else
        return [f.model(x, time, mode)]
    end
end

"""
    my_fit!(M::LearningModel, X, Y, time, j; payoffmodel=nothing, exp_params=nothing, use_switching_constraint=false)

Train a learning model for a specific time step and mode.

# Arguments
- `M`: The learning model to train
- `X`: Input features (state vectors)
- `Y`: Target values
- `time`: Time step index
- `j`: Mode index
- `payoffmodel`: Optional payoff model for constraint-based training
- `exp_params`: Optional experiment parameters 
- `use_switching_constraint`: Whether to apply switching constraints during training
"""
function my_fit!(M::LearningModel, X, Y, time, j; 
                payoffmodel=nothing, exp_params=nothing, use_switching_constraint=false)
    if M.pkg == "sklearn"
        X = permutedims(X)
        ScikitLearn.fit!(M.model[time, j], X, Y)
    elseif M.pkg == "sc"
        # Pass constraint parameters only when using lux/sc models
        lux_fit!(M, X, Y, time, j)
    elseif M.pkg == "lux"
        # Pass constraint parameters only when using lux/sc models
        lux_fit!(M, X, Y, time, j)
    elseif M.pkg == "MLJ"
        # For MLJ models, we would need a different approach to implement the constraint
        # For now, just fit the model as before
        X = permutedims(X)
        mach = machine(M.model[1], DataFrame(X, :auto), Y)
        M.model[2][time, j] = MLJ.fit!(mach, verbosity=0)
        
        # If you want to implement constraint for MLJ models, you could:
        # 1. Create a custom model type that wraps the original model
        # 2. Implement a post-processing step that adjusts predictions based on the constraint
    end
    return nothing
end

"""
    my_loss(learning_model, X, y, n, j)

Calculate the mean squared error loss for a model's predictions.

# Arguments
- `learning_model`: The model to evaluate
- `X`: Input features (state vectors)
- `y`: True target values
- `n`: Time step index
- `j`: Mode index

# Returns
- Mean squared error loss
"""
function my_loss(learning_model, X, y, n, j)
    # Get predictions from the model
    preds = learning_model(X, n, j)
    
    # Calculate mean squared error
    loss = (y .- preds) .^ 2
    return 0.5 * mean(loss)
end

function disp_loss(loss, name)
    @printf("%s loss %.4f\n", name, loss)
end

function display_loss(accuracy, loss)
    @printf("    training accuracy %.2f, loss %.4f\n", 100 * accuracy, loss)
end

function save_learningmodel_disk(learningmodel,type,dir)
    if type == "sklearn"
        name = learningmodel.name
        joblib.dump(learningmodel.model,dir*"/"*name*".pkl")
    elseif type == "sc"
        #models = learningmodel.model
        name = learningmodel.name
        save(dir*"/"*name*".jld2",Dict("learningmodel"=>learningmodel))
    end
end

function load_learningmodel_disk(name,dir)
    extension = splitext(name)[2]
    model_name = splitext(name)[1]
    if extension == ".pkl"
        models = joblib.load(dir*"/"*name)
        return LearningModel(models,model_name,"sklearn")
    elseif extension == ".jld2"
        model_dict = load(dir*"/"*name)
        learningmodel = model_dict["learning_model"]
        return learningmodel
    end
end

#function that loads models given their absolute paths
function load_models(paths)
    models = Any[FileIO.load(path)["learning_model"] for path in paths]
    return models
end



function save_tuplemodels_folder(learningmodels,dir)
    for learningmodel in learningmodels
        save_learningmodel_disk(learningmodel,learningmodel.pkg,dir)
    end
end

function load_folder_models(dir)
    models = readdir(dir)
    @show models
    models = [load_learningmodel_disk(model,dir) for model in models]
    return (models...,)
end
