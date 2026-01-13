# model
include("../edes_model.jl")
using Distributions, Random

function simulate_trajectory(param_values, individuals, Dmeal::Float64, local_timepoints::Vector{Float64}; 
    noise_distribution::Distribution = Normal(0.0, 0.0005), rng::AbstractRNG = Random.default_rng())

    # Initialize result array
    result = Array{Float64, 3}(undef, length(param_values), length(local_timepoints), 2) # 2 for glucose and insulin

    for (i, (individual, params)) in enumerate(zip(individuals, param_values))

        bw, Gb, Ib = individual.bw, individual.Gb, individual.Ib
        u0 = edes_initial_conditions(Gb, Ib)
        # Set parameters for the current individual and k5 value
        p = edes_parameters(Dmeal, bw, Gb, Ib; params...)

        # Define the ODE problem
        prob = ODEProblem(edesode, u0, (local_timepoints[1], local_timepoints[end]), p)

        # Solve the ODE
        sol = solve(prob, Tsit5(), saveat = local_timepoints)

        # Extract glucose and insulin values
        glucose_values = sol[2, :] .+ rand(rng, noise_distribution, length(local_timepoints)).*sol[2,:] # Add noise to glucose
        insulin_values = sol[3, :] .+ rand(rng, noise_distribution, length(local_timepoints)).*sol[3,:] # Add noise to insulin

        # Store results
        result[i, :, 1] = glucose_values
        result[i, :, 2] = insulin_values
    end

    return result
end
