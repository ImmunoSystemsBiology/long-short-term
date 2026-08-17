# simulation of the 2-element Windkessel problem
# as in Regazzoni et al.
using OrdinaryDiffEq, CairoMakie, Random, ProgressMeter

struct WindkesselPatient
    model_parameters::Vector{Float64}
    alpha::Float64
end

function windkessel2!(du, u, p, t, Q)
    Rp, Cp, Rd, Cd, Pvs = p

    Pp, Pd = u

    du[1] = (Pd - Pp) / (Rp * Cp) + Q(t) / Cp
    du[2] = (Pp - Pd) / (Rp * Cd) + (Pvs - Pd) / (Rd * Cd)

end

function slow_scale!(du, u, p, t)
    du[1] = 0.0
    du[2] = 0.0
    du[3] = p[1]*u[3]
    du[4] = -p[1]*u[4]
end

function get_windkessel_prob(patient::WindkesselPatient, Qf, tspan, u0, τ)
    # obtain model parameters at τ using the patient's model parameters and α
    if τ < 0.0
        error("τ must be non-negative")
    elseif iszero(τ)
        return ODEProblem((du, u, p, t) -> windkessel2!(du, u, p, t, Qf), u0, tspan, patient.model_parameters)
    end
    u0_long = patient.model_parameters[1:end-1]
    tspan_long = (0.0, τ)
    slow_prob = ODEProblem(slow_scale!, u0_long, tspan_long, [patient.alpha])
    parameters_τ = Array(solve(slow_prob, Tsit5()))[:, end]

    # simulate the Windkessel model with the obtained parameters
    patient_params = vcat(parameters_τ, patient.model_parameters[end])
    ODEProblem((du, u, p, t) -> windkessel2!(du, u, p, t, Qf), u0, tspan, patient_params)
end

function create_patient(rng, lower_bounds, upper_bounds, Pvs)
    variable_params = rand(rng, eltype(lower_bounds), size(LOWER_BOUNDS)) .* (upper_bounds .- lower_bounds) .+ lower_bounds
    model_parameters = vcat(variable_params, Pvs)
    alpha = rand(rng) * 1e-2
    return WindkesselPatient(model_parameters, alpha)
end

function simulate_dataset(
    rng, 
    n_patients, 
    Qf, 
    Pvs, 
    timepoints_long, 
    timepoints_short,
    timepoints_gt, 
    W_noise,
    lower_bounds,
    upper_bounds)

    patients = WindkesselPatient[]
    for i in 1:n_patients
        patient = create_patient(rng, lower_bounds, upper_bounds, Pvs)
        push!(patients, patient)
    end

    dataset = Dict{Symbol, Any}()
    dataset[:patients] = patients

    dataset[:measurements] = Vector{Matrix{Float64}}(undef, n_patients)
    dataset[:ground_truth] = Vector{Array{Float64,3}}(undef, n_patients)
    @showprogress for (i, patient) in enumerate(patients)
        trajectory = zeros(length(timepoints_long), length(timepoints_short))
        gts = zeros(length(timepoints_long), 2, length(timepoints_gt))
        for (j, τ) in enumerate(timepoints_long)
            prob = get_windkessel_prob(patient, Qf, (-8.0, last(timepoints_short)), [80.0, 80.0], τ)
            sol = Array(solve(prob, Tsit5(), saveat=timepoints_short, save_idxs=1:2))
            sol_true = Array(solve(prob, Tsit5(), saveat=timepoints_gt, save_idxs=1:2))
            gts[j, :, :] = sol_true
            trajectory[j, :] = sol[1, :] .+ randn(rng, length(timepoints_short)) .* W_noise
        end
        dataset[:measurements][i] = trajectory
        dataset[:ground_truth][i] = gts
    end

    return dataset
end


# TODO: example figure