# estimate parameters on the fast-scale model for the data
using JLD2, OrdinaryDiffEq, Random, ProgressMeter, Optimization, OptimizationOptimJL, QuasiMonteCarlo
using LineSearches
include("simulation.jl")

rng = MersenneTwister(42)
const SIM_LOWER_BOUNDS = [8.1757e-2, 3.1197e-2, 6.9006e-1, 1.1851]
const SIM_UPPER_BOUNDS = [1.7326e-1, 3.1464e-2, 1.1599, 2.1731]

const EST_LOWER_BOUNDS = [8.1757e-2, 3.1197e-2, 6.9006e-1, 1e-3]
const EST_UPPER_BOUNDS = [1.7326e-1, 3.1464e-2, 1.1599 * 50, 2.1731]
# Simulate dataset
Qf(t) = 100.0 * sin(2π * t / 0.8) + 100.0
Pvs = 5.0
t_long = 0:50
t_meas = range(0.0, 8.0, step=1e-2)

dataset = simulate_dataset(
    rng, 
    100, 
    Qf, 
    Pvs, 
    t_long, 
    t_meas, 
    range(0.0, 8.0, step=1e-3), 
    0.75,
    SIM_LOWER_BOUNDS,
    SIM_UPPER_BOUNDS
)

function objective(x, (Q, Pvs, Pp, timepoints))

    p_ode = vcat(x[1:4], Pvs)
    Pd0 = x[5]; Pp0 = Pp[1]
    σPd = x[6]
    prob = ODEProblem((du, u, p, t) -> windkessel2!(du, u, p, t, Q), [Pp0, Pd0], (0.0, last(timepoints)), p_ode)

    Pp_est = Array(solve(prob, Tsit5(), saveat=timepoints, save_idxs=1))

    error = sum(abs2, Pp_est - Pp)
    n = length(Pp)
    negloglik = (n/2)*log(2π*σPd^2) + (1/(2*σPd^2))*error
    return negloglik
end

function estimate_parameters(measurement, t_meas, rng, Q, Pvs, lower_bounds, upper_bounds; n_guesses=50)

    lb_u0 = 0.0; lb_sigma = 0.0
    ub_u0 = 200.0; ub_sigma = 2.0

    lower_bounds_full = vcat(lower_bounds, lb_u0, lb_sigma)
    upper_bounds_full = vcat(upper_bounds, ub_u0, ub_sigma)

    # Generate initial guesses using Latin Hypercube Sampling
    initial_guesses = QuasiMonteCarlo.sample(
        n_guesses, lower_bounds_full, upper_bounds_full, QuasiMonteCarlo.LatinHypercubeSample(rng)
    )

    objectives = Float64[]
    params = Vector{Float64}[]

    optfunc = OptimizationFunction(objective, Optimization.AutoForwardDiff())

    @showprogress for gidx = axes(initial_guesses, 2)
        guess = initial_guesses[:, gidx]
        optprob = OptimizationProblem(optfunc, guess, (Q, Pvs, measurement, t_meas),lb=lower_bounds_full, ub=upper_bounds_full)
        try
            optsol = solve(optprob, BFGS(linesearch=BackTracking()); maxiters=1000)
            push!(objectives, optsol.objective)
            push!(params, optsol.u)
        catch e
            @warn "Optimization failed for guess $guess: $e"

        end
    end

    if isempty(objectives)
        error("All optimization attempts failed.")
    end

    return params, objectives
end

result_params = Array{Float64, 3}(undef, length(dataset[:measurements]), length(t_long), 6)

for (patientidx, meas_traj) in enumerate(dataset[:measurements])

    for (measidx, meas) in enumerate(eachrow(meas_traj))
        println("Estimating parameters for patient $patientidx, measurement $measidx")
        params, objectives = estimate_parameters(
            meas, 
            t_meas, 
            rng, 
            Qf, 
            Pvs, 
            EST_LOWER_BOUNDS, 
            EST_UPPER_BOUNDS; 
            n_guesses=5
        )
        
        opt_params = params[argmin(objectives)]
        result_params[patientidx, measidx, :] = opt_params
    end
end

jldsave("data/windkessel_estimation_results.jld2", (params=result_params))

# TODO: make a figure to visualize the results
