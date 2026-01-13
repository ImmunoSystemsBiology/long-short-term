using OrdinaryDiffEq, Distributions, Random
include("model.jl")

function generate_initial_patients(
    number_of_patients::Int,
    parameter_names::Vector{Symbol},
    spread::Float64,
    tspan::Tuple{Float64, Float64},
    physiological_ranges::Vector{Tuple{Float64, Float64}}; maximum_tries = 100000)

    initial_patients = Vector{Tuple{Vector{Float64}, Vector{Float64}}}()
    parameters = yildirim_parameters()

    while length(initial_patients) < number_of_patients && maximum_tries > 0
        maximum_tries -= 1
    
        parameters_new = copy(parameters)
        for param in parameter_names
            parameters_new[parameter_index(param)] = rand(Uniform(1.0 - spread, 1.0 + spread)) * parameters[parameter_index(param)]
        end

        u0 = yildirim_state()

        # run model
        prob = ODEProblem(progression_ode!, u0, tspan, parameters_new)
        sol = Array(solve(prob, saveat=tspan[2]))[:, end]
       
        # check if in physiological ranges
        in_range = isinrange(sol, physiological_ranges)

        if in_range
            push!(initial_patients, (sol, parameters_new))
        end

        # if length(initial_patients) % 10 == 0 && length(initial_patients) > 0
        println("Generated $(length(initial_patients)) patients...")
        # end
    end

    return initial_patients

end

function isinrange(values, ranges)
    for i in eachindex(values)
        if values[i] < ranges[i][1] || values[i] > ranges[i][2]
            return false
        end
    end
    return true
end

tspan_ss = (0.0, 2000.0)
spread = 0.2
N_patients = 100
parameter_names = [:ffa_ramax, :cf, :ksi_infl, :ksi_ffa, :sgkd, :sgku, :hgp_bas, :k_infl, :alpha_a, :amax, :alpha_p, :pmax, :alpha_isr, :alpha_m, :mmax, :eg0]

parameter_indices = [parameter_index(p) for p in parameter_names]

ranges = [
    (70.0, 100.0),
    (8.0, 12.0),
    (300.0, 500.0),
    (0.7, 0.9),
    (800.0, 1200.0),
    (400.0, 600.0),
    (-Inf, Inf),
    (0.0, Inf)
]

patients = generate_initial_patients(N_patients, parameter_names, spread, tspan_ss, ranges)