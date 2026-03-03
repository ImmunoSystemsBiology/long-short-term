using JLD2, OrdinaryDiffEq, Random, QuasiMonteCarlo, Optimization, OptimizationOptimJL, Distributions, LineSearches, ProgressMeter, Statistics
include("../edes_model.jl")
using Colors, CairoMakie
# Load in the data
MIR_patients, LIR_patients, healthy_patients = jldopen("data/mir_lir_simulations.jld2", "r") do file
    (file["MIR_patients"], file["LIR_patients"], file["healthy_patients"])
end


function loglikelihood(parameters, (patient, model, Dmeal))

    k1, k5, k6, sigma_glucose, sigma_insulin = exp.(parameters)
    _p = edes_parameters(
        Dmeal,
        patient.bw,
        patient.glucose[1],
        patient.insulin[1];
        k1 = k1,
        k5 = k5,
        k6 = k6
    )
    _model = remake(model; p = _p)
    prediction = Array(solve(_model, Tsit5(),saveat = patient.local_t))
    ll = 0.0
    try
        ll += sum(logpdf.(Normal.(prediction[2, :], sigma_glucose), patient.glucose))
        ll += sum(logpdf.(Normal.(prediction[3, :], sigma_insulin), patient.insulin))
    catch
        return -Inf
    end

    return ll
end

function fit_edes(patient::NamedTuple, Dmeal = 75000.0; 
    lower_bounds = fill(-10.0, 5), upper_bounds = fill(2.0, 5), n_initial_samples = 1000, 
    n_optimizations = 20)
    bw = patient.bw
    Gb = patient.glucose[1]
    Ib = patient.insulin[1]

    u0 = edes_initial_conditions(Gb, Ib)

    model = ODEProblem(edesode, u0, (patient.local_t[1], patient.local_t[end]), edes_parameters(Dmeal, bw, Gb, Ib))

    loss_function = (p, (patient, model, Dmeal)) -> -loglikelihood(p, (patient, model, Dmeal))
    optfun = OptimizationFunction(loss_function, Optimization.AutoForwardDiff())
    # take initial samples
    initial_samples = QuasiMonteCarlo.sample(n_initial_samples, lower_bounds, upper_bounds, LatinHypercubeSample())
    #println(size(initial_samples))
    initial_losses = [loss_function(sample, (patient, model, Dmeal)) for sample in eachcol(initial_samples)]

    # take the best n_optimizations samples and optimize from there
    best_indices = partialsortperm(initial_losses, 1:n_optimizations)

    optim_results = []
    for i in best_indices
        sample = initial_samples[:,i]
        opt = Optimization.OptimizationProblem(optfun, sample, (patient, model, Dmeal), lb=lower_bounds, ub=upper_bounds)
        result = solve(opt, Optim.LBFGS(linesearch=BackTracking()))
        push!(optim_results, result)
    end

    best_result = argmin([res.objective for res in optim_results])
    best_result = optim_results[best_result]
    return best_result
end

function predict_edes(optim_result, patient::NamedTuple, Dmeal = 75000.0; save_at = patient.local_t)
    k1, k5, k6 = exp.(optim_result.minimizer[1:3])
    bw = patient.bw
    Gb = patient.glucose[1]
    Ib = patient.insulin[1]
    u0 = edes_initial_conditions(Gb, Ib)
    p = edes_parameters(Dmeal, bw, Gb, Ib; k1 = k1, k5 = k5, k6 = k6)
    model = ODEProblem(edesode, u0, (patient.local_t[1], patient.local_t[end]), p)
    sol = solve(model, Tsit5(), saveat = save_at)
    return Array(sol)
end

# Example usage:
patient = MIR_patients[2]
result = fit_edes(patient)

prediction = predict_edes(result, patient; save_at = 0.1)

using CairoMakie, Colors

"""
    palette(c1::Color, c2::Color, n::Int; space=Lab)

Generate a color scheme with `n` colors by interpolating between `c1` and `c2`.
Interpolation is performed in the given color `space` (default = Lab).
"""
function palette(c1::Color, c2::Color, n::Int)
    return range(c1, stop=c2, length=n)
end

"""
    lighten(c::Color, amount::Float64=0.1)

Lighten color `c` by increasing its lightness by `amount` (default 0.1).
"""
function lighten(c::Color, amount::Float64=0.1)
    hsl = HSL(c)
    new_l = clamp(hsl.l + amount, 0, 1)  # keep within [0,1]
    return convert(typeof(c), HSL(hsl.h, hsl.s, new_l))
end

figure_prediction = let f = Figure()
    ax_glucose = CairoMakie.Axis(f[1, 1], xlabel = "Time (min)", ylabel = "Glucose (mmol/L)", title = "Glucose Prediction")
    ax_insulin = CairoMakie.Axis(f[2, 1], xlabel = "Time (min)", ylabel = "Insulin (μIU/mL)", title = "Insulin Prediction")

    lines!(ax_glucose, patient.local_t[1]:0.1:patient.local_t[end], prediction[2, :], color = :blue, label = "Predicted")
    scatter!(ax_glucose, patient.local_t, patient.glucose, color = :red, label = "Observed")

    lines!(ax_insulin, patient.local_t[1]:0.1:patient.local_t[end], prediction[3, :], color = :blue, label = "Predicted")
    scatter!(ax_insulin, patient.local_t, patient.insulin, color = :red, label = "Observed")

    axislegend(ax_glucose)
    #axislegend(ax_insulin)
    f
end

# trajectory example
patient_id = 2
patients_MIR = MIR_patients[[patient.patient_id == patient_id for patient in MIR_patients]]
patients_LIR = LIR_patients[[patient.patient_id == patient_id for patient in LIR_patients]]
patients_healthy = healthy_patients[[patient.patient_id == patient_id for patient in healthy_patients]]
Dmeal = 75000.0
parameter_values_MIR = [fit_edes(patient, Dmeal).u for patient in patients_MIR]
transformed_MIR = exp.(hcat(parameter_values_MIR...))

parameter_values_LIR = [fit_edes(patient, Dmeal).u for patient in patients_LIR]
transformed_LIR = exp.(hcat(parameter_values_LIR...))

parameter_values_healthy = [fit_edes(patient, Dmeal).u for patient in patients_healthy]
transformed_healthy = exp.(hcat(parameter_values_healthy...))

# plot parameter trajectory
figure_trajectory = let f = Figure(size=(850,600), fonts=(;bold = "Noto Sans SemiBold", default="Noto Sans Light"))

    lir_color = colorant"#c74300"
    mir_color = colorant"#008aa1"
    # purple for healthy
    healthy_color = colorant"#228c22"

    trajectories_grid = GridLayout(f[2,1])

    ax_k1 = CairoMakie.Axis(trajectories_grid[1, 1], xlabel = "Time (years)",ylabel = "k1")
    ax_k5 = CairoMakie.Axis(trajectories_grid[1, 2], xlabel = "Time (years)",ylabel = "k5", )
    ax_k6 = CairoMakie.Axis(trajectories_grid[1, 3], xlabel = "Time (years)", ylabel = "k6")

    days = [patient.global_t for patient in patients_MIR]

    lines!(ax_k1, days, transformed_MIR[1, :], color = mir_color, label="Muscle")
    lines!(ax_k5, days, transformed_MIR[2, :], color = mir_color)
    lines!(ax_k6, days, transformed_MIR[3, :], color = mir_color)

    lines!(ax_k1, days, transformed_LIR[1, :], color = lir_color, label="Liver")
    lines!(ax_k5, days, transformed_LIR[2, :], color = lir_color)
    lines!(ax_k6, days, transformed_LIR[3, :], color = lir_color)

    lines!(ax_k1, days, transformed_healthy[1, :], color = healthy_color, label="Healthy")
    lines!(ax_k5, days, transformed_healthy[2, :], color = healthy_color)
    lines!(ax_k6, days, transformed_healthy[3, :], color = healthy_color)

    Legend(trajectories_grid[2,1:3], ax_k1, "Insulin Resistance"; orientation = :horizontal)

    data_grid = GridLayout(f[1,1])
    ax_data_glucose = CairoMakie.Axis(data_grid[1, 1], xlabel = "Time (min)", ylabel = "Glucose (mmol/L)", title = "Glucose data", titlealign = :left)
    ax_data_insulin = CairoMakie.Axis(data_grid[1, 2], xlabel = "Time (min)", ylabel = "Insulin (μIU/mL)", title = "Insulin data", titlealign = :left)

    progression_color(c, n) = palette(lighten(c, 0.55), c, length(patients_MIR))[n]
    
    for (i, patient) in enumerate(patients_MIR)
        lines!(ax_data_glucose, patient.local_t, patient.glucose, color = progression_color(mir_color, i), label = i == 1 ? "Muscle" : "")
        scatter!(ax_data_glucose, patient.local_t, patient.glucose, color = progression_color(mir_color, i), label = i == 1 ? "Muscle" : "", markersize=6)
        lines!(ax_data_insulin, patient.local_t, patient.insulin, color = progression_color(mir_color, i), label = i == 1 ? "Muscle" : "")
        scatter!(ax_data_insulin, patient.local_t, patient.insulin, color = progression_color(mir_color, i), label = i == 1 ? "Muscle" : "",markersize=6)
    end

    for (i, patient) in enumerate(patients_LIR)
        lines!(ax_data_glucose, patient.local_t, patient.glucose, color = progression_color(lir_color, i), label = i == 1 ? "Liver" : "")
        scatter!(ax_data_glucose, patient.local_t, patient.glucose, color = progression_color(lir_color, i), label = i == 1 ? "Liver" : "", markersize=6)
        lines!(ax_data_insulin, patient.local_t, patient.insulin, color = progression_color(lir_color, i), label = i == 1 ? "Liver" : "")
        scatter!(ax_data_insulin, patient.local_t, patient.insulin, color = progression_color(lir_color, i), label = i == 1 ? "Liver" : "", markersize=6)
    end

    for (i, patient) in enumerate(patients_healthy)
        lines!(ax_data_glucose, patient.local_t, patient.glucose, color = progression_color(healthy_color, i), label = i == 1 ? "Healthy" : "")
        scatter!(ax_data_glucose, patient.local_t, patient.glucose, color = progression_color(healthy_color, i), label = i == 1 ? "Healthy" : "", markersize=6)
        lines!(ax_data_insulin, patient.local_t, patient.insulin, color = progression_color(healthy_color, i), label = i == 1 ? "Healthy" : "")
        scatter!(ax_data_insulin, patient.local_t, patient.insulin, color = progression_color(healthy_color, i), label = i == 1 ? "Healthy" : "", markersize=6)
    end

    # colorbar
    lightness_colormap = palette(lighten(colorant"gray", 0.45), colorant"gray", 100)
    cb = Colorbar(data_grid[1,3], colormap = lightness_colormap, label = "Progression (years)", colorrange=(0,30), vertical=true)
    #cb.ticksformat = x -> string(Int(x))
    

    f
end

save("figures/example_parameter_trajectory.png", figure_trajectory)

# get all trajectories
patient_ids = unique([patient.patient_id for patient in MIR_patients])

MIR_parameters = zeros(5, length(patient_ids), length([patient for patient in MIR_patients if patient.patient_id == 1]))
LIR_parameters = copy(MIR_parameters)
healthy_parameters = copy(MIR_parameters)


@showprogress for patient_id in patient_ids
    patients_MIR = MIR_patients[[patient.patient_id == patient_id for patient in MIR_patients]]
    patients_LIR = LIR_patients[[patient.patient_id == patient_id for patient in LIR_patients]]
    patients_healthy = healthy_patients[[patient.patient_id == patient_id for patient in healthy_patients]]
    Dmeal = 75000.0
    parameter_values_MIR = [fit_edes(patient, Dmeal).u for patient in patients_MIR]
    transformed_MIR = exp.(hcat(parameter_values_MIR...))

    parameter_values_LIR = [fit_edes(patient, Dmeal).u for patient in patients_LIR]
    transformed_LIR = exp.(hcat(parameter_values_LIR...))

    parameter_values_healthy = [fit_edes(patient, Dmeal).u for patient in patients_healthy]
    transformed_healthy = exp.(hcat(parameter_values_healthy...))

    # save to JLD2
    MIR_parameters[:, patient_id, :] .= transformed_MIR
    LIR_parameters[:, patient_id, :] .= transformed_LIR
    healthy_parameters[:, patient_id, :] .= transformed_healthy

end

jldsave("data/parameter_trajectories.jld2", MIR_parameters=MIR_parameters, LIR_parameters=LIR_parameters, healthy_parameters=healthy_parameters)

# fit unseen parameters
MIR_unseen, LIR_unseen, healthy_unseen = jldopen("data/mir_lir_simulations_unseen.jld2", "r") do file
    (file["MIR_patients"], file["LIR_patients"], file["healthy_patients"])
end


estimated_parameters = NamedTuple[]
for (patients, category) in zip([MIR_unseen, LIR_unseen, healthy_unseen], ["MIR", "LIR", "Healthy"])
    @showprogress for patient in patients
        result = fit_edes(patient)
        push!(estimated_parameters, (patient_id = patient.patient_id, category = category, global_t = patient.global_t, parameters = exp.(result.u)))
    end
end

estimated_parameters[2]

jldsave("data/estimated_parameters_unseen.jld2", estimated_parameters=estimated_parameters)

MIR_parameters, LIR_parameters, healthy_parameters = jldopen("data/parameter_trajectories.jld2", "r") do file
    file["MIR_parameters"], file["LIR_parameters"], file["healthy_parameters"]
end
figure_parameter_trajectories = let f = Figure(size=(350,600), fonts=(;bold = "Noto Sans SemiBold", default="Noto Sans Light"))


    lir_color = colorant"#c74300"
    mir_color = colorant"#008aa1"
    healthy_color = colorant"#5c8a73"

    #lir_color = mir_color = healthy_color = colorant"#808080"

    ax_k1 = Axis(f[1, 1], xlabel = "Time (days)", ylabel = "k1", title = "Parameter trajectories", titlealign = :left, subtitle = "k₁ trajectory", titlesize=16)
    ax_k5 = Axis(f[2, 1], xlabel = "Time (days)", ylabel = "k5", titlealign = :left, subtitle = "k₅ trajectory")
    ax_k6 = Axis(f[3, 1], xlabel = "Time (days)", ylabel = "k6", titlealign = :left, subtitle = "k₆ trajectory")
    limits!(ax_k1, 0, 30, 0, 0.05)

    mean_k1_MIR = mean(MIR_parameters[1, :, :], dims=1)
    mean_k5_MIR = mean(MIR_parameters[2, :, :], dims=1)
    mean_k6_MIR = mean(MIR_parameters[3, :, :], dims=1)
    println(size(mean_k1_MIR))
    mean_k1_LIR = mean(LIR_parameters[1, :, :], dims=1)
    mean_k5_LIR = mean(LIR_parameters[2, :, :], dims=1)
    mean_k6_LIR = mean(LIR_parameters[3, :, :], dims=1)

    mean_k1_healthy = mean(healthy_parameters[1, :, :], dims=1)
    mean_k5_healthy = mean(healthy_parameters[2, :, :], dims=1)
    mean_k6_healthy = mean(healthy_parameters[3, :, :], dims=1)

    days = [patient.global_t for patient in MIR_patients if patient.patient_id == 1]


    for i in 1:length(patient_ids)
        days = [patient.global_t for patient in MIR_patients if patient.patient_id == i]
        lines!(ax_k1, days, MIR_parameters[1, i, :], color = mir_color, alpha=0.08)
        lines!(ax_k5, days, MIR_parameters[2, i, :], color = mir_color, alpha=0.08)
        lines!(ax_k6, days, MIR_parameters[3, i, :], color = mir_color, alpha=0.08)

        lines!(ax_k1, days, LIR_parameters[1, i, :], color = lir_color, alpha=0.08)
        lines!(ax_k5, days, LIR_parameters[2, i, :], color = lir_color, alpha=0.08)
        lines!(ax_k6, days, LIR_parameters[3, i, :], color = lir_color, alpha=0.08)

        lines!(ax_k1, days, healthy_parameters[1, i, :], color = healthy_color, alpha=0.08)
        lines!(ax_k5, days, healthy_parameters[2, i, :], color = healthy_color, alpha=0.08)
        lines!(ax_k6, days, healthy_parameters[3, i, :], color = healthy_color, alpha=0.08)
    end

    lines!(ax_k1, days, mean_k1_MIR[:], color = mir_color, linewidth=3, label="Muscle")
    lines!(ax_k5, days, mean_k5_MIR[:], color = mir_color, linewidth=3)
    lines!(ax_k6, days, mean_k6_MIR[:], color = mir_color, linewidth=3)

    lines!(ax_k1, days, mean_k1_LIR[:], color = lir_color, linewidth=3, label="Liver")
    lines!(ax_k5, days, mean_k5_LIR[:], color = lir_color, linewidth=3)
    lines!(ax_k6, days, mean_k6_LIR[:], color = lir_color, linewidth=3)

    lines!(ax_k1, days, mean_k1_healthy[:], color = healthy_color, linewidth=3, label="Healthy")
    lines!(ax_k5, days, mean_k5_healthy[:], color = healthy_color, linewidth=3)
    lines!(ax_k6, days, mean_k6_healthy[:], color = healthy_color, linewidth=3)

    #Legend(f[4,1], ax_k1, "Insulin Resistance"; orientation = :horizontal)

    f
end

#save("figures/all_parameter_trajectories_improved_small.png", figure_parameter_trajectories)