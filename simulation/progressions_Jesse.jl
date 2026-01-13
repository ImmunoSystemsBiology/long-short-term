include("simulate_edes.jl")

using Colors, Random, JLD2

"""
    lighten(c::Color, amount::Float64=0.1)

Lighten color `c` by increasing its lightness by `amount` (default 0.1).
"""
function lighten(c::Color, amount::Float64=0.1)
    hsl = HSL(c)
    new_l = clamp(hsl.l + amount, 0, 1)  # keep within [0,1]
    return convert(typeof(c), HSL(hsl.h, hsl.s, new_l))
end

"""
    darken(c::Color, amount::Float64=0.1)

Darken color `c` by decreasing its lightness by `amount` (default 0.1).
"""
function darken(c::Color, amount::Float64=0.1)
    hsl = HSL(c)
    new_l = clamp(hsl.l - amount, 0, 1)  # keep within [0,1]
    return convert(typeof(c), HSL(hsl.h, hsl.s, new_l))
end

"""
    palette(c1::Color, c2::Color, n::Int; space=Lab)

Generate a color scheme with `n` colors by interpolating between `c1` and `c2`.
Interpolation is performed in the given color `space` (default = Lab).
"""
function palette(c1::Color, c2::Color, n::Int)
    return range(c1, stop=c2, length=n)
end


function MIR_progression(
    bw::Vector{Float64}, Gb::Float64, Ib::Float64, Dmeal::Float64, local_timepoints::Vector{Float64}, 
    global_timepoints::Vector{Float64}, 
    λ_k5::Real, λ_k9::Real,
    Gb_rate::Real, Ib_rate::Real; 
    dynamic_noise::Real = 0.0, 
    static_noise::Real = 0.0,
    k3 = 6.07e-3,
    k4 = 2.35e-3,
    k5 = 0.0424,
    G_liv_b = 0.043,
    k9 = 3.83e-2,
    noise_distribution::Distribution = Normal(0.0, 0.05),
    rng::AbstractRNG = Random.default_rng())

    # progression rates
    λ_k5 = λ_k5 * (1.0 + dynamic_noise * randn(rng))
    λ_k9 = λ_k9 * (1.0 + dynamic_noise * randn(rng))
    Gb_rate = Gb_rate * (1.0 + dynamic_noise * randn(rng))
    Ib_rate = Ib_rate * (1.0 + dynamic_noise * randn(rng))

    k3s = max.(0.0, k3 .* (1.0 .+ static_noise .* randn(rng, length(global_timepoints))))
    k4s = max.(0.0, k4 .* (1.0 .+ static_noise .* randn(rng, length(global_timepoints))))
    G_liv_bs = max.(0.0, G_liv_b .* (1.0 .+ static_noise .* randn(rng, length(global_timepoints))))

    k5s = max.(0.0, k5 .* exp.(-λ_k5 .* global_timepoints))
    k9s = max.(0.0, k9 .* exp.(-λ_k9 .* global_timepoints))
    Gb_s = max.(0.0, Gb .+ Gb_rate .* global_timepoints)
    Ib_s = max.(0.0, Ib .+ Ib_rate .* global_timepoints)

    param_values = [(;k3 = k3s[i], k4 = k4s[i], k5 = k5s[i], EGPb = G_liv_bs[i], k9 = k9s[i]) for i in eachindex(global_timepoints)]
    individuals = [(;bw = bw[i], Gb = Gb_s[i], Ib = Ib_s[i]) for i in eachindex(global_timepoints)]

    # Simulate the trajectory for the given individual
    return simulate_trajectory(param_values, individuals, Dmeal, local_timepoints; noise_distribution=noise_distribution, rng=rng)
end

function LIR_progression(
    bw::Vector{Float64}, Gb::Float64, Ib::Float64, Dmeal::Float64, local_timepoints::Vector{Float64}, 
    global_timepoints::Vector{Float64}, λ_k3::Real, λ_k4::Real, λ_k5::Real, λ_k1::Real, λ_k6::Real, G_liv_b_rate::Real, Gb_rate::Real, Ib_rate::Real; 
    dynamic_noise::Real = 0.0,
    k3 = 6.07e-3,
    k4 = 2.35e-3,
    k5 = 0.0424,
    k1 = 0.0105,
    k6 = 2.2975,
    G_liv_b = 0.043,
    noise_distribution::Distribution = Normal(0.0, 0.05),
    rng::AbstractRNG = Random.default_rng())

    # progression rates
    λ_k3 = λ_k3 * (1.0 + dynamic_noise * randn(rng))
    λ_k4 = λ_k4 * (1.0 + dynamic_noise * randn(rng))
    λ_k5 = λ_k5 * (1.0 + dynamic_noise * randn(rng))
    k1_rate = λ_k1 * (1.0 + dynamic_noise * randn(rng))
    k6_rate = λ_k6 * (1.0 + dynamic_noise * randn(rng))
    G_liv_b_rate = G_liv_b_rate * (1.0 + dynamic_noise * randn(rng))
    Gb_rate = Gb_rate * (1.0 + dynamic_noise * randn(rng))
    Ib_rate = Ib_rate * (1.0 + dynamic_noise * randn(rng))

    k3s = max.(0.0, k3 .* exp.(-λ_k3 .* global_timepoints))
    k4s = max.(0.0, k4 .* exp.(-λ_k4 .* global_timepoints))
    k5s = max.(0.0, k5 .* exp.(-λ_k5 .* global_timepoints))
    k6s = max.(0.0, k6 .* (1.0 .+ k6_rate .* global_timepoints))
    G_liv_bs = max.(0.0, G_liv_b .* (1.0 .+ G_liv_b_rate .* global_timepoints))

    Gb_s = max.(0.0, Gb .+ Gb_rate .* global_timepoints)
    k1s = max.(0.0, k1 .+ k1_rate .* global_timepoints)
    Ib_s = max.(0.0, Ib .+ Ib_rate .* global_timepoints)

    param_values = [(;k3 = k3s[i], k4 = k4s[i], k5 = k5s[i], EGPb = G_liv_bs[i], k1 = k1s[i], k8 = 2.0, k6 = k6s[i]) for i in eachindex(global_timepoints)]
    individuals = [(;bw = bw[i], Gb = Gb_s[i], Ib = Ib_s[i]) for i in eachindex(global_timepoints)]
    
    # Simulate the trajectory for the given individual
    return simulate_trajectory(param_values, individuals, Dmeal, local_timepoints; noise_distribution=noise_distribution, rng=rng)
end


local_timepoints = [0.0, 30.0, 60.0, 90.0, 120.0]
global_timepoints = [0.0, 6.0, 12.0, 18.0, 24.0, 30.0]

# simulate MIR progressions
n_LIR_progressions = n_MIR_progressions = n_healthy_progressions = 20
filename = "data/mir_lir_simulations_final.jld2"
Random.seed!(27052023)
Dmeal = 75000.0

MIR_patients = NamedTuple[]

for patient_id in 1:n_MIR_progressions

    weight_offset = cumsum(rand(Normal(2.0, 2.0), length(global_timepoints)))
    weight_std = rand(InverseGamma(3.0, 4.0))
    bw = [rand(Normal(70.0 + k, weight_std)) for k in weight_offset] # body weight in kg
    Gb = rand(Normal(5.5, 0.2)) # fasting glucose in mmol/L
    Ib = rand(Normal(15.0, 3.0)) # fasting insulin in μIU/mL

    λ_k5 = max(0.0, rand(Normal(0.06, 0.06/2)))
    λ_k9 = max(0.0, rand(Normal(0.001, 0.001/10)))
    Gb_rate = max(0.0, rand(Normal(0.05, 0.05/10)))
    Ib_rate = max(0.0, rand(Normal(0.05, 0.05/10)))
    static_noise = 0.05
    dynamic_noise = 0.05

    result_MIR = MIR_progression(bw, Gb, Ib, Dmeal, local_timepoints, global_timepoints, λ_k5, λ_k9, Gb_rate, Ib_rate; static_noise=static_noise, dynamic_noise=dynamic_noise)

    for (j, gt) in enumerate(global_timepoints)
        push!(MIR_patients, (patient_id=patient_id, global_t=gt, bw=bw[j], local_t=local_timepoints, glucose=result_MIR[j, :, 1], insulin=result_MIR[j, :, 2]))
    end
end

LIR_patients = NamedTuple[]
for patient_id in 1:n_LIR_progressions

    weight_offset = cumsum(rand(Normal(2.0, 2.0), length(global_timepoints)))
    weight_std = rand(InverseGamma(3.0, 4.0))
    bw = [rand(Normal(70.0 + k, weight_std)) for k in weight_offset] # body weight in kg
    Gb = rand(Normal(5.5, 0.2)) # fasting glucose in mmol/L
    Ib = rand(Normal(15.0, 3.0)) # fasting insulin in μIU/mL

    λ_k3 = max(0.0, rand(Normal(0.9, 0.9/10)))
    λ_k4 = max(0.0, rand(Normal(0.9, 0.9/10)))
    λ_k5 = max(0.0, rand(Normal(0.01, 0.01)))
    k1_rate = max(0.0, rand(Normal(3e-4, 3e-4/10)))
    k6_rate = max(0.0, rand(Normal(4e-3, 4e-3/10)))
    G_liv_b_rate = max(0.0, rand(Normal(0.08, 0.08/10)))
    Gb_rate = max(0.0, rand(Normal(0.05, 0.05/10)))
    Ib_rate = max(0.0, rand(Normal(0.05, 0.05/10)))
    dynamic_noise = 0.05

    result_LIR = LIR_progression(bw, Gb, Ib, Dmeal, local_timepoints, global_timepoints, λ_k3, λ_k4, λ_k5, k1_rate, k6_rate, G_liv_b_rate, Gb_rate, Ib_rate; dynamic_noise=dynamic_noise)

    for (j, gt) in enumerate(global_timepoints)
        push!(LIR_patients, (patient_id=patient_id, global_t=gt, bw=bw[j], local_t=local_timepoints, glucose=result_LIR[j, :, 1], insulin=result_LIR[j, :, 2]))
    end
end

healthy_patients = NamedTuple[]
for patient_id in 1:n_healthy_progressions

    weight_offset = cumsum(rand(Normal(0.0, 2.0), length(global_timepoints)))
    weight_std = rand(InverseGamma(3.0, 4.0))
    bw = [rand(Normal(70.0 + k, weight_std)) for k in weight_offset] # body weight in kg
    Gb = rand(Normal(5.5, 0.2)) # fasting glucose in mmol/L
    Ib = rand(Normal(15.0, 3.0)) # fasting insulin in μIU/mL

    λ_k5 = max(0.0, rand(Normal(0.005, 0.005)))
    λ_k9 = 0.0
    Gb_rate = max(0.0, rand(Normal(0.004, 0.004/10)))
    Ib_rate = max(0.0, rand(Normal(0.04, 0.04/10)))
    static_noise = 0.05
    dynamic_noise = 0.05

    result_healthy = MIR_progression(bw, Gb, Ib, Dmeal, local_timepoints, global_timepoints, λ_k5, λ_k9, Gb_rate, Ib_rate; static_noise=static_noise, dynamic_noise=dynamic_noise)

    for (j, gt) in enumerate(global_timepoints)
        push!(healthy_patients, (patient_id=patient_id, global_t=gt, bw=bw[j], local_t=local_timepoints, glucose=result_healthy[j, :, 1], insulin=result_healthy[j, :, 2]))
    end
end

# save to JLD2
jldsave(filename, MIR_patients=MIR_patients, LIR_patients=LIR_patients, healthy_patients=healthy_patients)

using CairoMakie

glucose_palette(n) = palette(colorant"#FFC273", lighten(colorant"#15798F", 0.05), n)
insulin_palette(n) = glucose_palette(n) #palette(colorant"#008aa1", lighten(colorant"#008aa1", 0.55), n)

figure_simulation = let f = Figure(size=(600,400))
    axs_g = []
    axs_i = []

    insulin_c = insulin_palette(length(global_timepoints))
    glucose_c = glucose_palette(length(global_timepoints))


    for i in 1:3
        ax_glucose = Axis(f[1, i], xlabel = "Time (min)", ylabel = "Glucose (mmol/L)", title = i == 1 ? "Muscle Insulin Resistance" : i == 2 ? "Liver Insulin Resistance" : "Healthy")
        ax_insulin = Axis(f[2, i], xlabel = "Time (min)", ylabel = "Insulin (μIU/mL)")
        push!(axs_g, ax_glucose)
        push!(axs_i, ax_insulin)
    end

    for pid in 1:n_MIR_progressions
        MIR_patient = MIR_patients[[patient.patient_id == pid for patient in MIR_patients]]
        LIR_patient = LIR_patients[[patient.patient_id == pid for patient in LIR_patients]]
        healthy_patient = healthy_patients[[patient.patient_id == pid for patient in healthy_patients]]
        patients_ML = [MIR_patient, LIR_patient, healthy_patient]
        for i in 1:3

            ax_glucose = axs_g[i]
            ax_insulin = axs_i[i]

            result = patients_ML[i]
            for (j, pat) in enumerate(result)
                timepoints = local_timepoints
                glucose_values = pat.glucose
                insulin_values = pat.insulin

                lines!(ax_glucose, timepoints, glucose_values, color = (glucose_c[end-j+1], 0.5), linewidth=1, label = "Day $(global_timepoints[j])")
                lines!(ax_insulin, timepoints, insulin_values, color = (insulin_c[end-j+1], 0.5), linewidth=1, label = "Day $(global_timepoints[j])")
            end
        end

    end
    linkyaxes!(axs_g...)
    linkyaxes!(axs_i...)

    f
end