# fixed constants
const k5_min = 1e-6
const k5_max = 0.05
const k5_a = 1.5
const k5_b = 1.0

using OrdinaryDiffEq, CairoMakie



# define a subject
mutable struct Subject
    id::Int
    G0::Float64
    I0::Float64
    Sm::Float64
    Sl::Float64
    LS::Float64
    cms::Float64
    cls::Float64
    gamma::Float64
    cw::Float64
    cwl::Float64
    sigma_ls::Float64
    sigma_sm::Float64
    sigma_sl::Float64
end

# compute values based on k1, k5, and k6
function Subject(id, fasting_glucose, fasting_insulin, k5, cms, cls, gamma, cw, cwl)
    G0 = fasting_glucose
    I0 = fasting_insulin
    Sl = 1.0
    Δk5 = k5_max - k5_min
    Sm = -log(Δk5 / (k5 - k5_min) - 1) / k5_a
    LS = 0.0
    sigma_ls = sigma_sm = sigma_sl = 0.01
    return Subject(id, G0, I0, Sm, Sl, LS, cms, cls, gamma, cw, cwl, sigma_ls, sigma_sm, sigma_sl)
end

function subject_ode!(du, u, p, t, subject)

    # unpack state variables
    G, I, β, k6, BW, Sm, Sl = u

    LS =  -1e-7 # example lifestyle score function
    # unpack parameters
    k5_I, kb, kLS = p
    k2 = subject.I0*(1+β+(2-Sm-Sl)) / subject.G0  # insulin sensitivity parameter

    # unpack subject-specific parameters
    cms = subject.cms
    cls = subject.cls
    cw = subject.cw
    cwl = subject.cwl

    # compute derivatives
    du[1] = I*(1+β+(2-Sm-Sl)) - k2 * G + kLS * LS * (G > 4.0) # dG/dt
    du[2] = k5_I * (k2 * G - I*(1+β+(2-Sm-Sl)))  # dI/dt
    du[3] =  kb * LS * (1 - β) * 1 / (1 + exp(- 1e6 * (G - subject.G0))) # dβ/dt
    du[4] =  (subject.gamma * LS - β ) * k6 * 1 / (1 + exp(- 1e6 * (G - subject.G0)))   # dk6/dt
    du[5] = cw * LS * G - cwl * (BW - 70)^3  # dBW/dt
    du[6] = - cms * LS * Sm  # dSm/dt
    du[7] = - cls * LS * Sl  # dSl/dt

    return nothing
end

function simulate_subject(subject::Subject, tspan, BW0, k60, p)
    u0 = [subject.G0, subject.I0, 0.0, k60, BW0, subject.Sm, subject.Sl]
    prob = ODEProblem((du, u, p, t) -> subject_ode!(du, u, p, t, subject), u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=1.0)
    return sol
end

sub = Subject(1, 5.0, 12.0, 0.04, 0.003, 0.001, 4e-2, 8e-3, 4e-8)
tspan = (0.0, 365.0 * 80)
BW0 = 70.0
k60 = 3.0

p = [0.08, 5e-6, 0.1]

sol = simulate_subject(sub, tspan, BW0, k60, p)

function compute_parameters(sol, sub, k1_0, k1_a)

    Sm = sol[6, :]
    Sl = sol[7, :]

    k5 = k5_min .+ (k5_max - k5_min) ./ (1 .+ exp.(- (k5_a .* Sm .+ k5_b .* Sl)))

    k1 = k1_0 .- k1_a .* (Sl .- sub.Sl) .* (Sl .< sub.Sl)

    return k1, k5
end

k1, k5 = compute_parameters(Array(sol), sub, 0.0105, 0.005)
k6 = sol[4, :]
sol[4,:]
Gb = sol[1, :]
Ib = sol[2, :]
bw = sol[5, :]
figure_params = let f = Figure(size=(1200, 800))

    ax_k1 = Axis(f[1, 1], xlabel="Time (years)", ylabel="k1", title="k1 over time")
    lines!(ax_k1, sol.t ./ 365, k1, color=:blue)

    ax_k5 = Axis(f[1, 2], xlabel="Time (years)", ylabel="k5", title="k5 over time")
    lines!(ax_k5, sol.t ./ 365, k5, color=:red)

    ax_beta = Axis(f[2, 0], xlabel="Time (years)", ylabel="Beta-cell function", title="Beta-cell function over time")
    lines!(ax_beta, sol.t ./ 365, sol[3, :], color=:black)

    ax_k6 = Axis(f[2, 1], xlabel="Time (years)", ylabel="k6", title="k6 over time")
    lines!(ax_k6, sol.t ./ 365, k6, color=:green)

    ax_Gb = Axis(f[2, 2], xlabel="Time (years)", ylabel="Glucose (mmol/L)", title="Fasting Glucose over time")
    lines!(ax_Gb, sol.t ./ 365, Gb, color=:orange)

    ax_Ib = Axis(f[3, 1], xlabel="Time (years)", ylabel="Insulin (pmol/L)", title="Fasting Insulin over time")
    lines!(ax_Ib, sol.t ./ 365, Ib, color=:purple)

    ax_bw = Axis(f[3, 2], xlabel="Time (years)", ylabel="Body Weight (kg)", title="Body Weight over time")
    lines!(ax_bw, sol.t ./ 365, bw, color=:brown)

    ax_SM = Axis(f[3, 0], xlabel="Time (years)", ylabel="Muscle Insulin Sensitivity", title="Muscle Insulin Sensitivity over time")
    lines!(ax_SM, sol.t ./ 365, sol[6, :], color=:cyan)

    ax_SL = Axis(f[4, 0], xlabel="Time (years)", ylabel="Liver Insulin Sensitivity", title="Liver Insulin Sensitivity over time")
    lines!(ax_SL, sol.t ./ 365, sol[7, :], color=:magenta)

    return f
end
