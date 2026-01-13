using Random
using OrdinaryDiffEq
using Distributions
using DataFrames
using CSV
using Plots

# Two-timescale simulation with asynchronous remission and an intermediate state.
# - Slow axis: discrete visits (e.g., weeks). Each subject has a latent state in {healthy, intermediate, sick}.
#   Transitions are local (prefer moving to neighbor states), subject-specific propensities -> asynchronous behavior.
# - Intermediate state represents a gradual transition: its ODE parameters are a convex mix
#   of healthy and sick parameters (mixture modelling interpretation).
# - Fast axis: after a perturbation (dose), simulate the short ODE response (minutes) using
#   the state-specific parameters active at that slow visit.

# Seed for reproducibility
Random.seed!(2025)

# Population / time settings
N = 50                      # number of subjects
Tslow = 12                  # number of slow visits per subject
slow_times = 0:(Tslow-1)

fast_tspan = (0.0, 120.0)   # fast axis in minutes (after perturbation)
fast_sampling = 0:2:120     # sample every 2 minutes for richer curves

# Base ODE parameters for extremes
# dx/dt = -k*x + b
params_extreme = Dict(
    :healthy => (k=0.06, b=0.0),
    :sick    => (k=0.01, b=0.01)
)

# Intermediate state is a convex combination of healthy & sick
# we will allow subject-specific "mixing strength" for the intermediate state to model gradualness
# mixing w in [0,1]: param_intermediate = (1-w)*params_healthy + w*params_sick

# Subject-specific progression/remission propensities (introduces asynchrony)
# For each subject, sample a progression tendency p_prog ~ Beta(2,6) (low average) and
# remission tendency p_remit ~ Beta(2,6) (low average). These govern chance to move forward/back
# at each slow step (when eligible).
prog_tendencies = rand.(Beta(2,6), N)
remit_tendencies = rand.(Beta(2,6), N)

# Also sample subject-specific intermediate mixing weight dispersion: how "close" intermediate is to sick
mixing_weights = clamp.(rand(Normal(0.5, 0.12), N), 0.05, 0.95)

# Transition function follows nearest-neighbor moves on the chain Healthy <-> Intermediate <-> Sick
# with probabilities controlled by the subject-specific tendencies.
function transition_state(current::Symbol, subj::Int)
    p_prog = prog_tendencies[subj]
    p_remit = remit_tendencies[subj]
    # small baseline persistence (stickiness)
    p_stay = 0.75

    if current == :healthy
        # can stay or progress to intermediate
        if rand() < p_stay
            return :healthy
        else
            return rand() < p_prog ? :intermediate : :healthy
        end
    elseif current == :intermediate
        # can move to healthy, stay, or progress to sick
        r = rand()
        if r < 0.10*(1 - p_remit) # small chance immediate remission reduced by low remission tendency
            return :healthy
        elseif r < 0.10*(1 - p_remit) + p_stay
            return :intermediate
        else
            return rand() < p_prog ? :sick : :intermediate
        end
    elseif current == :sick
        # can stay or remit to intermediate/healthy
        if rand() < p_stay
            return :sick
        else
            return rand() < p_remit ? :intermediate : :sick
        end
    end
end

# Analytical integrator for the 1D linear ODE
function integrate_interval(x_start, k, b, Δ)
    if Δ == 0.0
        return x_start
    end
    return x_start * exp(-k*Δ) + b*(1 - exp(-k*Δ))/k
end

# ODE (wrapper if we use DifferentialEquations solver)
function ode!(du, u, p, t)
    k, b = p
    du[1] = -k*u[1] + b
end

# Container for simulated rows
rows = Vector{NamedTuple}()

# Run simulation
for subj in 1:N
    state = :healthy
    mix_w = mixing_weights[subj]
    for (tslow_idx, tslow) in enumerate(slow_times)
        # at t=0 everyone is healthy by design
        if tslow > 0
            state = transition_state(state, subj)
        end

        # determine parameters for this subject at this slow visit
        if state == :healthy
            par = params_extreme[:healthy]
        elseif state == :sick
            par = params_extreme[:sick]
        else
            # intermediate: convex combo depending on subject mixing weight
            wh = 1.0 - mix_w
            ws = mix_w
            par = (
                k = wh*params_extreme[:healthy].k + ws*params_extreme[:sick].k,
                b = wh*params_extreme[:healthy].b + ws*params_extreme[:sick].b
            )
        end

        # simulate fast response after a dose. Initial condition models a bolus/dose.
        u0 = [1.0]
        # using analytic integration at saved fast_sampling points for speed
        fast_vals = [integrate_interval(u0[1], par.k, par.b, t) for t in fast_sampling]
        # add observation noise
        noisy = fast_vals + rand.(Normal(0,0.03), length(fast_vals))

        # store each fast-time observation as a row
        for (ft, y_true, y_obs) in zip(fast_sampling, fast_vals, noisy)
            push!(rows, (subject=subj, slow_time=tslow, state=String(state), mix_weight=mix_w, fast_time=ft, y_true=y_true, y_obs=y_obs))
        end
    end
end

# Save
df = DataFrame(rows)
CSV.write("sim_two_timescale_intermediate.csv", df)
println("Wrote sim_two_timescale_intermediate.csv with ", nrow(df), " rows")

# Quick diagnostic plots: sample subjects and mean curves per state
sample_subs = rand(1:N, min(6, N))
ps = []
for s in 1:length(sample_subs)
    subj = sample_subs[s]
    subdf = df[df.subject .== subj, :]
    p1 = plot(title="Subject $subj fast responses (visits stacked)", xlabel="minutes", ylabel="y")
    for tslow in unique(subdf.slow_time)
        rowsubset = subdf[subdf.slow_time .== tslow, :]
        plot!(p1, rowsubset.fast_time, rowsubset.y_obs, label="visit $tslow state=$(first(rowsubset.state))")
    end
    push!(ps, p1)
end
plot(ps..., layout=(3,2), size=(900,1200))

# mean profile per nominal state (pooling across subjects & visits labeled by true state)
mean_by_state = combine(groupby(df, [:state, :fast_time]), :y_obs => mean => :mean_y, :y_obs => std => :std_y)
q = plot(title="Mean response by true state", xlabel="minutes", ylabel="y")
for st in unique(mean_by_state.state)
    sub = mean_by_state[mean_by_state.state .== st, :]
    plot!(q, sub.fast_time, sub.mean_y, label=st)
end

plot(p, q)