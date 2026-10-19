# replication of Regazzoni et al. conditional neural ODE experiment
# learn the long-scale dynamics dθ/dτ = NN(θ, c) of the pre-estimated Windkessel parameters, with a conditional c per patient
using JLD2, Lux, ComponentArrays, Optimization, OptimizationOptimJL, OptimizationOptimisers, OrdinaryDiffEq, ProgressMeter, LineSearches
using StableRNGs, StatsBase, CairoMakie, UMAP

rng = StableRNG(42)

# load the parameters (patients × long-scale timepoints × [Rp, Cp, Rd, Cd, Pd0, σ])
windkessel_est = jldopen("data/windkessel_estimation_results.jld2", "r") do file
    file["params"]
end

# load the ground-truth α of each patient from the dataset
alphas = jldopen("data/windkessel_dataset.jld2", "r") do file
    [p.alpha for p in file["dataset"][:patients]]
end

t_long = Vector{Float64}(0:50)
# the neural ODE is solved on τ / τ_max
time_scaled = t_long ./ last(t_long)

# Rd and Cd arranged as (parameter × patient × time); the neural ODE evolves their absolute values,
# as their rate of change depends on the current value
trajectory_parameters = permutedims(windkessel_est[:, :, 3:4], (3, 1, 2))
pnames = ["Rd - Rd(0)", "Cd - Cd(0)"]

# define model; inputs are (Rd, Cd, conditional)
chain = Chain(
    Dense(3, 8, Lux.softplus),
    Dense(8, 8, Lux.softplus),
    Dense(8, 2)
)
st = Lux.initialstates(rng, chain)

# all individuals are solved as a single ODE with state (parameter × patient)
neural_ode(u, p, t, chain, st) = first(chain(vcat(u, p.conditionals'), p.network, st))

function solve_node(p, u0, t, chain, st)
    prob = ODEProblem((u, p, t) -> neural_ode(u, p, t, chain, st), convert.(eltype(p), u0), (first(t), last(t)), p)
    return solve(prob, Tsit5(), saveat=t)
end

# define loss function; each individual starts from its estimated parameters at τ = 0
function loss_node(p, (chain, y, t, st))
    sol = solve_node(p, y[:, :, 1], t, chain, st)
    if !SciMLBase.successful_retcode(sol)
        return Inf
    end
    predictions = Array(sol)
    # relative squared error
    loss = sum(abs2, (predictions - y) ./ y)
    loss /= (size(y, 2) * size(y, 3))

    return loss
end

function create_progress_callback(n_iters)
    progress = Progress(n_iters)
    function progress_callback(p, l)
        ProgressMeter.next!(progress; showvalues = [(:loss, l)])
        return false
    end
    return progress_callback
end

neural_network_solutions = SciMLBase.OptimizationSolution[]

n_runs = 25
for i in 1:n_runs
    println("Run $i of $n_runs")
    initial_network = Lux.initialparameters(rng, chain)
    initial_conditionals = zeros(size(trajectory_parameters, 2))
    initial_params = ComponentArray((network = initial_network, conditionals = initial_conditionals))

    try
        optfunc = OptimizationFunction(loss_node, Optimization.AutoForwardDiff())
        optprob = OptimizationProblem(optfunc, initial_params, (chain, trajectory_parameters, time_scaled, st))
        sol = solve(optprob, Optimisers.Adam(1e-2), maxiters=1000, callback=create_progress_callback(1000))

        optprob_2 = OptimizationProblem(optfunc, sol.u, (chain, trajectory_parameters, time_scaled, st))
        sol_2 = solve(optprob_2, Optim.LBFGS(linesearch=BackTracking()), maxiters=2000, callback=create_progress_callback(2000))
        push!(neural_network_solutions, sol_2)
    catch e
        println("Optimization failed, skipping this run... ($e)")
        continue
    end
end

losses = [sol.objective for sol in neural_network_solutions]
conditionals = hcat([Array(sol.u.conditionals) for sol in neural_network_solutions]...)

# save the conditional parameters to a JLD2 file
jldsave("data/windkessel_conditional_parameters_node.jld2"; conditionals=conditionals, losses=losses, network=[Array(sol.u.network) for sol in neural_network_solutions])

# rank correlation between the conditional parameter and the ground-truth α (the sign of the conditional is arbitrary)
alpha_correlations = [corspearman(c, alphas) for c in eachcol(conditionals)]
best_run = argmin(losses)
println("Lowest loss in run $best_run: loss = $(losses[best_run]), Spearman ρ with α = $(alpha_correlations[best_run])")

alpha_range = extrema(alphas)

umap_result = UMAP.fit(conditionals', 2; n_neighbors=10, min_dist=2.5)

figure_umap = let f = Figure()
    ax = CairoMakie.Axis(f[1, 1], xlabel="UMAP 1", ylabel="UMAP 2", title="UMAP of conditional parameters")
    scatter!(ax, umap_result.embedding[1, :], umap_result.embedding[2, :], color=alphas, colormap=:viridis, colorrange=alpha_range,
        markersize=18, strokecolor=:black, strokewidth=0.9)
    Colorbar(f[1, 2], limits=alpha_range, colormap=:viridis, label="α")
    f
end

save("figures/windkessel_conditional_parameters_umap_node.png", figure_umap)
save("figures/windkessel_conditional_parameters_umap_node.eps", figure_umap)

# test the reliability of recovering α from the conditional parameters
figure_conditional_parameters = let f = Figure(size=(900, 350))
    ax1 = CairoMakie.Axis(f[1, 1], xlabel="α", ylabel="Conditional parameter value",
        title="Run $best_run (lowest loss), ρ = $(round(alpha_correlations[best_run], digits=2))")
    scatter!(ax1, alphas, conditionals[:, best_run], color=alphas, colormap=:viridis, colorrange=alpha_range,
        markersize=10, strokecolor=:black, strokewidth=0.5)

    ax2 = CairoMakie.Axis(f[1, 2], xlabel="Final loss", ylabel="|Spearman ρ| with α", title="All runs")
    scatter!(ax2, losses, abs.(alpha_correlations), color=:black, markersize=10)
    f
end

save("figures/windkessel_conditional_parameters_alpha_node.png", figure_conditional_parameters)
save("figures/windkessel_conditional_parameters_alpha_node.eps", figure_conditional_parameters)

# trajectories are shown relative to τ = 0 for comparison with mixture.jl and conditionalNN.jl
figure_reconstruction = let f = Figure(size=(1200, 300))
    n_individuals = size(trajectory_parameters, 2)
    t_fine = range(0.0, 1.0, length=200)
    u0 = trajectory_parameters[:, :, 1]
    predictions = Array(solve_node(neural_network_solutions[best_run].u, u0, t_fine, chain, st)) .- u0

    for param in 1:2
        ax = CairoMakie.Axis(f[1, param], xlabel="Time", ylabel=pnames[param])
        for i in 1:n_individuals
            lines!(ax, t_fine .* last(t_long), predictions[param, i, :], color=fill(alphas[i], length(t_fine)), colormap=:viridis, colorrange=alpha_range, alpha=0.4)
        end
        scatter!(ax, repeat(t_long, inner=n_individuals), vec(trajectory_parameters[param, :, :] .- u0[param, :]), color=repeat(alphas, length(t_long)),
            colormap=:viridis, colorrange=alpha_range, markersize=4, alpha=0.3)
    end
    Colorbar(f[1, 3], limits=alpha_range, colormap=:viridis, label="α")
    f
end

save("figures/windkessel_reconstruction_conditional_node.png", figure_reconstruction)
save("figures/windkessel_reconstruction_conditional_node.eps", figure_reconstruction)
