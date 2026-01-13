using JLD2, SimpleChains, ComponentArrays, Optimization, OptimizationOptimJL, OptimizationOptimisers, Zygote, ProgressMeter,LineSearches

using UMAP

# load data
MIR_parameters, LIR_parameters, healthy_parameters = jldopen("data/mir_lir_simulations_final.jld2", "r") do file
    file["MIR_parameters"], file["LIR_parameters"], file["healthy_parameters"]
end

timepoints = [0.0, 6.0, 12.0, 18.0, 24.0, 30.0]

# define model
chain = SimpleChain(
    static(2),
    TurboDense(relu, 16),
    TurboDense(relu, 16),
    TurboDense(relu, 16),
    TurboDense(identity, 3)
)

# define loss function
function loss_nn(p, (chain, y, t))
    loss = 0.0
    conditionals = p.conditionals
    for individual in axes(y, 2)
        for (i, time) in enumerate(t)
            input = vcat(conditionals[individual], [time])
            pred = chain(input, p.network)
            loss += sum(abs2, pred - y[:, individual, i]) ./ y[1, individual, i] # relative error
        end
    end
    return loss
end

total_parameters = cat(MIR_parameters, LIR_parameters, healthy_parameters; dims=2)
trajectory_parameters = total_parameters[1:3,:,:]

neural_network_solutions = SciMLBase.OptimizationSolution[]

for i in 1:25
    println("Run $i of 25")
    initial_network = SimpleChains.init_params(chain)
    initial_conditionals = zeros(size(trajectory_parameters, 2))
    initial_params = ComponentArray((network = initial_network, conditionals = initial_conditionals))

    try
        loss_nn(initial_params, (chain, trajectory_parameters, timepoints)) # test run
    catch
        println("Error in loss function evaluation, skipping this run...")
        continue
    end
    function create_progress_callback(n_iters)
        progress = Progress(n_iters)
        function progress_callback(p, l)
            ProgressMeter.next!(progress; showvalues = [(:loss, l)])
            return false
        end
        return progress_callback
    end
    try 
        optfunc = OptimizationFunction(loss_nn, Optimization.AutoZygote())
        optprob = OptimizationProblem(optfunc, initial_params, (chain, trajectory_parameters, timepoints))
        sol = solve(optprob, ADAM(1e-2), maxiters=3000, callback=create_progress_callback(3000))

        optprob_2 = OptimizationProblem(optfunc, sol.u, (chain, trajectory_parameters, timepoints))
        sol_2 = solve(optprob_2, Optim.LBFGS(linesearch=BackTracking()), maxiters=2000, callback=create_progress_callback(2000))
        push!(neural_network_solutions, sol_2)
    catch
        println("Optimization failed, skipping this run...")
        continue
    end
end

conditionals = hcat([sol.u.conditionals for sol in neural_network_solutions]...)
umap_result = UMAP.fit(conditionals', 2; n_neighbors=25, min_dist=0.8)
umap_result.embedding

figure_umap = let f = Figure()
    ax = CairoMakie.Axis(f[1, 1], xlabel="UMAP 1", ylabel="UMAP 2", title="UMAP of conditional parameters")

    lir_color = colorant"#c74300"
    mir_color = colorant"#008aa1"
    healthy_color = colorant"#5c8a73"

    for i in 1:size(conditionals, 1)
        if i <= size(MIR_parameters, 2)
            color = mir_color
            label = "MIR"
        elseif i <= size(MIR_parameters, 2) + size(LIR_parameters, 2)
            color = lir_color
            label = "LIR"
        else
            color = healthy_color
            label = "Healthy"
        end
        scatter!(ax, [umap_result.embedding[i][1]], [umap_result.embedding[i][2]], color=color, label=label, markersize=12, strokecolor=:black, strokewidth=0.9)
    end

    axislegend(ax; position=:rt, merge=true)
    f
end

using CairoMakie



# visualize the results
conditional_parameters = sol_2.u.conditionals

figure_conditional_parameters = let f = Figure()
    ax = CairoMakie.Axis(f[1, 1], xlabel="Group", ylabel="Conditional parameter value", title="Conditional parameters")

    group_labels = ["MIR", "LIR", "Healthy"]
    group_positions = [1, 2, 3]

    lir_color = colorant"#c74300"
    mir_color = colorant"#008aa1"
    healthy_color = colorant"#5c8a73"

    boxplot!(ax, fill(1, size(MIR_parameters, 2)), conditional_parameters[1:size(MIR_parameters, 2)], color=mir_color, label="MIR", markersize=8)
    boxplot!(ax, fill(2, size(LIR_parameters, 2)), conditional_parameters[size(MIR_parameters, 2)+1:size(MIR_parameters, 2)+size(LIR_parameters, 2)], color=lir_color, label="LIR", markersize=8)
    boxplot!(ax, fill(3, size(healthy_parameters, 2)), conditional_parameters[size(MIR_parameters, 2)+size(LIR_parameters, 2)+1:end], color=healthy_color, label="Healthy", markersize=8)

    f
end

figure_reconstruction = let f = Figure()
    ax1 = CairoMakie.Axis(f[1, 1], xlabel="Time (min)", ylabel="Value", title="Reconstruction of trajectories")
    ax2 = CairoMakie.Axis(f[2, 1], xlabel="Time (min)", ylabel="Value", title="Reconstruction of trajectories")
    ax3 = CairoMakie.Axis(f[3, 1], xlabel="Time (min)", ylabel="Value", title="Reconstruction of trajectories")

    group_labels = ["MIR", "LIR", "Healthy"]
    group_positions = [1, 2, 3]

    lir_color = colorant"#c74300"
    mir_color = colorant"#008aa1"
    healthy_color = colorant"#5c8a73"

    timepoints_fine = range(0, stop=30, length=100)
    for sol_2 in neural_network_solutions
        conditional_parameters = sol_2.u.conditionals
        for (i, individual) in enumerate(1:size(trajectory_parameters, 2))
            condition = conditional_parameters[i]
            if i <= size(MIR_parameters, 2)
                color = mir_color
                label = "MIR"
            elseif i <= size(MIR_parameters, 2) + size(LIR_parameters, 2)
                color = lir_color
                label = "LIR"
            else
                color = healthy_color
                label = "Healthy"
            end

            predictions = [chain(vcat(condition, [t]), sol_2.u.network) for t in timepoints_fine]
        # println(predictions)
            parameter_1 = getindex.(predictions, 1)
            parameter_2 = getindex.(predictions, 2)
            parameter_3 = getindex.(predictions, 3)

            lines!(ax1, timepoints_fine, parameter_1, color=(color, 0.2), label=label)
            lines!(ax2, timepoints_fine, parameter_2, color=(color, 0.2))
            lines!(ax3, timepoints_fine, parameter_3, color=(color, 0.2))

            scatter!(ax1, timepoints, trajectory_parameters[1, i, :], color=color, markersize=8)
            scatter!(ax2, timepoints, trajectory_parameters[2, i, :], color=color, markersize=8)
            scatter!(ax3, timepoints, trajectory_parameters[3, i, :], color=color, markersize=8)

        end
    end

    # axislegend(ax1; position=:rt, merge=true)
    # axislegend(ax2; position=:rt, merge=true)
    # axislegend(ax3; position=:rt, merge=true)
    f
end