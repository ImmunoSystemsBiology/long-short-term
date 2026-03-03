# figures for neural network reconstruction
using CairoMakie, JLD2, UMAP, Random

using Lux
using ComponentArrays: ComponentArray
using OrdinaryDiffEq
using StableRNGs
rng = StableRNG(42)

MIR_parameters, LIR_parameters, healthy_parameters = jldopen("data/parameter_trajectories.jld2", "r") do file
    file["MIR_parameters"], file["LIR_parameters"], file["healthy_parameters"]
end

timepoints = [0.0, 6.0, 12.0, 18.0, 24.0, 30.0]

total_parameters = cat(MIR_parameters, LIR_parameters, healthy_parameters; dims=2)
trajectory_parameters = total_parameters[1:3,:,:]

# neural ODE

# load the conditional parameters from JLD2 file
data, net = jldopen("data/conditional_parameters_node.jld2", "r") do file
    file["conditionals"], file["network"]
end

labels = vcat(fill("MIR", 25), fill("LIR", 25), fill("Healthy", 25))

# perform UMAP dimensionality reduction
figure_umap_node = let f = Figure(size=(250,280))
    Random.seed!(52723)  # for reproducibility
    umap_result = UMAP.fit(data', 2;n_neighbors=5, min_dist=1.)
    ax = CairoMakie.Axis(f[1, 1], 
    xlabel="UMAP 1", ylabel="UMAP 2",
    xgridvisible=false, ygridvisible=false,
    xticksvisible=false, yticksvisible=false,
    xticks=([], []), yticks=([], []),
    leftspinevisible=true, bottomspinevisible=true,
    rightspinevisible=false, topspinevisible=false,
    xlabelfont=:bold, ylabelfont=:bold,
    spinewidth=1.,aspect=1,
    xlabelsize=9, ylabelsize=9,)

    lir_color = colorant"#c74300"
    mir_color = colorant"#008aa1"
    healthy_color = colorant"#5c8a73"

    for i in 1:size(data, 1)
        if i <= 25
            color = mir_color
            label = "MIR"
            marker = :circle
        elseif i <= 50
            color = lir_color
            label = "LIR"
            marker = :utriangle
        else
            color = healthy_color
            label = "Healthy"
            marker = :diamond
        end
        scatter!(ax, [umap_result.embedding[i][1]], [umap_result.embedding[i][2]], 
        color=(color,1), label=label, markersize=14, strokecolor=:black, strokewidth=0.5, marker=marker)
    end

    Legend(f[2,1], ax; orientation=:horizontal, merge=true,
        labelsize=12, bgcolor=:transparent, framevisible=false)

    #axislegend(ax; position=:lt, merge=true)
    f
end

save("figures/conditional_parameters_umap_node.svg", figure_umap_node)

figure_box = let f = Figure(size=(250,220))
    ax = CairoMakie.Axis(f[1, 1], xlabel="Group", ylabel="Conditional parameter value",
    xticks=([1, 2, 3], ["MIR", "LIR", "Healthy"]),
    xgridvisible=false, ygridvisible=false,
    xticksvisible=true, yticksvisible=true,
    leftspinevisible=true, bottomspinevisible=true,
    rightspinevisible=false, topspinevisible=false,
    xlabelfont=:bold, ylabelfont=:bold,
    spinewidth=1., aspect=1,
    xlabelsize=9, ylabelsize=9,)

    group_labels = ["MIR", "LIR", "Healthy"]
    group_positions = [1, 2, 3]

    lir_color = colorant"#c74300"
    mir_color = colorant"#008aa1"
    healthy_color = colorant"#5c8a73"

    boxplot!(ax, fill(1, size(MIR_parameters, 2)), data[1:25,1], color=mir_color, label="MIR", markersize=8)
    boxplot!(ax, fill(2, size(LIR_parameters, 2)), data[26:50,1], color=lir_color, label="LIR", markersize=8)
    boxplot!(ax, fill(3, size(healthy_parameters, 2)), data[51:75,1], color=healthy_color, label="Healthy", markersize=8)
    
    f
end

save("figures/conditional_parameters_box_node.svg", figure_box)

# define model
chain = Chain(
    Dense(4, 6, tanh),
    Dense(6, 6, tanh),
   # Dense(8, 8, tanh),
    Dense(6, 3)
)
_, st = Lux.setup(rng, chain)

function reconstruct(pnet)
    ComponentArray(
        layer_1 = (
            weight = reshape(pnet[1:24], (6, 4)),
            bias = pnet[25:30],
        ),
        layer_2 = (
            weight = reshape(pnet[31:66], (6, 6)),
            bias = pnet[67:72],
        ),
        layer_3 = (
            weight = reshape(pnet[73:90], (3, 6)),
            bias = pnet[91:93],
        ),
    )
end

function neural_ode!(du, u, p, t, st)
    du .= chain([u; p.conditional], reconstruct(p.network), st)[1]
end

figure_reconstruction_flat = let f = Figure(size=(900, 300))
    ax1 = CairoMakie.Axis(f[1, 1], xlabel="Time (min)", ylabel="k1")
    ax2 = CairoMakie.Axis(f[1, 2], xlabel="Time (min)", ylabel="k5")
    ax3 = CairoMakie.Axis(f[1, 3], xlabel="Time (min)", ylabel="k6")

    group_labels = ["MIR", "LIR", "Healthy"]
    group_positions = [1, 2, 3]

    lir_color = colorant"#c74300"
    mir_color = colorant"#008aa1"
    healthy_color = colorant"#5c8a73"

    _, st = Lux.setup(rng, chain)

    timepoints_fine = range(0, stop=30, length=100)
    selected_result = 1
    conditional_parameters = data[:,selected_result]
    network = net[selected_result]
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

        params_indiv = ComponentArray(
            network = network,
            conditional = condition
        )
        prob = ODEProblem((du, u, p, t) -> neural_ode!(du, u, p, t, st), trajectory_parameters[:, individual, 1], (timepoints_fine[1], timepoints_fine[end]), params_indiv)
        sol = solve(prob, Tsit5(), saveat=timepoints_fine)
        predictions = Array(sol)
    # println(predictions)
        parameter_1 = predictions[1, :]
        parameter_2 = predictions[2, :]
        parameter_3 = predictions[3, :]

        lines!(ax1, timepoints_fine, parameter_1, color=(color, 0.2), label=label)
        lines!(ax2, timepoints_fine, parameter_2, color=(color, 0.2))
        lines!(ax3, timepoints_fine, parameter_3, color=(color, 0.2))

        scatter!(ax1, timepoints, trajectory_parameters[1, i, :], color=color, markersize=8)
        scatter!(ax2, timepoints, trajectory_parameters[2, i, :], color=color, markersize=8)
        scatter!(ax3, timepoints, trajectory_parameters[3, i, :], color=color, markersize=8)

    end
    
    Legend(f[2,1:3], ax1; orientation=:horizontal, merge=true)
    # axislegend(ax1; position=:rt, merge=true)
    # axislegend(ax2; position=:rt, merge=true)
    # axislegend(ax3; position=:rt, merge=true)
    f
end

save("figures/reconstruction_neural_ode_flat.svg", figure_reconstruction_flat)

# conditional NN

# load the conditional parameters from JLD2 file
data, net = jldopen("data/conditional_parameters_nn.jld2", "r") do file
    file["conditionals"], file["network"]
end

labels = vcat(fill("MIR", 25), fill("LIR", 25), fill("Healthy", 25))

# perform UMAP dimensionality reduction
figure_umap_nn = let f = Figure(size=(250,280))
    Random.seed!(52723)  # for reproducibility
    umap_result = UMAP.fit(data', 2;n_neighbors=9, min_dist=2.)
    ax = CairoMakie.Axis(f[1, 1], 
    xlabel="UMAP 1", ylabel="UMAP 2",
    xgridvisible=false, ygridvisible=false,
    xticksvisible=false, yticksvisible=false,
    xticks=([], []), yticks=([], []),
    leftspinevisible=true, bottomspinevisible=true,
    rightspinevisible=false, topspinevisible=false,
    xlabelfont=:bold, ylabelfont=:bold,
    spinewidth=1.,aspect=1,
    xlabelsize=9, ylabelsize=9,)

    lir_color = colorant"#c74300"
    mir_color = colorant"#008aa1"
    healthy_color = colorant"#5c8a73"

    for i in 1:size(data, 1)
        if i <= 25
            color = mir_color
            label = "MIR"
            marker = :circle
        elseif i <= 50
            color = lir_color
            label = "LIR"
            marker = :utriangle
        else
            color = healthy_color
            label = "Healthy"
            marker = :diamond
        end
        scatter!(ax, [umap_result.embedding[i][1]], [umap_result.embedding[i][2]], 
        color=(color,1), label=label, markersize=14, strokecolor=:black, strokewidth=0.5, marker=marker)
    end

    Legend(f[2,1], ax; orientation=:horizontal, merge=true,
        labelsize=12, backgroundcolor=:transparent, framevisible=false)

    #axislegend(ax; position=:lt, merge=true)
    f
end

save("figures/conditional_parameters_umap_nn.svg", figure_umap_nn)

figure_box = let f = Figure(size=(250,220))
    ax = CairoMakie.Axis(f[1, 1], xlabel="Group", ylabel="Conditional parameter value",
    xticks=([1, 2, 3], ["MIR", "LIR", "Healthy"]),
    xgridvisible=false, ygridvisible=false,
    xticksvisible=true, yticksvisible=true,
    leftspinevisible=true, bottomspinevisible=true,
    rightspinevisible=false, topspinevisible=false,
    xlabelfont=:bold, ylabelfont=:bold,
    spinewidth=1., aspect=1,
    xlabelsize=9, ylabelsize=9,)

    group_labels = ["MIR", "LIR", "Healthy"]
    group_positions = [1, 2, 3]

    lir_color = colorant"#c74300"
    mir_color = colorant"#008aa1"
    healthy_color = colorant"#5c8a73"
    selection = 18
    boxplot!(ax, fill(1, size(MIR_parameters, 2)), data[1:25,selection], color=mir_color, label="MIR", markersize=8)
    boxplot!(ax, fill(2, size(LIR_parameters, 2)), data[26:50,selection], color=lir_color, label="LIR", markersize=8)
    boxplot!(ax, fill(3, size(healthy_parameters, 2)), data[51:75,selection], color=healthy_color, label="Healthy", markersize=8)
    
    f
end

save("figures/conditional_parameters_box_nn.svg", figure_box)

import SimpleChains as sc
# define model
softplus(x) = log1p(exp(x))

# define model
chain = sc.SimpleChain(
    static(2),
    sc.TurboDense(softplus, 32),
    sc.TurboDense(softplus, 32),
    sc.TurboDense(softplus, 3)
)

figure_reconstruction = let f = Figure(size=(900, 300))
    ax1 = CairoMakie.Axis(f[1, 1], xlabel="Time (years)", ylabel="k1")
    ax2 = CairoMakie.Axis(f[1, 2], xlabel="Time (years)", ylabel="k5")
    ax3 = CairoMakie.Axis(f[1, 3], xlabel="Time (years)", ylabel="k6")

    group_labels = ["MIR", "LIR", "Healthy"]
    group_positions = [1, 2, 3]

    lir_color = colorant"#c74300"
    mir_color = colorant"#008aa1"
    healthy_color = colorant"#5c8a73"

    timepoints_fine = range(0, stop=30, length=100)
    for sol_2 in neural_network_solutions[[18]]
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
    Legend(f[2,1:3], ax1; orientation=:horizontal, merge=true)
    # axislegend(ax1; position=:rt, merge=true)
    # axislegend(ax2; position=:rt, merge=true)
    # axislegend(ax3; position=:rt, merge=true)
    f
end

save("figures/reconstruction_nn.svg", figure_reconstruction)