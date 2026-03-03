using JLD2, DataFrames, TemporalMixtureModels, CairoMakie, Statistics, Random, ProgressMeter

# load data
MIR_parameters, LIR_parameters, healthy_parameters = jldopen("data/parameter_trajectories.jld2", "r") do file
    file["MIR_parameters"], file["LIR_parameters"], file["healthy_parameters"]
end

timepoints_og = [0.0, 6.0, 12.0, 18.0, 24.0, 30.0]

# prepare data in long format
ids = Int[]
k1s = Float64[]
k5s = Float64[]
k6s = Float64[]
times = Float64[]

for (individual, time) in Iterators.product(1:size(MIR_parameters, 2), 1:length(timepoints_og))
    push!(ids, individual)
    push!(k1s, MIR_parameters[1, individual, time])
    push!(k5s, MIR_parameters[2, individual, time])
    push!(k6s, MIR_parameters[3, individual, time])
    push!(times, timepoints_og[time])
end

for (individual, time) in Iterators.product(1:size(LIR_parameters, 2), 1:length(timepoints_og))
    push!(ids, individual + size(MIR_parameters, 2))
    push!(k1s, LIR_parameters[1, individual, time])
    push!(k5s, LIR_parameters[2, individual, time])
    push!(k6s, LIR_parameters[3, individual, time])
    push!(times, timepoints_og[time])
end

for (individual, time) in Iterators.product(1:size(healthy_parameters, 2), 1:length(timepoints_og))
    push!(ids, individual + size(MIR_parameters, 2) + size(LIR_parameters, 2))
    push!(k1s, healthy_parameters[1, individual, time])
    push!(k5s, healthy_parameters[2, individual, time])
    push!(k6s, healthy_parameters[3, individual, time])
    push!(times, timepoints_og[time])
end

parameters = hcat(k1s, k5s, k6s)

# prepare for 10-fold cv
possible_n_components = [2, 3, 4, 5]
possible_orders = [1, 2, 3, 4]

function fit_and_eval(ids, t, y, fit_folds, eval_fold, n_components, order, n_repeats)
    # prepare training and evaluation data

    train_indices = in.(ids, Ref(vcat(fit_folds...)))
    test_indices = in.(ids, Ref(eval_fold))

    # define and fit model
    component_model = @component begin
        y[1] ~ PolynomialRegression(order)
        y[2] ~ PolynomialRegression(order)
        y[3] ~ PolynomialRegression(order)
    end

    model = fit_mixture(component_model, n_components, t[train_indices], y[train_indices, :], ids[train_indices]; n_repeats=n_repeats, max_iter=500, verbose=false)

    # evaluate on test data
    bic(model, t[test_indices], y[test_indices, :], ids[test_indices])
end

function cross_validate_aic(ids, t, y, folds, n_components, order; n_repeats=3)
    bics = Float64[]
    for i in eachindex(folds)
        eval_fold = folds[i]
        fit_folds = folds[1:end .!= i]
        bic_value = fit_and_eval(ids, t, y, fit_folds, eval_fold, n_components, order, n_repeats)
        push!(bics, bic_value)
    end
    return bics
end

function cross_validate(ids, t, y, n_folds, possible_n_components, possible_orders; n_repeats=3, n_refits=3)
    results = Tuple{Int, Int, Vector{Float64}}[]
    progm = Progress(length(possible_n_components) * length(possible_orders)*n_repeats)
    for _ in 1:n_repeats
        # make random folds
        ids_shuffled = shuffle(unique(ids))
        folds = [ids_shuffled[i:n_folds:end] for i in 1:n_folds]
        for n_components in possible_n_components
            for order in possible_orders
                bics = cross_validate_aic(ids, t, y, folds, n_components, order, n_repeats=n_refits)
                push!(results, (n_components, order, bics))
                #println("n_components: $n_components, order: $order, mean_aic: $(mean(aics)), std_aic: $(std(aics))")
                next!(progm)
            end
        end
    end
    return results
end

cv_results = cross_validate(ids, times, parameters, 10, possible_n_components, possible_orders, n_repeats=10, n_refits=10)

figure_cv = let f = Figure(size=(350,200))

    ax = CairoMakie.Axis(f[1,1], xlabel="Polynomial order", ylabel="BIC",
    xlabelcolor=:gray, ylabelcolor=:gray, xticklabelsize=12, yticklabelsize=8,
    topspinevisible=false, rightspinevisible=false, xgridvisible=false, ygridvisible=true, bottomspinevisible=false, xticksvisible=false, ygridwidth=0.5, yminorgridvisible=true, yminorgridwidth=0.25)
    # results per n_components
    n_components_values = unique([r[1] for r in cv_results])
    order_values = unique([r[2] for r in cv_results])
    dodges = [-0.24, -0.08, 0.08, 0.24]
    for (i, nc) in enumerate(n_components_values)
        nc_results = filter(r -> r[1] == nc, cv_results)

        nc_order = Float64[]
        nc_aics = Float64[]

        for res in nc_results
            order = res[2]
            aics = res[3]
            order_rep = fill(order, length(aics)) .+ dodges[i]

            append!(nc_order, order_rep)
            append!(nc_aics, aics)
        end
        boxplot!(ax, nc_order, nc_aics, label="$nc"; whiskerwidth=0.5, width=0.2, color=Makie.ColorSchemes.colorschemes[:Blues_4][i], markersize=5, strokewidth=0.5)
    end
    Legend(f[1,2], ax, "Components"; titlehalign=:left, orientation=:vertical, titlesize=9, gridshalign=:left, labelsize=9)
    f

end

save("figures/mixture_model_cv_bic.svg", figure_cv)

# fit final model on all data with best parameters (3 components, order 1)
best_n_components = 3
best_order = 1

component_model = @component begin
    y[1] ~ PolynomialRegression(best_order)
    y[2] ~ PolynomialRegression(best_order)
    y[3] ~ PolynomialRegression(best_order)
end
Random.seed!(27052023)
model = fit_mixture(component_model, best_n_components, times, parameters, ids; n_repeats=10, max_iter=1000, verbose=true)
# get cluster assignments
assignments = argmax(posterior_responsibilities(model, times, parameters, ids), dims=2)
assignments = [a[1][2] for a in eachrow(assignments)]

t_pred = range(0, stop=30, length=100)

bootstrap_model, _ = bootstrap(component_model, best_n_components, 100, times, parameters, ids; n_repeats=10, max_iter=1000)

# get cis
lower_bounds = fill(Inf, length(t_pred), 3, best_n_components)
upper_bounds = fill(-Inf, length(t_pred), 3, best_n_components)
bootstrap_model
for mod in bootstrap_model
    preds = predict(mod, t_pred)
    for component in 1:best_n_components
        for param in 1:3
            lower_bounds[:, param, component] = min.(lower_bounds[:, param, component], preds[component][:, param])
            upper_bounds[:, param, component] = max.(upper_bounds[:, param, component], preds[component][:, param])
        end
    end
end



predictions= predict(model, t_pred)[1]
# # plot final results
figure_final = let f = Figure(size=(700, 280))
    ax = CairoMakie.Axis(f[1,1], xlabel="Time (min)", ylabel="k1", 
    xlabelcolor=:gray, ylabelcolor=:gray, xticklabelsize=8,yticklabelsize=8,xgridvisible=false, ygridvisible=false, topspinevisible=false, rightspinevisible=false)
    ax2 = CairoMakie.Axis(f[1,2], xlabel="Time (min)", ylabel="k5", 
     xlabelcolor=:gray, ylabelcolor=:gray, xticklabelsize=8,yticklabelsize=8,xgridvisible=false, ygridvisible=false, topspinevisible=false, rightspinevisible=false)
    ax3 = CairoMakie.Axis(f[1,3], xlabel="Time (min)", ylabel="k6", 
     xlabelcolor=:gray, ylabelcolor=:gray, xticklabelsize=8,yticklabelsize=8,xgridvisible=false, ygridvisible=false, topspinevisible=false, rightspinevisible=false)
    colors = [colorant"#008aa1", colorant"#c74300", colorant"#5c8a73"]
    
    predictions = predict(model, t_pred)

    for (component,comp_col) in zip([1,2,3], [3,1,2])
        band!(ax, t_pred, lower_bounds[:,1,component], upper_bounds[:,1, component], color=(colors[comp_col], 0.2))
        band!(ax2, t_pred, lower_bounds[:,2,component], upper_bounds[:,2, component], color=(colors[comp_col], 0.2))
        band!(ax3, t_pred, lower_bounds[:,3,component], upper_bounds[:,3, component], color=(colors[comp_col], 0.2))
    end

    for component in 1:best_n_components

        

        lines!(ax, t_pred, predictions[component][:,1], color=colors[component], linewidth=1.5, label="Model $component")
        lines!(ax2, t_pred, predictions[component][:,2], color=colors[component], linewidth=1.5)
        lines!(ax3, t_pred, predictions[component][:,3], color=colors[component], linewidth=1.5)



        component_indices = findall(assignments .== component)
        for individual in component_indices
            locations = findall(ids .== individual)
            scatter!(ax, timepoints_og, parameters[locations, 1], color=(colors[component], 0.2), label="Data $component")
            scatter!(ax2, timepoints_og, parameters[locations, 2], color=(colors[component], 0.2))
            scatter!(ax3, timepoints_og, parameters[locations, 3], color=(colors[component], 0.2))
        end
    end
    Legend(f[2,1:3], ax; orientation=:horizontal, merge=true)
    f
end

save("figures/mixture_model_final_fit.svg", figure_final)

groups = vcat(fill("MIR", size(MIR_parameters, 2)), fill("LIR", size(LIR_parameters, 2)), fill("Healthy", size(healthy_parameters, 2)))

# make a confusion matrix of groups and assignments
group_assignment = DataFrame(group = groups, assignment = assignments)
println(combine(groupby(group_assignment, [:group, :assignment]), nrow => :count))

# label clusters according to majority group
cluster_labels = Dict{Int, String}()
for cluster in 1:best_n_components
    cluster_data = filter(row -> row.assignment == cluster, group_assignment)
    majority_group = combine(groupby(cluster_data, :group), nrow => :count)
    sorted_majority = sort(majority_group, :count, rev=true)
    cluster_labels[cluster] = sorted_majority.group[1]
end


# heatmap of group assignments
figure_confusion = let f = Figure(size=(200,180), aspect=1)
    ax = CairoMakie.Axis(f[1,1], xlabel="Assigned cluster", ylabel="True group",
    xticks = (1:3, ["1", "2", "3"]), yticks = (1:3, ["MIR", "LIR", "Healthy"]), xlabelcolor=:gray, ylabelcolor=:gray, xticklabelsize=8, yticklabelsize=8)
    group_labels = ["MIR", "LIR", "Healthy"]
    assignment_labels = ["1", "2", "3"]
    confusion_matrix = zeros(Int, length(group_labels), length(assignment_labels))
    for row in eachrow(group_assignment)
        true_group = findfirst(==(row.group), group_labels)
        assigned_cluster = row.assignment
        confusion_matrix[assigned_cluster, true_group] += 1
    end
    heatmap!(ax, confusion_matrix; colormap=:Blues)
    for i in 1:size(confusion_matrix, 1)
        for j in 1:size(confusion_matrix, 2)
            value = confusion_matrix[i, j]
            if value < 12
                color = :black
            else
                color = :white
            end
            text!(ax, i, j, text=string(confusion_matrix[i, j]); color=color, align = (:center, :center), font=:bold)
        end
    end
    resize_to_layout!(f)
    f
end

save("figures/mixture_model_confusion_matrix.png", figure_confusion)

# open unseen parameter trajectories, and categorize them with the fitted model
unseen_trajectories = jldopen("data/estimated_parameters_unseen.jld2") do file
    file["estimated_parameters"]
end

unseen_trajectories

# group trajectories by patient_Id and category combination
trajectories_by_individual = Dict{Tuple{Int, String}, Vector{NamedTuple}}()
for point in unseen_trajectories
    patient_id = (point.patient_id, point.category)
    if !haskey(trajectories_by_individual, patient_id)
        trajectories_by_individual[patient_id] = [point]
    else
        push!(trajectories_by_individual[patient_id], point)
    end
end

# classify trajectories
n_timepoints_unseen = Int[]
correct = Bool[]
for trajectory in trajectories_by_individual
    individual_key = trajectory[1]
    trajectory_points = trajectory[2]
    times_individual = [p.global_t for p in trajectory_points]
    parameters_individual = hcat([p.parameters[1:3] for p in trajectory_points]...)
    parameters_individual = Matrix(transpose(parameters_individual))
    println(times_individual)
    # get cluster assignments
    assignments_individual = argmax(
        posterior_responsibilities(model, times_individual, parameters_individual, fill(1, size(parameters_individual, 1))), dims=2)
    println("Patient ID: $(individual_key[1]), Category: $(individual_key[2]), Assigned cluster: $(cluster_labels[assignments_individual[1][2]])")

    push!(n_timepoints_unseen, size(parameters_individual, 1))
    if cluster_labels[assignments_individual[1][2]] == individual_key[2]
        push!(correct, true)
    else
        push!(correct, false)
    end
end

accuracy = sum(correct) / length(correct)
println("Classification accuracy on unseen data: $accuracy")

accuracy_by_timepoints = Dict{Int, Vector{Bool}}()
for (i, n_tp) in enumerate(n_timepoints_unseen)
    if !haskey(accuracy_by_timepoints, n_tp)
        accuracy_by_timepoints[n_tp] = Bool[]
    end
    push!(accuracy_by_timepoints[n_tp], correct[i])
end

for (n_tp, results) in accuracy_by_timepoints
    acc = sum(results) / length(results)
    println("Number of timepoints: $n_tp, Accuracy: $acc out of $(length(results))")
end

# make a heatmap confusion matrix for unseen data
figure_confusion_unseen = let f = Figure(size=(200,180), aspect=1)
    ax = CairoMakie.Axis(f[1,1], xlabel="Assigned cluster", ylabel="True group",
    xticks = (1:3, ["1", "2", "3"]), yticks = (1:3, ["MIR", "LIR", "Healthy"]), xlabelcolor=:gray, ylabelcolor=:gray, xticklabelsize=8, yticklabelsize=8)
    group_labels = ["MIR", "LIR", "Healthy"]
    assignment_labels = ["1", "2", "3"]
    confusion_matrix = zeros(Int, length(group_labels), length(assignment_labels))
    for trajectory in trajectories_by_individual
        individual_key = trajectory[1]
        trajectory_points = trajectory[2]
        times_individual = [p.global_t for p in trajectory_points]
        parameters_individual = hcat([p.parameters[1:3] for p in trajectory_points]...)
        parameters_individual = Matrix(transpose(parameters_individual))
        assignments_individual = argmax(
            posterior_responsibilities(model, times_individual, parameters_individual, fill(1, size(parameters_individual, 1))), dims=2)
        true_group = findfirst(==(individual_key[2]), group_labels)
        assigned_cluster = assignments_individual[1][2]
        confusion_matrix[assigned_cluster, true_group] += 1
    end
    heatmap!(ax, confusion_matrix; colormap=:Blues)
    for i in 1:size(confusion_matrix, 1)
        for j in 1:size(confusion_matrix, 2)
            value = confusion_matrix[i, j]
            if value < 12
                color = :black
            else
                color = :white
            end
            text!(ax, i, j, text=string(confusion_matrix[i, j]); color=color, align = (:center, :center), font=:bold)
        end
    end
    resize_to_layout!(f)
    f
end

save("figures/mixture_model_confusion_matrix_unseen.png", figure_confusion_unseen)