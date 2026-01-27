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

figure_cv = let f = Figure()

    ax = CairoMakie.Axis(f[1,1], xlabel="order", ylabel="bic", title="BIC from cross-validation")
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
        boxplot!(ax, nc_order, nc_aics, label="n components: $nc"; whiskerwidth=0.5, width=0.2)
    end
    Legend(f[1,2], ax; orientation=:vertical)
    f

end

save("figures/mixture_model_cv_bic.png", figure_cv)

# fit final model on all data with best parameters (3 components, order 1)
best_n_components = 3
best_order = 1

component_model = @component begin
    y[1] ~ PolynomialRegression(best_order)
    y[2] ~ PolynomialRegression(best_order)
    y[3] ~ PolynomialRegression(best_order)
end

model = fit_mixture(component_model, best_n_components, times, parameters, ids; n_repeats=10, max_iter=1000, verbose=true)
# get cluster assignments
assignments = argmax(posterior_responsibilities(model, times, parameters, ids), dims=2)
assignments = [a[1][2] for a in eachrow(assignments)]

t_pred = range(0, stop=30, length=100)

# # plot final results
# figure_final = let f = Figure(size=(700, 300))
#     ax = CairoMakie.Axis(f[1,1], xlabel="Time (min)", ylabel="k1", title="Mixture Model Reconstruction")
#     limits!(ax, 0, 30, 0, 0.05)
#     ax2 = CairoMakie.Axis(f[1,2], xlabel="Time (min)", ylabel="k5")
#     ax3 = CairoMakie.Axis(f[1,3], xlabel="Time (min)", ylabel="k6")
#     colors = [colorant"#008aa1", colorant"#c74300", colorant"#5c8a73"]
#     for component in 1:best_n_components
#         component_indices = findall(assignments .== component)
#         for individual in component_indices
#             locations = findall(ids .== individual)
#             scatter!(ax, timepoints_og, parameters[locations, 1], color=colors[component], markersize=8)
#             scatter!(ax2, timepoints_og, parameters[locations, 2], color=colors[component], markersize=8)
#             scatter!(ax3, timepoints_og, parameters[locations, 3], color=colors[component], markersize=8)
#         end
#     end
#     f
# end

groups = vcat(fill("MIR", size(MIR_parameters, 2)), fill("LIR", size(LIR_parameters, 2)), fill("Healthy", size(healthy_parameters, 2)))

# make a confusion matrix of groups and assignments
group_assignment = DataFrame(group = groups, assignment = assignments)
println(combine(groupby(group_assignment, [:group, :assignment]), nrow => :count))

