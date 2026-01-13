using JLD2, DataFrames, TemporalMixtureModels, CairoMakie, Statistics, Random, ProgressMeter

# load data
MIR_parameters, LIR_parameters, healthy_parameters = jldopen("data/parameter_trajectories_final.jld2", "r") do file
    file["MIR_parameters"], file["LIR_parameters"], file["healthy_parameters"]
end

timepoints_og = [0.0, 6.0, 12.0, 18.0, 24.0, 30.0]

# prepare data in long format
ids = Int[]
k1s = Float64[]
k5s = Float64[]
k6s = Float64[]
times = Float64[]

for (individual, time) in Iterators.product(1:size(MIR_parameters, 2), 1:length(timepoints))
    push!(ids, individual)
    push!(k1s, MIR_parameters[1, individual, time])
    push!(k5s, MIR_parameters[2, individual, time])
    push!(k6s, MIR_parameters[3, individual, time])
    push!(times, timepoints[time])
end

for (individual, time) in Iterators.product(1:size(LIR_parameters, 2), 1:length(timepoints))
    push!(ids, individual + size(MIR_parameters, 2))
    push!(k1s, LIR_parameters[1, individual, time])
    push!(k5s, LIR_parameters[2, individual, time])
    push!(k6s, LIR_parameters[3, individual, time])
    push!(times, timepoints[time])
end

for (individual, time) in Iterators.product(1:size(healthy_parameters, 2), 1:length(timepoints))
    push!(ids, individual + size(MIR_parameters, 2) + size(LIR_parameters, 2))
    push!(k1s, healthy_parameters[1, individual, time])
    push!(k5s, healthy_parameters[2, individual, time])
    push!(k6s, healthy_parameters[3, individual, time])
    push!(times, timepoints[time])
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

cv_results = cross_validate(ids, times, parameters, 5, possible_n_components, possible_orders, n_repeats=10, n_refits=10)

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


n_bootstrap = 100  # Number of bootstrap samples
confidence_intervals, _, _ = bootstrap_ci(model, input_data; n_bootstrap=n_bootstrap)
# Create a range of time points for prediction
t_pred = -1.0:0.01:1.0

# get cluster assignments
assignments = [x[2] for x in argmax(TemporalMixtureModels.posterior_responsibilities(model, input_data), dims=2)[:]]

figure_mixture = let f = Figure(size=(1200, 400))

    # Plot the fitted component models
    predictions = predict(model, t_pred)

    COMPONENT_COLORS = [colorant"#c74300", colorant"#008aa1", colorant"#00c743"]
    assignments_per_timepoints = repeat(assignments, inner=6) # each individual has 3 timepoints
    assignment_colors = [COMPONENT_COLORS[assignments_per_timepoints[i]] for i in 1:length(assignments_per_timepoints)]
    println(assignments_per_timepoints)
    for (axi, variable) in enumerate([:k1, :k5, :k6])
        ax = Axis(f[1, axi], xlabel="Time", ylabel="Value", title="variable: $variable")
        # Plot the original data points
        scatter!(ax, input_data[input_data[!,:var_name] .== String(variable),:time], input_data[input_data[!,:var_name] .== String(variable),:value]; color=assignment_colors, markersize=5, label="Data - $variable")
        # orders = [2,3,1]
        for k in axes(predictions[variable], 2)
            lines!(ax, t_pred, predictions[variable][:, k];
            label="Component $k - $variable", color=COMPONENT_COLORS[k], linewidth=2)

            # get the confidence bounds. This is still a bit clunky because there is no API for this yet
            lower_bound_parameters = confidence_intervals[k][variable][:lower]
            upper_bound_parameters = confidence_intervals[k][variable][:upper]

            model_lb = PolynomialRegression(length(lower_bound_parameters)-1, lower_bound_parameters)
            model_ub = PolynomialRegression(length(upper_bound_parameters)-1, upper_bound_parameters)

            y_lower = predict(model_lb, t_pred)
            y_upper = predict(model_ub, t_pred)

            println(lower_bound_parameters)

            # Plot the confidence intervals as shaded areas
            lines!(ax, t_pred, y_lower; color=(COMPONENT_COLORS[k], 0.8), linestyle=:dash, label="")
            lines!(ax, t_pred, y_upper; color=(COMPONENT_COLORS[k], 0.8), linestyle=:dash, label="")
            band!(ax, t_pred, y_lower, y_upper;
            color=(COMPONENT_COLORS[k], 0.05), label="Component $k - $variable")

        end
    end

    f
end

groups = vcat(fill("MIR", size(MIR_parameters, 2)), fill("LIR", size(LIR_parameters, 2)), fill("Healthy", size(healthy_parameters, 2)))

# make a confusion matrix of groups and assignments
group_assignment = DataFrame(group = groups, assignment = assignments)
println(combine(groupby(group_assignment, [:group, :assignment]), nrow => :count))

