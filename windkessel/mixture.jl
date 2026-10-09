# train and test a mixture model on the Windkessel model parameters
using JLD2, TemporalMixtureModels, Random, CairoMakie

rng = MersenneTwister(42)

# load the parameters
windkessel_est = jldopen("data/windkessel_estimation_results.jld2", "r") do file
    file["params"]
end

# load the dataset
dataset = jldopen("data/windkessel_dataset.jld2", "r") do file
    file["dataset"]
end

windkessel_est_n = windkessel_est .- reshape(windkessel_est[:, 1, :], size(windkessel_est,1), 1, size(windkessel_est,3))

# train set (25 patients)
train_indices = randperm(rng, size(windkessel_est,1))[1:25]
train_params = windkessel_est[train_indices, :, :]
train_params_n = train_params .- reshape(train_params[:, 1, :], size(train_params,1), 1, size(train_params,3))

test_indices = setdiff(1:size(windkessel_est,1), train_indices)
test_params = windkessel_est[test_indices, :, :]
test_params_n = test_params .- reshape(test_params[:, 1, :], size(test_params,1), 1, size(test_params,3))

t_obs = Vector{Float64}(0:50)
ids = repeat(1:25, length(t_obs))
t_obs_train = repeat(t_obs, inner=25)

ids_test = repeat(1:length(test_indices), length(t_obs))
t_obs_test = repeat(t_obs, inner=length(test_indices))

y_obs = reshape(train_params_n, size(train_params_n,1)*size(train_params_n,2), size(train_params_n,3))[:,3:4]

y_obs_test = reshape(test_params_n, size(test_params_n,1)*size(test_params_n,2), size(test_params_n,3))[:,3:4]

component_model = @component begin
    y[1] ~ PolynomialRegression(1)
    y[2] ~ PolynomialRegression(1)
end

nc = 3
result = fit_mixture(component_model, nc, t_obs_train, y_obs, ids; n_repeats=200)

alphas = [p.alpha for p in dataset[:patients]]
# predict all data
ids = repeat(1:size(windkessel_est,1), length(t_obs))
t_obs_all = repeat(t_obs, inner=size(windkessel_est,1))
y_obs_all = reshape(windkessel_est_n, size(windkessel_est_n,1)*size(windkessel_est_n,2), size(windkessel_est_n,3))[:,3:4]
predicted_components = TemporalMixtureModels.posterior_responsibilities(result, t_obs_all, y_obs_all, ids)
groups = [argmax(predicted_components[i,:]) for i in 1:size(predicted_components,1)]

figure_alphas_per_group = let f = Figure(size=(600, 300))
    ax = Axis(f[1, 1], xlabel="Group", ylabel="α")
    boxplot!(ax, groups, alphas, color=Makie.wong_colors()[groups])
    f
end

save("figures/windkessel_mixture_model_alphas_per_group.eps", figure_alphas_per_group)

windkessel_est
predictions = predict(result, 0:50)
predictions[1]
figure_results = let f = Figure(size=(1200, 300))
    pnames = ["Rd-Rd(0)", "Cd - Cd(0)"]
    for param in 1:2
        params = windkessel_est_n[:,:,param+2]
        ax = Axis(f[1, param], xlabel="Time", ylabel=pnames[param])
        for i in 1:size(params,1)
            color = if groups[i] == 1
                1
            elseif groups[i] == 2
                2
            else
                3
            end
            scatter!(ax, t_obs, params[i,:], color=(Makie.wong_colors()[color], 0.2), markersize=4)
        end

        for (i,pred) in enumerate(predictions)
            lines!(ax, 0:50, pred[:,param], color=Makie.wong_colors()[i], linewidth=5)
        end

    end
    f
end

save("figures/windkessel_mixture_model_results.eps", figure_results)
    
