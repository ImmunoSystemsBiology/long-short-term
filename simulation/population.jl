using Distributions, Random

function generate_individual(body_weight::Distribution, fasting_glucose::Distribution, fasting_insulin::Distribution; rng::AbstractRNG = Random.default_rng())
    bw = rand(rng, body_weight)
    Gb = rand(rng, fasting_glucose)
    Ib = rand(rng, fasting_insulin)
    return (bw = bw, Gb = Gb, Ib = Ib)
end

function generate_population(n_individuals::Int, body_weight::Distribution, fasting_glucose::Distribution, fasting_insulin::Distribution; rng::AbstractRNG = Random.default_rng())
    population = Vector{NamedTuple{(:bw, :Gb, :Ib)}}(undef, n_individuals)
    for i in 1:n_individuals
        population[i] = generate_individual(body_weight, fasting_glucose, fasting_insulin; rng = rng)
    end
    return population
end
