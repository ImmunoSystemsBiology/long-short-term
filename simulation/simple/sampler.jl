# --- sticky HDP-HMM (truncated) Gibbs sampler for ODE emissions ---
# This is a practical, usable implementation skeleton in Julia that implements
# a truncated sticky-HDP-HMM approximation (finite K) and a blocked Gibbs
# sampler using FFBS (forward-filter backward-sample) for latent states.
# Emissions are ODE-response curves (fast-time measurements) simulated per visit.
#
# Key approximations / choices made for tractability here:
# - We use a finite truncation K states (weak-limit approximation of HDP).
# - Global weights beta are given a symmetric Dirichlet prior (approximate SB).
# - Transition rows are Dirichlet with extra mass kappa on self-transition (sticky).
# - For sampling z (state sequences) we treat each subject's visits as an HMM
#   with emissions independent across visits (emission at a visit is the vector of
#   fast-time observations for that visit). The emission likelihood is computed by
#   solving the ODE under the candidate state's parameters and evaluating a multivariate
#   Gaussian likelihood with independent noise across fast-time samples.
# - Cluster parameters theta_k (k=1..K) are updated via random-walk MH using ODE
#   likelihood of all visit data assigned to that cluster.
#
# This code is meant to be readable and modifiable, not the fastest possible.

using Random, Distributions, LinearAlgebra, StatsBase
using OrdinaryDiffEq
using CSV, DataFrames

# ------------------------- Helper I/O -------------------------
# expects CSV like the simulator: columns subject, slow_time, fast_time, y_obs
println("Loading simulated data...")
df = CSV.read("sim_two_timescale_intermediate.csv", DataFrame)

# Organize data: for each subject and slow_time (visit) collect the vector of fast times and y_obs
struct VisitData
    subject::Int
    visit::Int
    fast_t::Vector{Float64}
    y::Vector{Float64}
end

function build_visits(df)
    visits = Vector{VisitData}()
    grouped = groupby(df, [:subject, :slow_time])
    for g in grouped
        s = g.subject[1]
        v = g.slow_time[1]
        push!(visits, VisitData(s, v, collect(g.fast_time), collect(g.y_obs)))
    end
    return visits
end

visits = build_visits(df)
S = length(unique(df.subject))
V = length(visits)
println("Loaded $S subjects and $V visits total")

# Map visits by subject (ordered by slow_time) to build per-subject HMM sequences
subject_to_visit_idxs = Dict{Int, Vector{Int}}()
for (idx, vis) in enumerate(visits)
    if haskey(subject_to_visit_idxs, vis.subject)
        push!(subject_to_visit_idxs[vis.subject], idx)
    else
        subject_to_visit_idxs[vis.subject] = [idx]
    end
end
# sort indices by visit time to ensure temporal ordering
for (s, idxs) in subject_to_visit_idxs
    sort!(idxs, by=i->visits[i].visit)
end

# -------------------- Model / Prior hyperparameters --------------------
K = 6                      # truncation level (max number of states)
alpha0 = 1.0               # concentration for transition Dirichlet (per row)
gamma0 = 1.0               # concentration for global stick-breaking Dirichlet
kappa = 10.0               # sticky parameter (extra mass on self-transition)
sigma_y = 0.03             # known observation noise sd (can be learned)

# Prior for theta_k: independent normals on transformed scale (k positive, b real)
# We'll parametrize theta as (logk, b) to keep k>0
prior_mu = [log(0.03), 0.02]
prior_cov = Diagonal([0.5, 0.5])
prior_dist = MvNormal(prior_mu, prior_cov)

# -------------------- Initialization --------------------
# Initialize global weights beta ~ Dirichlet(gamma0/K)
beta = rand(Dirichlet(fill(gamma0/K, K)))
# Initialize transition matrix rows A[j,:] ~ Dirichlet(alpha0*beta + kappa*e_j)
A = zeros(K,K)
for j in 1:K
    base = alpha0 .* beta
    base[j] += kappa
    A[j, :] = rand(Dirichlet(base))
end
# Initialize state parameters theta_k by sampling from prior
theta = [exp(prior_mu[1]) => prior_mu[2] for _ in 1:K] # placeholder pairs; we'll store as Dict
# We'll store as vector of tuples (k,b) but param proposals in log-space
theta = [(exp(prior_mu[1]), prior_mu[2]) for _ in 1:K]

# Initialize latent state assignments per visit by sampling from beta
z = [rand(Categorical(beta)) for _ in 1:V]

# Precompute unique fast time grid and a function to compute emission likelihoods
fast_grid = unique(df.fast_time)
fast_grid = sort(fast_grid)

# analytic integrator for linear ODE dx/dt = -k x + b given initial x0=1
function ode_predict_linear(k, b, times)
    return [1.0*exp(-k*t) + b*(1 - exp(-k*t))/k for t in times]
end

function log_emission_likelihood(theta_k, visit::VisitData; σ=sigma_y)
    k, b = theta_k
    pred = ode_predict_linear(k, b, visit.fast_t)
    # independent Gaussian noise
    res = visit.y .- pred
    ll = -0.5 * sum((res.^2) ./ (σ^2)) - (length(res)/2)*log(2π*σ^2)
    return ll
end

# caching emission likelihoods per (visit, k)
emission_cache = Dict{Tuple{Int,Int}, Float64}()  # (visit_idx, k) => loglik
function get_emission_loglik(visit_idx::Int, k::Int)
    key = (visit_idx, k)
    if haskey(emission_cache, key)
        return emission_cache[key]
    else
        ll = log_emission_likelihood(theta[k], visits[visit_idx])
        emission_cache[key] = ll
        return ll
    end
end

# -------------------- FFBS for a single subject --------------------
# Given sequences of visit indices for a subject, emission logliks and A, sample z sequence
function sample_z_for_subject(visit_idxs::Vector{Int})
    Tsub = length(visit_idxs)
    # Forward pass: compute alpha messages (log-space)
    log_alpha = zeros(K, Tsub)
    # initial prior for first state = beta (global) -- use log
    for k in 1:K
        log_alpha[k,1] = log(beta[k]) + get_emission_loglik(visit_idxs[1], k)
    end
    # normalize in log-space
    for t in 2:Tsub
        for k in 1:K
            # logsum over j of alpha[j,t-1] + log A[j,k]
            logs = log.(A[:,k]) .+ log_alpha[:, t-1]
            maxlog = maximum(logs)
            logsum = maxlog + log(sum(exp.(logs .- maxlog)))
            log_alpha[k,t] = get_emission_loglik(visit_idxs[t], k) + logsum
        end
    end
    # Backward sampling (FFBS)
    zsub = zeros(Int, Tsub)
    # sample last state from normalized alpha[:,T]
    probs_last = exp.(log_alpha[:,Tsub] .- logsumexp(log_alpha[:,Tsub]))
    zsub[Tsub] = rand(Categorical(probs_last))
    for t in (Tsub-1):-1:1
        k_next = zsub[t+1]
        # p(z_t = j | z_{t+1}=k_next, data) ∝ alpha[j,t] * A[j,k_next]
        logp = log_alpha[:,t] .+ log.(A[:, k_next])
        probs = exp.(logp .- logsumexp(logp))
        zsub[t] = rand(Categorical(probs))
    end
    return zsub
end

# utility log-sum-exp
function logsumexp(v)
    m = maximum(v)
    return m + log(sum(exp.(v .- m)))
end

# -------------------- Gibbs sampler --------------------
niter = 800
burnin = 200
thin = 5
samples = Dict(:z => [], :A => [], :theta => [])

println("Starting Gibbs sampler (K=$K, iterations=$niter) ...")
for iter in 1:niter
    # 1) Sample latent state sequences z for each subject via FFBS
    # rebuild global visit order z vector by subject
    # clear emission cache because theta may have changed in previous iter
    empty!(emission_cache)
    start_idx = 1
    for (s, visit_idxs) in subject_to_visit_idxs
        zsub = sample_z_for_subject(visit_idxs)
        for (local_idx, vidx) in enumerate(visit_idxs)
            z[vidx] = zsub[local_idx]
        end
    end

    # 2) Update transition counts and sample A rows
    C = zeros(Int, K, K)
    # for each subject, count transitions along its visit sequence
    for (_s, visit_idxs) in subject_to_visit_idxs
        for t in 1:(length(visit_idxs)-1)
            i = visit_idxs[t]
            j = visit_idxs[t+1]
            C[z[i], z[j]] += 1
        end
    end
    # sample A[j,:] ~ Dirichlet(alpha0*beta + kappa*e_j + C[j,:])
    for j in 1:K
        base = alpha0 .* beta .+ C[j, :]
        base[j] += kappa
        A[j, :] = rand(Dirichlet(base))
    end

    # 3) (Optional) update beta (global weights) via Dirichlet approximation: use normalized counts + gamma0
    # here we use simple update: beta ~ Dirichlet(gamma0/K + sum_j C[j,k])
    nk = sum(C, dims=1)[:] .+ 1e-6 # ensure positive
    global beta = rand(Dirichlet(gamma0/K .+ nk))

    # 4) Sample/update theta_k via MH using visits assigned to k
    # Propose on transformed space: φ = (logk, b)
    for k_idx in 1:K
        # collect visit indices assigned to k
        assigned = findall(i->z[i]==k_idx, 1:V)
        if isempty(assigned)
            # sample from prior to allow birth of new cluster
            φ_current = [log(theta[k_idx][1]), theta[k_idx][2]]
            φ_prop = φ_current .+ rand(MvNormal([0.0,0.0], Diagonal([0.05, 0.01])))
            θ_prop = (exp(φ_prop[1]), φ_prop[2])
            # accept with prior ratio only (no data)
            logp_prior_curr = logpdf(prior_dist, φ_current)
            logp_prior_prop = logpdf(prior_dist, φ_prop)
            if log(rand()) < (logp_prior_prop - logp_prior_curr)
                theta[k_idx] = θ_prop
            end
        else
            # compute current log-likelihood for assigned visits
            ll_curr = 0.0
            for vidx in assigned
                ll_curr += log_emission_likelihood(theta[k_idx], visits[vidx])
            end
            φ_current = [log(theta[k_idx][1]), theta[k_idx][2]]
            # propose
            φ_prop = φ_current .+ rand(MvNormal([0.0,0.0], Diagonal([0.02,0.005])))
            θ_prop = (exp(φ_prop[1]), φ_prop[2])
            # compute prop log-lik
            ll_prop = 0.0
            for vidx in assigned
                ll_prop += log_emission_likelihood(θ_prop, visits[vidx])
            end
            # prior terms
            logp_prior_curr = logpdf(prior_dist, φ_current)
            logp_prior_prop = logpdf(prior_dist, φ_prop)
            # MH acceptance
            log_accept = (ll_prop + logp_prior_prop) - (ll_curr + logp_prior_curr)
            if log(rand()) < log_accept
                theta[k_idx] = θ_prop
            end
        end
    end

    # monitoring / store samples after burnin
    if iter > burnin && (iter - burnin) % thin == 0
        push!(samples[:A], copy(A))
        push!(samples[:theta], deepcopy(theta))
        push!(samples[:z], copy(z))
    end

    if iter % 50 == 0
        println("iter $iter: sample stored so far: $(length(samples[:z]))")
    end
end

println("Gibbs sampling complete. Stored $(length(samples[:z])) posterior samples.")

# Save last sample's Viterbi-ish path for inspection
last_z = samples[:z][end]
# map back to data and add a column
zcol = [string(last_z[i]) for i in 1:V]
# attach to visits -> for quick inspection, write per-visit assignments
out_rows = Vector{NamedTuple}()
for (i, vis) in enumerate(visits)
    push!(out_rows, (subject=vis.subject, visit=vis.visit, visit_idx=i, assigned_state=last_z[i]))
end
outdf = DataFrame(out_rows)

df_states = df[df[!,:fast_time] .== 0.0, :]
outdf[!, :real_state] = df_states[!, :state] 

# plot states over visits per subject
plot(outdf.visit, outdf.assigned_state, group=outdf.real_state, seriestype=:scatter, legend=false,
    title="Visit state assignments per subject", xlabel="Visit (slow_time)", ylabel="Assigned state")

# get a count matrix of combinations of (real_state, assigned_state)
ct = combine(groupby(outdf, [:real_state, :assigned_state]), nrow => :count)
println("State assignment counts:")
println(ct)

CSV.write("hmm_assignments.csv", outdf)
println("Wrote visit assignments to hmm_assignments.csv")