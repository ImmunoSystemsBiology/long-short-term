= Combining ODE simulation with Dirichlet Process Hidden Markov Models to discover dynamic parameter trajectories
In many areas of medicine, patients are repeatedly exposed to interventions that elicit short-term dynamic responses — such as glucose excursions during an oral glucose tolerance test, blood pressure responses to pharmacological challenge, or biomarker transients after immune stimulation. These responses can be modeled mechanistically with ordinary differential equations (ODEs), which capture the underlying physiology in a compact and interpretable way. However, patients are not static: their physiological parameters evolve over longer timescales, often in heterogeneous and nonlinear ways. Some individuals progress steadily toward dysfunction, others remain stable, and some undergo remission or relapse. Capturing this diversity is essential for patient monitoring, prognosis, and tailoring interventions.

To address this challenge, we propose a modelling strategy that combines short-term ODE models with a flexible Bayesian nonparametric state model, called the Hierarchical Dirichlet Process - Hidden Markov Model. 

In this framework, the short-term mechanistic response at each visit is explained by an ODE with state-specific parameters, while the slow evolution of these states across visits is governed by a flexible nonparametric Bayesian state model. We use a sticky prior, which encourages biologically realistic persistence, while still allowing asynchronous transitions between various states across individuals. This dual-scale perspective — mechanistic ODEs for fast intervention responses and Bayesian latent states for slow patient trajectories — provides a powerful, interpretable, and data-driven way to track disease progression and remission, opening up new possibilities for stratified patient monitoring and personalized intervention design.

== Model Explanation
We assume we have the following data:
- Subjects $i = 1, ..., N$
- Each subject $i$ has visits $t = 1, ..., T_i$ at slow times $s_(i,t)$
- At each visit, we observe a short fast-time vector $y_(i,t) = (y_(i,t)(tau_1),...,y_(i,t)(tau_L))$

We let $z_(i,t) in {1,2,...}$ represent the latent state of subject $i$ at visit $t$.

The Dirichlet Process is defined by the following priors:
- $beta ~ "GEM"(gamma)$, where $"GEM"$ represents the Griffiths-Engen-McCloskey distribution
- For each state $j$: transition row $pi_j | beta ~ "Dirichlet"(alpha beta + kappa e_j)$, where $kappa$ is the sticky parameter that adds mass to the self-transition $pi_(j,j)$, enocouraging persistence.



