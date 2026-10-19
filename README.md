# Multi-timescale ODE modeling
This project evaluates and presents the use of latent class mixture models for identifying trajectories of ODE model parameters over a long time period. The method is compared to the conditional neural network approach used by Regazzoni et al. and highlights their differences. 

## Development of MIR/LIR
Using the EDES model, we simulated biologically realistic trajectories of OGTT data over a long time period that mimics the development of muscle insulin resistance (MIR) and liver insulin resistance (LIR). To obtain subsequent parsimonious parameter trajectories, we re-estimated a subset of parameters at each long-scale time point using the simulated OGTT data. 

## Recovery of parameter trajectories
We applied latent class mixture models to the re-estimated parameter trajectories to identify distinct classes of parameter trajectories. The results were compared to those obtained using conditional neural networks, as described in Regazzoni et al. We additionally used a latent class conditional neural network approach to combine the strengths of both methods. 
