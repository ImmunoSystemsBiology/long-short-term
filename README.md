# Multi-timescale ODE modeling
This project evaluates and presents the use of latent class mixture models for identifying trajectories of ODE model parameters over a long time period. The method is compared to the conditional neural network approach used by Regazzoni et al. and highlights their differences. 

## Development of MIR/LIR
Using the EDES model, we simulated biologically realistic trajectories of OGTT data over a long time period that mimics the development of muscle insulin resistance (MIR) and liver insulin resistance (LIR). To obtain subsequent parsimonious parameter trajectories, we re-estimated a subset of parameters at each long-scale time point using the simulated OGTT data. 

## Recovery of parameter trajectories
We applied latent class mixture models to the re-estimated parameter trajectories to identify distinct classes of parameter trajectories. The results were compared to those obtained using conditional neural networks, as described in Regazzoni et al. We additionally used a latent class conditional neural network approach to combine the strengths of both methods. 

### Questions
 - Univariate vs Multivariate?
 - Alignment with R package?
 - Cross-validation?

## Influence of measurement noise
We investigated the impact of measurement noise on the recovery of parameter trajectories. Simulations were conducted to assess how different levels of measurement noise affected the performance of the latent class mixture models and the conditional neural networks. 

## Class Switching
We explored the phenomenon of class switching, where individuals may transition between different latent classes over time as a more realistic representation of disease progression. We used a hierarchical dirichlet process hidden markov model to allow for class switching and compared the results to the static latent class mixture models.

Can we do a complex simulation?

We have n subjects, each with the following random characteristics:

Parameters controlling the OGTT response
- fasting glucose (G0)
- fasting insulin (I0)
- muscle insulin sensitivity (Sm)
- liver insulin sensitivity (Sl)
- beta cell function (beta)
- body weight (BW)

Lifestyle score:
- lifestyle score (LS)
A lifestyle score that influences muscle and liver insulin sensitivity. A higher lifestyle score indicates a healthier lifestyle. A positive lifestyle score leads to an increase or maintenance of insulin sensitivity, while a negative lifestyle score leads to a decrease in insulin sensitivity.

Parameters controlling the long-term progression:
- muscle lifestyle sensitivity (Cms)
- liver lifestyle sensitivity (Cls)
- beta cell capacity (gamma)
- weight gain tendency (Cw)
- weight loss tendency (Cwl)

Model of lifestyle and muscle insulin sensitivity:
- $LS(t+1) = LS(t) + \epsilon_{ls}, \text{ where } \epsilon_{ls} \sim N(0, \sigma_{ls})$
- $S_m(t+1) = S_{m}(t) + C_{ms} \cdot LS(t+1) + \epsilon_{sm}, \text{ where } \epsilon_{sm} \sim N(0, \sigma_{sm})$

Model of lifestyle and liver insulin sensitivity:
- $S_l(t+1) = S_{l}(t) + C_{ls} \cdot LS(t+1) + \epsilon_{sl}, \text{ where } \epsilon_{sl} \sim N(0, \sigma_{sl})$

Muscle and liver insulin sensitivity influence the EDeS model parameter $k_5$, which saturates at a maximum value of $k_{5,max}$ and a minimum value of $k_{5,min}$ as an S-shaped curve, and is at value $k_{5,0}$ at the initial muscle insulin sensitivity $S_{m,0}$:
- $k_{5,min} = 1 \cdot 10^{-6}$
- $k_{5,max} = 0.05$
- $k_{5,0} = 0.04$
- $k_{5}(t) = k_{5,0}$ when $S_m(t) = S_{m,0}$
- $k_{5}(t) \approx k_{5,max}$ when $S_m(t) >> S_{m,0}$
- $k_{5}(t) \approx k_{5,min}$ when $S_m(t) << S_{m,0}$
The equation for $k_5(t)$ is:

$$ k_5(t) = k_{5,min} + \frac{(k_{5,max} - k_{5,min})}{1 + e^{-(aS_m(t) + bS_l(t))}} $$

where $a$ and $b$ are parameters that control the steepness of the S-curve and the influence of liver and muscle insulin sensitivity on $k_5(t)$. 

Liver insulin sensitivity additionally influences the EDeS model parameter $k_1$, which is at value $k_{1,0}$, which is also its minimum value. As liver insulin sensitivity decreases, $k_1(t)$ increases linearly.
- $k_{1,0} = 0.05$
- $k_1(t) = k_{1,0} + a \cdot (S_l(t) - S_{l,0})^+$

where $S_{l,0}$ is the initial liver insulin sensitivity and $a$ is a parameter that controls the influence of liver insulin sensitivity on $k_1(t)$. The term $(S_l(t) - S_{l,0})^+$ ensures that $k_1(t)$ does not decrease below $k_{1,0}$.

## Compensatory insulin secretion
In the EDeS model, the beta cell function is controlled by parameter $k_6$. To simulate compensatory insulin secretion, we introduce an optimization-based approach to adjust $k_6$ based on the current state of the system. The goal is to maintain glucose homeostasis by adjusting insulin secretion in response to changes in glucose levels. However, as the disease progresses, this compensatory mechanism is reduced, leading to a decline in beta cell function.

First of all, baseline glucose and insulin levels are set to $G_{b,0}$ and $I_{b,0}$. These increase with lifestyle score, which is a measure of the individual's lifestyle habits. The beta cell function is adjusted based on the current glucose level and the desired glucose level, which is set to a target value $G_{b,0}$. Adjustments in basal glucose and insulin are modelled as:
$$\frac{\mathrm{d} G_b}{\mathrm{d}t} = k_1 \cdot I_b - k_2 \cdot G_b - k_{LS} \cdot \tanh(LS(t)$$
$$\frac{\mathrm{d} I_b}{\mathrm{d}t} = k_4 \cdot G_b - k_5 \cdot I_b$$


The adjustment of $k_6$ is based on the current glucose level and the desired glucose level, which is set to a target value. The adjustment is made using a proportional control approach. However, as the disease progresses, the ability of the beta cells to compensate for insulin resistance declines, leading to a gradual decrease in $k_6$. This is modeled by introducing a decay factor that reduces $k_6$ over time:

$$\frac{\mathrm{d} \beta}{\mathrm{d}t} = k_b (G_b - G_{b,0})$$

where $G_b$ is the current glucose level, $G_{b,0}$ is the target glucose level, and $k_b$ is a parameter that controls the rate of change of beta cell function. Model parameter $k_6$ is then adjusted as follows:

$$\frac{\mathrm{d} k_6}{\mathrm{d} t} = \alpha (G_b - G_{b,0}) - \beta(t) \cdot k_6(t)$$

## Weight gain and loss
Body weight (BW) is influenced by lifestyle score (LS) and has a natural tendency to return to a baseline weight (BW0). Weight gain and loss are modeled as follows:
$$\frac{\mathrm{d} BW}{\mathrm{d} t} = C_w \cdot LS(t) - C_{wl} \cdot (BW(t) - BW_0)$$
where $C_w$ is the weight gain tendency and $C_{wl}$ is the weight loss tendency. A positive lifestyle score leads to weight loss, while a negative lifestyle score leads to weight gain. The term $- C_{wl} \cdot (BW(t) - BW_0)$ represents the natural tendency of body weight to return to the baseline weight.

## Simulation Experiments
1. Single parameter change (basic experiment)
2. Dual parameter change 
3. Multi-parameter change, based on diabetes development (increase in body weight, insulin resistance, beta cell function decline, increase in fasting values, etc.)
4. Compensatory parameter change (experimental, need to try this on EDES)

## Hypotheses
1. Latent class mixture models can accurately identify distinct trajectories of ODE model parameters over time
2. Latent class mixture models can outperform conditional neural networks in identifying parameter trajectories, especially when there are multiple distinct trajectories
3. Latent class mixture models can provide more interpretable results compared to conditional neural networks

