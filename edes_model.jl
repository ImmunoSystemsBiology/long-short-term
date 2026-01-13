using StaticArrays, OrdinaryDiffEq

function edesode(u, p, t)
  Ggut, Gpl, Ipl, Irem = u
  k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, tau_i, tau_d, beta, Gren, EGPb, Km, f, Vg, c1, sigma, Dmeal, bw, Gb, Ib = p

  # gut glucose
  dGgut = sigma * k1^sigma * t^(sigma-1) * exp(-(k1*t)^sigma) * Dmeal - k2 * Ggut

  # plasma glucose
  gliv = EGPb - k3 * (Gpl - Gb) - k4 * beta * Irem
  ggut = k2 * (f / (Vg * bw)) * Ggut
  u_ii =  EGPb * ((Km + Gb)/Gb) * (Gpl / (Km + Gpl))
  u_id = k5 * beta * Irem * (Gpl / (Km + Gpl))
  u_ren = c1 / (Vg * bw) * (Gpl - Gren) * (Gpl > Gren)

  dGpl  = gliv + ggut - u_ii - u_id - u_ren

  # plasma insulin
  i_pnc = beta^(-1) * (k6 * (Gpl - Gb) + (k7 / tau_i) * Gb + k8 * tau_d * dGpl)
  i_liv = k7 * Gb * Ipl / (beta * tau_i * Ib)
  i_int = k9 * (Ipl - Ib)
  dIpl = i_pnc - i_liv - i_int
  dIrem = i_int - k10 * Irem

  return SA[dGgut, dGpl, dIpl, dIrem]
end

function edes_parameters(Dmeal, bw, Gb, Ib ;k1 = 0.0105, k2 = 0.28, k3 = 6.07e-3, k4 = 2.35e-4, k5 = 0.0424, k6 = 2.2975, k7 = 1.15, k8 = 7.27, k9 = 3.83e-2, k10 = 2.84e-1, tau_i = 31.0,
    tau_d = 3.0, beta = 1.0, Gren = 9.0, EGPb = 0.043, Km = 13.2, f = 0.005551, Vg = 17.0 / bw, c1 = 0.1, sigma = 1.4)
    return SA[k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, tau_i, tau_d, beta, Gren, EGPb, Km, f, Vg, c1, sigma, Dmeal, bw, Gb, Ib]
end

function edes_initial_conditions(Gb, Ib)
    return SA[0.0, Gb, Ib, 0.0]
end