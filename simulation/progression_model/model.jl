# Yildirim diabetes progression model

function heav(x)
    return x .>= 0.0 ? 1.0 : 0.0
end

function progression_ode!(du, u, p, t)
    
    g, i, ffa, si, b, sigma, infl, w = u
    eg0, k, bv, mmax, alpha_m, km, alpha_isr, kisr, pmax, kp, alpha_p, p_b,
    amax, alpha_a, ka, a_b, tau_b, height, age_b, sex, cage, target_si, tau_si,
    bmi_h, mffa, ksi_infl, ksi_ffa, nsi_ffa, sw11, inc_i1, inc_i2, inc_i3,
    it1, it2, it3, tau_w, k_infl, n_infl, infl_b, tau_infl, hgp_bas, hepa_max,
    hepa_sic, a_hgp, gcg, k_gcg, s, png, sigma_b, sgmu, sgmd, sfm, sim, sgku,
    sgkd, sfk, sik, nsgku, nsgkd, nsfk, nsik, tau_sigma, sci, gclamp, iclamp,
    maxinfl, l0, l2, cf, ksif, aa, sif, ffa_ramax, lp1, lp2, tau_wsc, DEi = p

    nsi_infl = 6
    bmi = w / height^2
    age = age_b + cage * t / 365

    tsi1 = target_si * (ksi_infl / (ksi_infl + infl)) * (1 - mffa * ffa^nsi_ffa / (ffa^nsi_ffa + ksi_ffa^nsi_ffa))
    tsi2 = target_si * (ksi_infl^nsi_infl / (ksi_infl^nsi_infl + infl^nsi_infl)) * (1 - mffa * ffa^nsi_ffa / (ffa^nsi_ffa + ksi_ffa^nsi_ffa))
    tsi  = sw11 * tsi1 + (1 - sw11) * tsi2

    # intake_i = 2500 made into DEi
    inc_int = 0 + heav(t - it1) * heav(it2 - t) * heav(it3 - t) * inc_i1 + heav(t - it2) * heav(it3 - t) * inc_i2 + heav(t - it3) * inc_i3

    if t >= it2
        tau_w = tau_w * tau_wsc
    end

    w_base = 81
    expd   = 2500 / (7700 * w_base)
    k_w    = 1 / 7700

    m     = mmax * g^km / (alpha_m^km + g^km)
    isr   = sigma * m^kisr / (alpha_isr^kisr + m^kisr)
    p_isr = p_b + pmax * isr^kp / (alpha_p^kp + isr^kp)
    a_m   = a_b + amax * m^ka / (alpha_a^ka + m^ka)

    p = p_isr
    a = a_m

    fmass_p = 1.2 * bmi + 0.23 * age - 16.2
    fmass   = w * fmass_p / 100

    inc    = 1 + inc_int
    hepa_si = hepa_sic * si
    hgp     = (hgp_bas + hepa_max * (a_hgp + k_gcg * gcg)) /
            ((a_hgp + k_gcg * gcg) + hepa_si * i)

    intake = DEi * inc

    s_glucu = sgmu * g^nsgku / (g^nsgku + sgku^nsgku)
    s_glucd = sgmd * g^nsgkd / (g^nsgkd + sgkd^nsgkd)
    s_ffa   = sfm * ffa^nsfk / (ffa^nsfk + sfk^nsfk)
    s_infl  = sim * infl^nsik / (infl^nsik + sik^nsik)

    s_inf = sigma_b * (s_glucu - s_glucd * s_ffa - s_infl)

    siff = si * sif

    cl0 = l0 * 24 * 60 / bv
    cl2 = l2 * 24 * 60 / bv
    al0 = 2.45 * 24 * 60

    du[1] = gclamp + hgp - (eg0 + sci * si * i) * g
    du[2] = iclamp + b * isr / bv - k * i
    du[3] = (cl0 + cl2 * fmass) * (ffa_ramax * (ksif^aa) / (ksif^aa + (siff * i)^aa)) - cf * w * ffa
    du[4] = (tsi - si) / tau_si
    du[5] = (png + p * b - a * b - s * b^2) / tau_b
    du[6] = (s_inf - sigma) / tau_sigma
    du[7] = infl_b + (maxinfl * bmi^n_infl / (bmi^n_infl + k_infl^n_infl) - infl) / tau_infl
    du[8] = (k_w * intake - expd * w) / tau_w

end

function yildirim_parameters(; 
    eg0=24.48,
    k=700,
    bv=5,
    mmax=1,
    alpha_m=140,
    km=2,
    alpha_isr=1.2,
    kisr=2,
    pmax=4.55,
    kp=4,
    alpha_p=35,
    p_b=0,
    amax=5,
    alpha_a=0.37,
    ka=6,
    a_b=0.9,
    tau_b=1800,
    height=1.8,
    age_b=30,
    sex=1,
    cage=0,
    target_si=1.4,
    tau_si=1,
    bmi_h=25,
    mffa=0.8,
    ksi_infl=1.8,
    ksi_ffa=400,
    nsi_ffa=6,
    sw11=1,
    inc_i1=0,
    inc_i2=0,
    inc_i3=0,
    it1=0,
    it2=100000,
    it3=100000,
    tau_w=1.5,
    k_infl=40,
    n_infl=6,
    infl_b=0,
    tau_infl=1,
    hgp_bas=2000,
    hepa_max=3000,
    hepa_sic=1,
    a_hgp=4,
    gcg=0.1,
    k_gcg=0,
    s=0.0002,
    png=350,
    sigma_b=536,
    sgmu=1.5,
    sgmd=1,
    sfm=1.2,
    sim=0.25,
    sgku=81,
    sgkd=137,
    sfk=357,
    sik=0.6,
    nsgku=6,
    nsgkd=6,
    nsfk=6,
    nsik=4,
    tau_sigma=1,
    sci=1,
    gclamp=0,
    iclamp=0,
    maxinfl=1,
    l0=170,
    l2=8.1,
    cf=2,
    ksif=11,
    aa=2,
    sif=1,
    ffa_ramax=1.106,
    lp1=100,
    lp2=125,
    tau_wsc=1,
    DEi=2500
)
    return Float64[
        eg0, k, bv, mmax, alpha_m, km, alpha_isr, kisr, pmax, kp, alpha_p, p_b,
        amax, alpha_a, ka, a_b, tau_b, height, age_b, sex, cage, target_si, tau_si,
        bmi_h, mffa, ksi_infl, ksi_ffa, nsi_ffa, sw11, inc_i1, inc_i2, inc_i3,
        it1, it2, it3, tau_w, k_infl, n_infl, infl_b, tau_infl, hgp_bas, hepa_max,
        hepa_sic, a_hgp, gcg, k_gcg, s, png, sigma_b, sgmu, sgmd, sfm, sim, sgku,
        sgkd, sfk, sik, nsgku, nsgkd, nsfk, nsik, tau_sigma, sci, gclamp, iclamp,
        maxinfl, l0, l2, cf, ksif, aa, sif, ffa_ramax, lp1, lp2, tau_wsc, DEi
    ]
end

function parameter_index(val)
    params = [
        :eg0, :k, :bv, :mmax, :alpha_m, :km, :alpha_isr, :kisr, :pmax, :kp, :alpha_p, :p_b,
        :amax, :alpha_a, :ka, :a_b, :tau_b, :height, :age_b, :sex, :cage, :target_si, :tau_si,
        :bmi_h, :mffa, :ksi_infl, :ksi_ffa, :nsi_ffa, :sw11, :inc_i1, :inc_i2, :inc_i3,
        :it1, :it2, :it3, :tau_w, :k_infl, :n_infl, :infl_b, :tau_infl, :hgp_bas, :hepa_max,
        :hepa_sic, :a_hgp, :gcg, :k_gcg, :s, :png, :sigma_b, :sgmu, :sgmd, :sfm, :sim, :sgku,
        :sgkd, :sfk, :sik, :nsgku, :nsgkd, :nsfk, :nsik, :tau_sigma, :sci, :gclamp, :iclamp,
        :maxinfl, :l0, :l2, :cf, :ksif, :aa, :sif, :ffa_ramax, :lp1, :lp2, :tau_wsc, :DEi
    ]
    return findfirst(==(val), params)
end

function yildirim_state(;
    w = 81.0,  # body weight in kg
    g = 94.1,  # fasting glucose in
    i = 9.6,   # fasting insulin in
    ffa = 404.0, # fasting free fatty acids in umol/L
    si = 0.8,    # insulin sensitivity in ml/microU/day
    b = 1000.0,  # beta cell mass in mg
    sigma = 530.0, # insulin secretion rate in microU/mg/day
    infl = 0.56  # systematic inflammation index
)
    return [g, i, ffa, si, b, sigma, infl, w]
end 