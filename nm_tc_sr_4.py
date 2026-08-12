from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

try:
    from numba import njit

    HAVE_NUMBA = True
except ImportError:  # pure-Python fallback, ~250x slower
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        def wrap(func):
            return func

        return wrap(args[0]) if args and callable(args[0]) else wrap


# ---------------------------------------------------------------------------
# Fixed simulation settings
# ---------------------------------------------------------------------------
ONSET = 20
RES = int(1e4)
RED = int(1e2)          # fast traces -> 100 Hz
RED_SLOW = int(1e3)     # ISO / regulatory traces -> 10 Hz.  These are all
                        # 0.02 Hz or slower, and iso_phase() interpolates the
                        # recorded phase onto the fast base, so they do not
                        # need to share the fast sampling rate.  At 100 Hz a
                        # 24 h run would hold ~900 MB of slow traces alone.
DT = 1e3 / RES

_A = (0.5, 0.5, 1.0, 1.0)
_B = (0.75, 0.75, 0.0, 0.0)
_SQRT3 = math.sqrt(3.0)
_C1 = math.pi / math.sqrt(3.0)
_TWO_PI = 2.0 * math.pi

# ---------------------------------------------------------------------------
# Cortical module parameters
# ---------------------------------------------------------------------------
tau_p, tau_i = 30.0, 30.0
Qp_max, Qi_max = 30.0e-3, 60.0e-3
theta_p, theta_i = -58.5, -58.5
sigma_i = 6.0
alpha_Na = 1.70   # reduced from 2.0 → near the K-complex threshold so ISO jitter gates K-complexes stochastically
R_pump, Na_eq = 0.09, 9.5
gamma_e_c, gamma_g_c = 70e-3, 58.6e-3
nu_c = 120e-3
C_m = 1.0
g_L = 1.0
g_AMPA, g_GABA = 1.0, 1.0
E_AMPA, E_GABA = 0.0, -70.0
E_L_i = -64.0
E_K = -100.0
N_ip, N_pi, N_ii = 72.0, 90.0, 90.0
sigma_p_0, tau_s = 7.0, 100.0
g_KNa_0, tau_g = 1.33, 10.0

CORTEX_TC = (1.7, -64.0, 115.0)     # NM_TC calibration (default)
CORTEX_SR = (1.0, -66.0, 120.0)     # NM_Cortex_SR calibration

# ---------------------------------------------------------------------------
# Thalamic module parameters
# ---------------------------------------------------------------------------
tau_t, tau_r = 20.0, 20.0
Qt_max, Qr_max = 400.0e-3, 400.0e-3
theta_t, theta_r = -58.5, -58.5
sigma_t, sigma_r = 6.0, 6.0
gamma_e_t, gamma_g_t = 70e-3, 100e-3
nu_t = 120e-3
g_T_t, g_T_r = 3.0, 2.3
E_L_t, E_L_r = -70.0, -70.0
E_Ca, E_h = 120.0, -40.0
alpha_Ca, tau_Ca, Ca_0 = -51.8e-6, 10.0, 2.4e-4
k1, k2, k3, k4 = 2.5e7, 4e-4, 1e-1, 1e-3
g_inc = 2.0
dphi_t = 20e-3
N_rt, N_tr, N_rr = 3.0, 5.0, 25.0

# ---------------------------------------------------------------------------
# Sleep regulation parameters
# ---------------------------------------------------------------------------
tau_W, tau_N, tau_R = 1500e3, 600e3, 60e3
tau_E, tau_G, tau_A = 25e3, 10e3, 10e3
F_W_max, F_N_max, F_R_max = 6.5, 5.0, 5.0
alpha_W, alpha_N, alpha_R = 0.5, 0.175, 0.13
beta_W, beta_R = -0.4, -0.9
gamma_E, gamma_G, gamma_A = 5.0, 4.0, 2.0
g_GW, g_AW = -1.68, 1.0
g_GR, g_AR, g_ER = -1.3, 1.6, -4.0
g_EN = -2.0
H_max, theta_W = 1.0, 2.0
tau_hw, tau_hs = 34830e3, 30600e3
kappa = 1.5

# ---------------------------------------------------------------------------
# Infraslow oscillation defaults
# ---------------------------------------------------------------------------
F_ISO = 0.02            # Hz  -> 50 s period
# NE modulation depth (aU of C_E, gated by C_G).  0.9 puts the NE peak at
# C_E_eff ~ 0.77, i.e. sigma_p ~ 3.9 and g_KNa ~ 1.2 - genuinely wake-like, so
# the ascending phase produces real arousals.  Do not lower this much without
# rechecking the arousal measure: at 0.4 the cortex never leaves deep NREM
# (sigma_p only reaches 5.6) and the arousal index inverts.  See iso_analysis.
A_ISO = 0.90
B_ISO = 0.008           # thalamic g_LK modulation depth (mS/m^2)
G_LK_ISO = 0.026        # baseline g_LK (NM_TC N3); ISO adds on top

# The real infraslow NE fluctuation is quasi-periodic, not a clean sinusoid:
# cycle length wanders and peaks differ in size.  Two Ornstein-Uhlenbeck
# processes reproduce that, both band-limited so the drive stays smooth.
#
#   frequency:  dxi = -xi/tau_f dt + sigma_f sqrt(2/tau_f) dW,  stationary
#               std = ISO_JITTER, and the phase advances at w*(1 + xi).
#   amplitude:  same form, mean 1, stationary CV = ISO_AMP_CV.
#
# Using OU rather than white noise on the phase is deliberate: white phase
# noise (plain diffusion) puts power at every frequency and makes the drive
# visibly ragged between cycles.  Filtering it at tau_f keeps the period
# wandering on a multi-cycle timescale, which is what the recordings show.
# Set ISO_JITTER = ISO_AMP_CV = 0 for the deterministic sinusoid.
ISO_JITTER = 0.30       # relative frequency CV -> cycle-period variability
ISO_FREQ_TAU = 100.0    # frequency correlation time, s (~2 ISO cycles)
ISO_AMP_CV = 0.35       # amplitude CV
ISO_AMP_TAU = 60.0      # amplitude correlation time, s

# ---------------------------------------------------------------------------
# State layout
# ---------------------------------------------------------------------------
(VP, VI, NA, S_EP, S_EI, S_GP, S_GI, CY, X_EP, X_EI, X_GP, X_GI, CX,
 G_KNA, SIGMA_P) = range(15)
N_CORTEX = 15

(VT, VR, CA, S_ET, S_ER, S_GT, S_GR, TY, X_ET, X_ER, X_GT, X_GR, TX,
 H_T_T, H_T_R, M_H, M_H2) = range(17)
N_THALAMUS = 17

F_W, F_N, F_R, C_E, C_G, C_A, H = range(7)
N_SR = 7

# AC-coupling cutoff for the EEG proxy, Hz.  Not a knob: Vp is a DC-coupled
# population membrane potential resting near E_L_p, so its mean sits ~11 mV
# higher in wake/REM than in NREM (no I_KNa pulling it toward E_K).  Real EEG
# never shows that step because amplifiers are AC-coupled.  High-passing
# reproduces the recording chain - stage means collapse to 0 while the
# amplitude structure survives.  The reference MATLAB does the same
# (ft_preproc_bandpassfilter in NM_TC-master/Figures/Data_SO_Average.m).
EEG_HP = 0.5

FAST_KEYS = ("Vp", "Vi", "Vt", "Vr", "Ca", "act_h")
SLOW_KEYS = ("f_W", "f_N", "f_R", "C_E", "C_G", "C_A", "h", "g_KNa", "sigma_p",
             "NE", "g_LK", "iso", "iso_phi")


def _run_chunk(n_chunk, step0, C, Th, S, rand_c, rand_t,
               out_fast, out_slow, cnt_fast, cnt_slow,
               onset_steps, red, red_slow,
               tau_Na, E_L_p, N_pp, dphi_c,
               g_LK_base, g_h, N_tp, N_rp, N_pt, N_it,
               f_iso, A_iso, B_iso, iso_gate, iso_sr_fb, clamp_sr,
               iso_state, iso_jitter, iso_freq_tau, iso_amp_cv, iso_amp_tau):
    """Advance the model by n_chunk steps, mutating state in place."""
    ge2_c = gamma_e_c * gamma_e_c
    gg2_c = gamma_g_c * gamma_g_c
    nu2_c = nu_c * nu_c
    ge2_t = gamma_e_t * gamma_e_t
    gg2_t = gamma_g_t * gamma_g_t
    nu2_t = nu_t * nu_t
    Na_eq3 = Na_eq ** 3
    n_fast = out_fast.shape[1]
    n_slow = out_slow.shape[1]
    w_iso = _TWO_PI * f_iso / RES     # phase advance per integration step
    # ISO noise is updated every iso_sub steps (10 ms) rather than every step:
    # the drive is 0.02 Hz, so this is still ~5000x faster than the process
    # it perturbs, and it keeps two extra RNG draws out of the inner loop.
    iso_sub = 100
    dt_iso = iso_sub / RES                              # s
    sig_frq = iso_jitter * math.sqrt(2.0 * dt_iso / iso_freq_tau)
    sig_amp = iso_amp_cv * math.sqrt(2.0 * dt_iso / iso_amp_tau)

    for k in range(n_chunk):
        step = step0 + k

        # ---- infraslow noradrenergic drive for this step -----------------
        # phase and amplitude are integrated state, not functions of time, so
        # the analysis must use the recorded phase (see iso_phase)
        if step % iso_sub == 0:
            if sig_frq > 0.0:
                xi = iso_state[2]
                xi += -xi * dt_iso / iso_freq_tau + sig_frq * np.random.normal(0.0, 1.0)
                # the phase must stay monotonic - iso_phase interpolates on it
                iso_state[2] = xi if xi > -0.8 else -0.8
            if sig_amp > 0.0:
                a = iso_state[1]
                a += (1.0 - a) * dt_iso / iso_amp_tau + sig_amp * np.random.normal(0.0, 1.0)
                iso_state[1] = min(max(a, 0.0), 2.0)
        iso_state[0] += w_iso * (1.0 + iso_state[2])
        phase = iso_state[0]
        # nominally 0..1, but a high-amplitude cycle can overshoot; only the
        # floor is enforced, so peak size stays a real degree of freedom
        iso = iso_state[1] * 0.5 * (1.0 + math.sin(phase))
        if iso < 0.0:
            iso = 0.0

        # ================= one SRK4 step over all three modules ==========
        for n in range(4):
            a_dt = _A[n] * DT
            b = _B[n]

            # ---------------- sleep regulation -----------------------
            f_W, f_N, f_R = S[F_W, n], S[F_N, n], S[F_R, n]
            C_e, C_g, C_a, h = S[C_E, n], S[C_G, n], S[C_A, n], S[H, n]

            # NE seen by the rest of the model: baseline plus the ISO bump,
            # gated by the NREM marker C_G unless gating is disabled
            gate = C_g if iso_gate != 0.0 else 1.0
            C_e_eff = C_e + A_iso * gate * iso
            # thalamic K leak runs anti-phase to NE
            g_LK = g_LK_base + B_iso * gate * (1.0 - iso)

            if clamp_sr != 0.0:
                # hold the sleep switch stationary: the ISO is then the only
                # thing moving the background state, which is what you want
                # for a controlled phase analysis
                for v in range(N_SR):
                    S[v, n + 1] = S[v, 0]
            else:
                C_e_sr = C_e_eff if iso_sr_fb != 0.0 else C_e
                I_W = g_GW * C_g + g_AW * C_a
                I_N = g_EN * C_e_sr
                I_R = g_ER * C_e_sr + g_GR * C_g + g_AR * C_a

                S[F_W, n + 1] = S[F_W, 0] + a_dt * (
                    F_W_max * 0.5 * (1.0 + math.tanh((I_W - beta_W) / alpha_W)) - f_W) / tau_W
                S[F_N, n + 1] = S[F_N, 0] + a_dt * (
                    F_N_max * 0.5 * (1.0 + math.tanh((I_N + kappa * h) / alpha_N)) - f_N) / tau_N
                S[F_R, n + 1] = S[F_R, 0] + a_dt * (
                    F_R_max * 0.5 * (1.0 + math.tanh((I_R - beta_R) / alpha_R)) - f_R) / tau_R
                S[C_E, n + 1] = S[C_E, 0] + a_dt * (
                    math.tanh(S[F_W, n + 1] / gamma_E) - C_e) / tau_E
                S[C_G, n + 1] = S[C_G, 0] + a_dt * (
                    math.tanh(S[F_N, n + 1] / gamma_G) - C_g) / tau_G
                S[C_A, n + 1] = S[C_A, 0] + a_dt * (
                    math.tanh(S[F_R, n + 1] / gamma_A) - C_a) / tau_A
                H_wake = 1.0 if (f_W - theta_W) >= 0.0 else 0.0
                H_sleep = 1.0 if (theta_W - f_W) >= 0.0 else 0.0
                S[H, n + 1] = S[H, 0] + a_dt * ((H_max - h) / tau_hw * H_wake
                                                - h / tau_hs * H_sleep)

            # ---------------- cortex ---------------------------------
            Vp, Vi, Na = C[VP, n], C[VI, n], C[NA, n]
            s_ep, s_ei, s_gp, s_gi = C[S_EP, n], C[S_EI, n], C[S_GP, n], C[S_GI, n]
            x_ep, x_ei, x_gp, x_gi = C[X_EP, n], C[X_EI, n], C[X_GP, n], C[X_GI, n]
            cy, cx = C[CY, n], C[CX, n]
            g_KNa, sigma_p = C[G_KNA, n], C[SIGMA_P, n]
            ty = Th[TY, n]

            Qp = Qp_max / (1.0 + math.exp(-_C1 * (Vp - theta_p) / sigma_p))
            Qi = Qi_max / (1.0 + math.exp(-_C1 * (Vi - theta_i) / sigma_i))

            I_ep = g_AMPA * s_ep * (Vp - E_AMPA)
            I_gp = g_GABA * s_gp * (Vp - E_GABA)
            I_ei = g_AMPA * s_ei * (Vi - E_AMPA)
            I_gi = g_GABA * s_gi * (Vi - E_GABA)
            I_L_pp = g_L * (Vp - E_L_p)
            I_L_ii = g_L * (Vi - E_L_i)
            w_KNa = 0.37 / (1.0 + (38.7 / Na) ** 3.5)
            I_KNa = g_KNa * w_KNa * (Vp - E_K)
            Na3 = Na * Na * Na
            Na_pump = R_pump * (Na3 / (Na3 + 3375.0) - Na_eq3 / (Na_eq3 + 3375.0))

            nx_ep = ge2_c * (rand_c[0] + rand_c[1] / _SQRT3) * b
            nx_ei = ge2_c * (rand_c[2] + rand_c[3] / _SQRT3) * b

            C[VP, n + 1] = C[VP, 0] + a_dt * (-(I_L_pp + I_ep + I_gp) / tau_p
                                              - C_m * I_KNa)
            C[VI, n + 1] = C[VI, 0] + a_dt * (-(I_L_ii + I_ei + I_gi) / tau_i)
            C[NA, n + 1] = C[NA, 0] + a_dt * (alpha_Na * Qp - Na_pump) / tau_Na
            C[S_EP, n + 1] = C[S_EP, 0] + a_dt * x_ep
            C[S_EI, n + 1] = C[S_EI, 0] + a_dt * x_ei
            C[S_GP, n + 1] = C[S_GP, 0] + a_dt * x_gp
            C[S_GI, n + 1] = C[S_GI, 0] + a_dt * x_gi
            C[CY, n + 1] = C[CY, 0] + a_dt * cx
            C[X_EP, n + 1] = (C[X_EP, 0] + a_dt * (ge2_c * (N_pp * Qp + N_pt * ty - s_ep)
                                                   - 2.0 * gamma_e_c * x_ep) + nx_ep)
            C[X_EI, n + 1] = (C[X_EI, 0] + a_dt * (ge2_c * (N_ip * Qp + N_it * ty - s_ei)
                                                   - 2.0 * gamma_e_c * x_ei) + nx_ei)
            C[X_GP, n + 1] = C[X_GP, 0] + a_dt * (gg2_c * (N_pi * Qi - s_gp)
                                                  - 2.0 * gamma_g_c * x_gp)
            C[X_GI, n + 1] = C[X_GI, 0] + a_dt * (gg2_c * (N_ii * Qi - s_gi)
                                                  - 2.0 * gamma_g_c * x_gi)
            C[CX, n + 1] = C[CX, 0] + a_dt * (nu2_c * (Qp - cy) - 2.0 * nu_c * cx)
            # bifurcation parameters follow the ISO-modulated NE
            C[G_KNA, n + 1] = C[G_KNA, 0] + a_dt * (
                g_KNa_0 * (2.0 * C_g) * (1.0 - 0.6 * C_e_eff) * (1.0 - 0.95 * C_a)
                - g_KNa) / tau_g
            C[SIGMA_P, n + 1] = C[SIGMA_P, 0] + a_dt * (
                sigma_p_0 - (4.0 * C_e_eff + 2.0 * C_a) - sigma_p) / tau_s

            # ---------------- thalamus -------------------------------
            Vt, Vr, Ca = Th[VT, n], Th[VR, n], Th[CA, n]
            s_et, s_er, s_gt, s_gr = Th[S_ET, n], Th[S_ER, n], Th[S_GT, n], Th[S_GR, n]
            x_et, x_er, x_gt, x_gr = Th[X_ET, n], Th[X_ER, n], Th[X_GT, n], Th[X_GR, n]
            tyn, txn = Th[TY, n], Th[TX, n]
            h_T_t, h_T_r, m_h, m_h2 = Th[H_T_T, n], Th[H_T_R, n], Th[M_H, n], Th[M_H2, n]
            cyn = C[CY, n]

            Qt = Qt_max / (1.0 + math.exp(-_C1 * (Vt - theta_t) / sigma_t))
            Qr = Qr_max / (1.0 + math.exp(-_C1 * (Vr - theta_r) / sigma_r))

            I_et = g_AMPA * s_et * (Vt - E_AMPA)
            I_gt = g_GABA * s_gt * (Vt - E_GABA)
            I_er = g_AMPA * s_er * (Vr - E_AMPA)
            I_gr = g_GABA * s_gr * (Vr - E_GABA)

            m_inf_T_t = 1.0 / (1.0 + math.exp(-(Vt + 59.0) / 6.2))
            m_inf_T_r = 1.0 / (1.0 + math.exp(-(Vr + 52.0) / 7.4))
            h_inf_T_t = 1.0 / (1.0 + math.exp((Vt + 81.0) / 4.0))
            h_inf_T_r = 1.0 / (1.0 + math.exp((Vr + 80.0) / 5.0))
            tau_h_T_t = (30.8 + (211.4 + math.exp((Vt + 115.2) / 5.0))
                         / (1.0 + math.exp((Vt + 86.0) / 3.2))) / 3.7371928
            tau_h_T_r = (85.0 + 1.0 / (math.exp((Vr + 48.0) / 4.0)
                                       + math.exp(-(Vr + 407.0) / 50.0))) / 3.7371928

            m_inf_h = 1.0 / (1.0 + math.exp((Vt + 75.0) / 5.5))
            tau_m_h = 20.0 + 1000.0 / (math.exp((Vt + 71.5) / 14.2)
                                       + math.exp(-(Vt + 89.0) / 11.6))
            Ca4 = Ca * Ca * Ca * Ca
            P_h = k1 * Ca4 / (k1 * Ca4 + k2)

            I_L_tt = g_L * (Vt - E_L_t)
            I_L_rr = g_L * (Vr - E_L_r)
            I_LK_t = g_LK * (Vt - E_K)
            I_LK_r = g_LK * (Vr - E_K)
            I_T_t = g_T_t * m_inf_T_t * m_inf_T_t * h_T_t * (Vt - E_Ca)
            I_T_r = g_T_r * m_inf_T_r * m_inf_T_r * h_T_r * (Vr - E_Ca)
            I_h = g_h * (m_h + g_inc * m_h2) * (Vt - E_h)

            nx_et = ge2_t * (rand_t[0] + rand_t[1] / _SQRT3) * b

            Th[VT, n + 1] = Th[VT, 0] + a_dt * (-(I_L_tt + I_et + I_gt) / tau_t
                                                - C_m * (I_LK_t + I_T_t + I_h))
            Th[VR, n + 1] = Th[VR, 0] + a_dt * (-(I_L_rr + I_er + I_gr) / tau_r
                                                - C_m * (I_LK_r + I_T_r))
            Th[CA, n + 1] = Th[CA, 0] + a_dt * (alpha_Ca * I_T_t - (Ca - Ca_0) / tau_Ca)
            Th[H_T_T, n + 1] = Th[H_T_T, 0] + a_dt * (h_inf_T_t - h_T_t) / tau_h_T_t
            Th[H_T_R, n + 1] = Th[H_T_R, 0] + a_dt * (h_inf_T_r - h_T_r) / tau_h_T_r
            Th[M_H, n + 1] = Th[M_H, 0] + a_dt * ((m_inf_h * (1.0 - m_h2) - m_h) / tau_m_h
                                                  - k3 * P_h * m_h + k4 * m_h2)
            Th[M_H2, n + 1] = Th[M_H2, 0] + a_dt * (k3 * P_h * m_h - k4 * m_h2)
            Th[S_ET, n + 1] = Th[S_ET, 0] + a_dt * x_et
            Th[S_ER, n + 1] = Th[S_ER, 0] + a_dt * x_er
            Th[S_GT, n + 1] = Th[S_GT, 0] + a_dt * x_gt
            Th[S_GR, n + 1] = Th[S_GR, 0] + a_dt * x_gr
            Th[TY, n + 1] = Th[TY, 0] + a_dt * txn
            Th[X_ET, n + 1] = (Th[X_ET, 0] + a_dt * (ge2_t * (N_tp * cyn - s_et)
                                                     - 2.0 * gamma_e_t * x_et) + nx_et)
            Th[X_ER, n + 1] = Th[X_ER, 0] + a_dt * (ge2_t * (N_rt * Qt + N_rp * cyn - s_er)
                                                    - 2.0 * gamma_e_t * x_er)
            Th[X_GT, n + 1] = Th[X_GT, 0] + a_dt * (gg2_t * (N_tr * Qr - s_gt)
                                                    - 2.0 * gamma_g_t * x_gt)
            Th[X_GR, n + 1] = Th[X_GR, 0] + a_dt * (gg2_t * (N_rr * Qr - s_gr)
                                                    - 2.0 * gamma_g_t * x_gr)
            Th[TX, n + 1] = Th[TX, 0] + a_dt * (nu2_t * (Qt - tyn) - 2.0 * nu_t * txn)

        # ---- collapse the four stages, then re-draw noise ----------------
        for v in range(N_SR):
            S[v, 0] = (-3.0 * S[v, 0] + 2.0 * S[v, 1] + 4.0 * S[v, 2]
                       + 2.0 * S[v, 3] + S[v, 4]) / 6.0
        for v in range(N_CORTEX):
            C[v, 0] = (-3.0 * C[v, 0] + 2.0 * C[v, 1] + 4.0 * C[v, 2]
                       + 2.0 * C[v, 3] + C[v, 4]) / 6.0
        C[X_EP, 0] += ge2_c * (rand_c[0] - rand_c[1] * _SQRT3) / 4.0
        C[X_EI, 0] += ge2_c * (rand_c[2] - rand_c[3] * _SQRT3) / 4.0
        for v in range(N_THALAMUS):
            Th[v, 0] = (-3.0 * Th[v, 0] + 2.0 * Th[v, 1] + 4.0 * Th[v, 2]
                        + 2.0 * Th[v, 3] + Th[v, 4]) / 6.0
        Th[X_ET, 0] += ge2_t * (rand_t[0] - rand_t[1] * _SQRT3) / 4.0

        for i in range(2):
            rand_c[2 * i] = np.random.normal(0.0, dphi_c * DT)
            rand_c[2 * i + 1] = np.random.normal(0.0, DT)
        rand_t[0] = np.random.normal(0.0, dphi_t * DT)
        rand_t[1] = np.random.normal(0.0, DT)

        # ================= record ========================================
        if step >= onset_steps:
            if step % red == 0 and cnt_fast < n_fast:
                out_fast[0, cnt_fast] = C[VP, 0]
                out_fast[1, cnt_fast] = C[VI, 0]
                out_fast[2, cnt_fast] = Th[VT, 0]
                out_fast[3, cnt_fast] = Th[VR, 0]
                out_fast[4, cnt_fast] = Th[CA, 0]
                out_fast[5, cnt_fast] = Th[M_H, 0] + g_inc * Th[M_H2, 0]
                cnt_fast += 1
            if step % red_slow == 0 and cnt_slow < n_slow:
                gate0 = S[C_G, 0] if iso_gate != 0.0 else 1.0
                out_slow[0, cnt_slow] = S[F_W, 0]
                out_slow[1, cnt_slow] = S[F_N, 0]
                out_slow[2, cnt_slow] = S[F_R, 0]
                out_slow[3, cnt_slow] = S[C_E, 0]
                out_slow[4, cnt_slow] = S[C_G, 0]
                out_slow[5, cnt_slow] = S[C_A, 0]
                out_slow[6, cnt_slow] = S[H, 0]
                out_slow[7, cnt_slow] = C[G_KNA, 0]
                out_slow[8, cnt_slow] = C[SIGMA_P, 0]
                out_slow[9, cnt_slow] = S[C_E, 0] + A_iso * gate0 * iso
                out_slow[10, cnt_slow] = g_LK_base + B_iso * gate0 * (1.0 - iso)
                out_slow[11, cnt_slow] = iso
                out_slow[12, cnt_slow] = phase      # unwrapped, for the analysis
                cnt_slow += 1

    return cnt_fast, cnt_slow


_run_chunk_jit = (njit(cache=True, fastmath=True)(_run_chunk)
                  if HAVE_NUMBA else _run_chunk)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
PARAM_SR_DEFAULT = (5.8043, 0.0001, 0.0001, 0.5086)   # start of a 24 h day
NREM_IC = (0.05, 5.0, 0.0001, 1.0)                    # clamped in NREM

# --start-from presets: warm starts that drop the sleep switch straight into a
# vigilance state, so you do not have to simulate ~13.5 h of wake waiting for
# the homeostat to build.  These are initial conditions, NOT clamps - the
# switch still evolves, and only "wake" is a genuine fixed point.
#
# Persistence, measured by integrating the switch alone (see docstring):
#   wake  stable indefinitely (f_W 5.80 -> 5.53 over 1800 s)
#   nrem  stable for at least 1800 s; f_W drifts 1.50 -> 0.53, f_N pinned at 5
#   rem   holds ~540 s, then transitions to wake
#
# REM is transient by construction: acetylcholine from the REM population
# excites the wake population (g_AW = +1.0), so REM always terminates.  That
# is a property of the model, not a poorly chosen initial condition, and ~9
# min is a realistic REM bout length.
START_STATES = {
    #        f_W     f_N     f_R     h
    "wake": (5.8043, 0.0001, 0.0001, 0.5086),
    "nrem": (1.5,    5.0,    0.001,  0.85),
    "rem":  (0.3,    0.5,    5.0,    0.45),
}
G_H_DEFAULT = 0.049
CONNECTIVITY = (2.6, 2.6, 5.0, 10.0)


def simulate(T=3000, param_sr=NREM_IC, cortex="tc",
             g_LK=G_LK_ISO, g_h=G_H_DEFAULT, connectivity=CONNECTIVITY,
             f_iso=F_ISO, A_iso=A_ISO, B_iso=B_ISO,
             iso_jitter=ISO_JITTER, iso_freq_tau=ISO_FREQ_TAU,
             iso_amp_cv=ISO_AMP_CV, iso_amp_tau=ISO_AMP_TAU,
             iso_gate=True, iso_sr_feedback=False, clamp_sr=False,
             dphi_c=2.0, onset=ONSET, red=RED, red_slow=RED_SLOW,
             seed=None, use_jit=True, progress=True, progress_every=2.0,
             chunk_s=10, stream=None):
    """Run the model with an infraslow noradrenergic oscillation.

    Returns the usual traces plus NE (the ISO-modulated noradrenaline the
    rest of the model sees), g_LK (the modulated thalamic leak) and iso (the
    raw 0..1 oscillator), together with f_iso and onset so that the ISO phase
    of any sample can be reconstructed exactly.
    """
    stream = stream or sys.stderr
    fn = _run_chunk_jit if use_jit else _run_chunk

    if seed is not None:
        np.random.seed(int(seed))

    tau_Na, E_L_p, N_pp = CORTEX_TC if cortex == "tc" else CORTEX_SR
    N_tp, N_rp, N_pt, N_it = connectivity

    n_steps = (T + onset) * RES
    onset_steps = onset * RES
    n_fast = T * RES // red
    n_slow = max(1, T * RES // red_slow)

    S = np.zeros((N_SR, 5))
    S[F_W, 0], S[F_N, 0], S[F_R, 0], S[H, 0] = [float(v) for v in param_sr]
    S[C_E, 0] = math.tanh(S[F_W, 0] / gamma_E)
    S[C_G, 0] = math.tanh(S[F_N, 0] / gamma_G)
    S[C_A, 0] = math.tanh(S[F_R, 0] / gamma_A)

    C = np.zeros((N_CORTEX, 5))
    C[VP, 0], C[VI, 0], C[NA, 0] = E_L_p, E_L_i, Na_eq
    gate0 = S[C_G, 0] if iso_gate else 1.0
    ne0 = S[C_E, 0] + A_iso * gate0 * 0.5      # phase 0 -> iso = 0.5
    C[G_KNA, 0] = g_KNa_0 * (2.0 * S[C_G, 0]) * (1.0 - 0.6 * ne0) \
        * (1.0 - 0.95 * S[C_A, 0])
    C[SIGMA_P, 0] = sigma_p_0 - (4.0 * ne0 + 2.0 * S[C_A, 0])

    Th = np.zeros((N_THALAMUS, 5))
    Th[VT, 0], Th[VR, 0], Th[CA, 0] = E_L_t, E_L_r, Ca_0

    rand_c = np.zeros(4)
    rand_t = np.zeros(2)
    for i in range(2):
        rand_c[2 * i] = np.random.normal(0.0, dphi_c * DT)
        rand_c[2 * i + 1] = np.random.normal(0.0, DT)
    rand_t[0] = np.random.normal(0.0, dphi_t * DT)
    rand_t[1] = np.random.normal(0.0, DT)

    out_fast = np.zeros((len(FAST_KEYS), n_fast))
    out_slow = np.zeros((len(SLOW_KEYS), n_slow))
    iso_state = np.array([0.0, 1.0, 0.0])   # [unwrapped phase, amplitude, freq dev]

    chunk = max(RES, int(chunk_s) * RES)
    cnt_fast = cnt_slow = 0
    step0 = 0
    t_start = time.time()
    t_last = 0.0
    if progress:
        print(f"[nm_tc_sr_iso] {T} s = {n_steps:,} steps | ISO {f_iso} Hz "
              f"({1 / f_iso:.0f} s), A_iso={A_iso}, B_iso={B_iso}, "
              f"gate={'C_G' if iso_gate else 'off'}, sr_fb={iso_sr_feedback}, "
              f"clamp_sr={clamp_sr}, jitter={iso_jitter}, amp_cv={iso_amp_cv}",
              file=stream, flush=True)

    if progress and HAVE_NUMBA:
        print("  compiling (first chunk includes JIT, ~30-60 s if not cached) ...",
              file=stream, flush=True)

    while step0 < n_steps:
        n_chunk = min(chunk, n_steps - step0)
        cnt_fast, cnt_slow = fn(
            n_chunk, step0, C, Th, S, rand_c, rand_t,
            out_fast, out_slow, cnt_fast, cnt_slow,
            onset_steps, red, red_slow,
            tau_Na, E_L_p, N_pp, dphi_c,
            g_LK, g_h, N_tp, N_rp, N_pt, N_it,
            f_iso, A_iso, B_iso,
            1.0 if iso_gate else 0.0, 1.0 if iso_sr_feedback else 0.0,
            1.0 if clamp_sr else 0.0,
            iso_state, iso_jitter, iso_freq_tau, iso_amp_cv, iso_amp_tau)
        step0 += n_chunk

        el = time.time() - t_start
        done = step0 >= n_steps
        if progress and (el - t_last >= progress_every or done):
            t_last = el
            rate = step0 / el if el > 0 else 0.0
            eta = (n_steps - step0) / rate if rate > 0 else 0.0
            print(f"  [{100 * step0 / n_steps:5.1f}%] sim {_hms(step0 / RES)}"
                  f"/{_hms(n_steps / RES)}  {rate / 1e3:6.1f}k steps/s  "
                  f"elapsed {_hms(el)}  ETA {_hms(eta)}  "
                  f"| NE={S[C_E, 0] + A_iso * (S[C_G, 0] if iso_gate else 1.0) * 0.5:.3f} "
                  f"g_KNa={C[G_KNA, 0]:.2f} sig_p={C[SIGMA_P, 0]:.2f}",
                  file=stream, flush=True)

    fs, fs_slow = RES / red, RES / red_slow
    out = {k: out_fast[i, :cnt_fast] for i, k in enumerate(FAST_KEYS)}
    out.update({k: out_slow[i, :cnt_slow] for i, k in enumerate(SLOW_KEYS)})
    out["t"] = np.arange(cnt_fast) / fs
    out["t_slow"] = np.arange(cnt_slow) / fs_slow
    out["fs"] = fs
    out["fs_slow"] = fs_slow
    out["f_iso"] = f_iso
    out["onset"] = onset
    # provenance: everything needed to interpret or reproduce this run, so a
    # saved .npz can be replotted and understood without the command line
    out["A_iso"] = A_iso
    out["B_iso"] = B_iso
    out["iso_jitter"] = iso_jitter
    out["iso_amp_cv"] = iso_amp_cv
    out["iso_gate"] = float(iso_gate)
    out["iso_sr_feedback"] = float(iso_sr_feedback)
    out["clamp_sr"] = float(clamp_sr)
    out["g_LK_base"] = g_LK
    out["g_h"] = g_h
    out["param_sr"] = np.asarray(param_sr, dtype=np.float64)
    out["cortex"] = cortex
    if progress:
        print("  high-pass filtering Vp -> EEG ...", file=stream, flush=True)
    out["EEG"] = eeg_from_vp(out["Vp"], fs)
    return out


def stage_label(f_W, f_N, f_R):
    """Crude vigilance-state label from the regulatory populations."""
    if f_W >= theta_W:
        return "WAKE"
    return "REM " if f_R > f_N else "NREM"


def eeg_from_vp(Vp, fs, hp=EEG_HP, order=2):
    """AC-couple Vp the way an EEG amplifier does.  See EEG_HP."""
    from scipy.signal import butter, filtfilt
    if Vp.size < 4 * (order + 1) * 3:
        return Vp - Vp.mean()
    b, a = butter(order, hp / (fs / 2.0), btype="high")
    return filtfilt(b, a, Vp)


def load(path):
    """Load a saved run back into the dict shape simulate() returns.

    Lets a long unattended run be replotted or re-analysed later:
        out  = load("day.npz")
        prof = iso_analysis(out)
        _plot(out, prof, show=True)
    """
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    for k in ("fs", "fs_slow", "f_iso", "onset", "A_iso", "B_iso",
              "iso_jitter", "iso_amp_cv", "iso_gate", "iso_sr_feedback",
              "clamp_sr", "g_LK_base", "g_h"):
        if k in out:
            out[k] = float(out[k])
    if "cortex" in out:
        out["cortex"] = str(out["cortex"])
    if "EEG" not in out:                   # runs saved before EEG existed
        out["EEG"] = eeg_from_vp(out["Vp"], out["fs"])
    return out


def _hms(seconds):
    s = int(max(0.0, seconds))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


# ---------------------------------------------------------------------------
# ISO phase analysis
# ---------------------------------------------------------------------------
def iso_phase(out, which="fast"):
    """ISO phase in [0, 2pi) for every sample of the fast or slow traces.

    With a noisy drive the phase is integrated state, not a function of time,
    so it is read back from the recorded unwrapped phase (interpolated onto
    the fast time base when needed).  Falls back to the analytic phase for
    results saved before the drive became stochastic.
    """
    t = out["t"] if which == "fast" else out["t_slow"]
    phi = out.get("iso_phi") if hasattr(out, "get") else None
    if phi is None or np.size(phi) == 0:
        return (2.0 * np.pi * out["f_iso"] * (t + out["onset"])) % (2.0 * np.pi)
    if which != "fast":
        return phi % (2.0 * np.pi)
    return np.interp(t, out["t_slow"], phi) % (2.0 * np.pi)


def _bandpower_env(x, fs, lo, hi, order=4):
    """Envelope of x in [lo, hi] Hz, via zero-phase filtering + Hilbert."""
    from scipy.signal import butter, filtfilt, hilbert
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return np.abs(hilbert(filtfilt(b, a, x)))


def iso_analysis(out, n_bins=18, spindle_band=(11.0, 16.0), so_band=(0.5, 4.0),
                 spindle_sd=1.5, min_spindle_s=0.4):
    """Quantify how spindles and arousals distribute over the ISO cycle.

    Returns a dict with per-bin profiles (bin centres in radians) and, for
    detected spindle events, the circular mean phase and resultant length.

    Phase convention: NE = 0.5*(1+sin(phase)), so
        phase 0     -> NE mid, rising
        phase pi/2  -> NE peak
        phase pi    -> NE mid, falling
        phase 3pi/2 -> NE trough
    Ascending half is phase in (3pi/2, 2pi) u [0, pi/2); descending half is
    (pi/2, 3pi/2).
    """
    fs = out["fs"]
    ph = iso_phase(out, "fast")

    sp_env = _bandpower_env(out["Vt"], fs, *spindle_band)
    so_env = _bandpower_env(out["Vp"], fs, *so_band)
    vp = out["Vp"]

    edges = np.linspace(0.0, 2 * np.pi, n_bins + 1)
    idx = np.clip(np.digitize(ph, edges) - 1, 0, n_bins - 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    def binned(x):
        s = np.bincount(idx, weights=x, minlength=n_bins)
        n = np.bincount(idx, minlength=n_bins)
        return s / np.maximum(n, 1)

    prof = {"phase": centres, "spindle_env": binned(sp_env),
            "so_env": binned(so_env), "Vp_mean": binned(vp),
            "NE": 0.5 * (1 + np.sin(centres))}

    # discrete spindle events: envelope above mean + k*sd, held long enough
    thr = sp_env.mean() + spindle_sd * sp_env.std()
    above = sp_env > thr
    d = np.diff(above.astype(np.int8))
    starts = np.flatnonzero(d == 1) + 1
    stops = np.flatnonzero(d == -1) + 1
    if above[0]:
        starts = np.r_[0, starts]
    if above[-1]:
        stops = np.r_[stops, above.size]
    keep = (stops - starts) >= int(min_spindle_s * fs)
    starts, stops = starts[keep], stops[keep]
    peaks = np.array([s + np.argmax(sp_env[s:e]) for s, e in zip(starts, stops)],
                     dtype=int)
    ev_ph = ph[peaks] if peaks.size else np.array([])

    if ev_ph.size:
        z = np.exp(1j * ev_ph).mean()
        pref, R = np.angle(z) % (2 * np.pi), np.abs(z)
    else:
        pref, R = float("nan"), 0.0

    asc = (ev_ph < np.pi / 2) | (ev_ph > 3 * np.pi / 2)
    prof.update({
        "n_spindles": int(peaks.size),
        "spindle_phase": ev_ph,
        "spindle_pref_phase": pref,
        "spindle_R": R,
        "spindle_frac_descending": float(1.0 - asc.mean()) if ev_ph.size else float("nan"),
        "spindle_rate_hz": peaks.size / (out["t"][-1] if out["t"].size else 1.0),
    })

    # Continuous arousal proxy: loss of slow waves.  Excluding Vp keeps the
    # index to a single interpretable quantity, but note that Vp is NOT the
    # problematic term - measured over 1500 s at A_iso = 0.4 and 0.9, mean
    # depolarization peaks at the NE peak (70 deg) in both cases, i.e. it
    # always points the right way.
    #
    # The term that misbehaves is this one, and only at low A_iso.  There the
    # slow-wave envelope is dominated by the thalamic limb rather than the
    # cortical one: at the NE trough g_LK is maximal, the thalamus bursts
    # hardest, and those bursts break up cortical down states and suppress
    # slow-wave power - so SO power bottoms out at the trough, which inverts
    # the index.  Only once A_iso is large enough (~0.9, giving sigma_p ~3.9
    # at the NE peak) does the cortical g_KNa/sigma_p modulation dominate and
    # the arousal index peak at the NE peak as it should.  If you lower
    # A_iso, expect this measure to invert.
    ar = -(so_env - so_env.mean()) / so_env.std()
    prof["arousal_index"] = binned(ar)
    k = int(np.argmax(prof["arousal_index"]))
    prof["arousal_peak_phase"] = centres[k]
    k = int(np.argmax(prof["spindle_env"]))
    prof["spindle_env_peak_phase"] = centres[k]
    # kept for diagnosis: mean depolarization, which carries the thalamic
    # confound described above
    k = int(np.argmax(prof["Vp_mean"]))
    prof["Vp_peak_phase"] = centres[k]
    return prof


def _deg(rad):
    return float(np.degrees(rad) % 360.0)


def _phase_name(rad):
    d = _deg(rad)
    if 45 <= d < 135:
        return "NE peak"
    if 135 <= d < 225:
        return "descending (mid)"
    if 225 <= d < 315:
        return "NE trough"
    return "ascending (mid)"


def print_analysis(prof, stream=None):
    stream = stream or sys.stdout
    p = lambda *a: print(*a, file=stream)
    p("\nISO phase analysis")
    p("  phase convention: 90 deg = NE peak, 270 deg = NE trough")
    p(f"  spindles detected            : {prof['n_spindles']} "
      f"({prof['spindle_rate_hz'] * 60:.1f}/min)")
    if prof["n_spindles"]:
        p(f"  spindle preferred phase      : {_deg(prof['spindle_pref_phase']):6.1f} deg"
          f"   ({_phase_name(prof['spindle_pref_phase'])})")
        p(f"  spindle phase concentration R: {prof['spindle_R']:.3f}")
        p(f"  spindles on descending half  : "
          f"{100 * prof['spindle_frac_descending']:.1f}%   (chance = 50%)")
    p(f"  spindle power peaks at       : {_deg(prof['spindle_env_peak_phase']):6.1f} deg"
      f"   ({_phase_name(prof['spindle_env_peak_phase'])})")
    p(f"  arousal index peaks at       : {_deg(prof['arousal_peak_phase']):6.1f} deg"
      f"   ({_phase_name(prof['arousal_peak_phase'])})   [-slow-wave power]")
    p(f"  mean depolarization peaks at : {_deg(prof['Vp_peak_phase']):6.1f} deg"
      f"   ({_phase_name(prof['Vp_peak_phase'])})   [carries thalamic drive]")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _decimate(t, y, n_max=20000, envelope=True):
    """Reduce a trace for plotting.

    envelope=True keeps the min and max of each bin instead of one sample, so
    oscillatory traces keep their true visual extent.  Plain subsampling
    aliases them - a 13 Hz spindle thinned 400x can disappear entirely.
    """
    n = y.size
    if n <= n_max:
        return t, y
    if not envelope:
        s = int(np.ceil(n / n_max))
        return t[::s], y[::s]
    nb = max(1, n_max // 2)
    s = max(1, n // nb)
    nb = n // s
    yy = y[:nb * s].reshape(nb, s)
    to = np.repeat(t[:nb * s:s], 2)
    yo = np.empty(2 * nb)
    yo[0::2] = yy.min(1)
    yo[1::2] = yy.max(1)
    return to, yo


def _attach_adaptive(ax, line, t, y, n_max=20000):
    """Re-decimate on zoom so an interactive window stays responsive.

    The full arrays stay in memory but only what is on screen is drawn, so
    zooming into a long record recovers real detail without ever handing
    matplotlib millions of vertices.  Narrowing xlim alone would not do this:
    matplotlib still transforms every vertex on each redraw.
    """
    def on_xlim(a):
        x0, x1 = a.get_xlim()
        i0, i1 = np.searchsorted(t, (x0, x1))
        i0, i1 = max(0, i0 - 1), min(t.size, i1 + 1)
        if i1 - i0 < 2:
            return
        line.set_data(*_decimate(t[i0:i1], y[i0:i1], n_max))
        a.figure.canvas.draw_idle()
    ax.callbacks.connect("xlim_changed", on_xlim)


def _plot(out, prof, path=None, show=False):
    """Draw the summary figure.  Saves to `path` and/or shows it interactively.

    The Agg backend is only forced when we are strictly writing a file - it is
    non-interactive, so forcing it unconditionally would make plt.show() a
    silent no-op.  With show=True we leave matplotlib to pick its backend;
    set MPLBACKEND to override (TkAgg is the dependable one here, the default
    GTK4Agg emits libGL errors on this machine).
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 13))
    gs = fig.add_gridspec(6, 3, height_ratios=[1, 1, 1, 1, 1, 1.3], hspace=0.35,
                          wspace=0.3)

    # --- the five time-series panels, all sharing the time axis -----------
    tf = out["t"]
    ax_eeg = fig.add_subplot(gs[0, :])
    ln, = ax_eeg.plot(*_decimate(tf, out["EEG"]), lw=0.4, color="0.2")
    ax_eeg.set_ylabel("EEG (mV)\nhigh-passed")
    ax_eeg.set_title("Infraslow noradrenergic modulation of NREM sleep")
    if show:
        _attach_adaptive(ax_eeg, ln, tf, out["EEG"])

    ax_vp = fig.add_subplot(gs[1, :], sharex=ax_eeg)
    ln, = ax_vp.plot(*_decimate(tf, out["Vp"]), lw=0.4)
    ax_vp.set_ylabel("$V_p$ (mV)\ncortex")
    if show:
        _attach_adaptive(ax_vp, ln, tf, out["Vp"])

    ax_vt = fig.add_subplot(gs[2, :], sharex=ax_eeg)
    ln, = ax_vt.plot(*_decimate(tf, out["Vt"]), lw=0.4, color="tab:orange")
    ax_vt.set_ylabel("$V_t$ (mV)\nrelay")
    if show:
        _attach_adaptive(ax_vt, ln, tf, out["Vt"])

    # slow traces need decimating too, but plainly - they are smooth, so an
    # envelope would only double the vertex count for no visual gain
    ts = out["t_slow"]
    def sl(key):
        return _decimate(ts, out[key], envelope=False)

    ax_ne = fig.add_subplot(gs[3, :], sharex=ax_eeg)
    ax_ne.plot(*sl("NE"), lw=1.0, color="tab:red", label="NE (ISO)")
    ax_ne.plot(*sl("C_E"), lw=1.0, color="tab:red", ls=":", label="$C_E$ (tonic)")
    ax_ne.set_ylabel("noradrenaline\n(aU)")
    ax_ne.legend(loc="upper right", fontsize=8, ncol=2)

    ax_k = fig.add_subplot(gs[4, :], sharex=ax_eeg)
    ax_k.plot(*sl("g_KNa"), lw=1.0, label="$g_{KNa}$")
    ax_k.plot(*sl("sigma_p"), lw=1.0, label="$\\sigma_p$")
    ax_k2 = ax_k.twinx()
    ax_k2.plot(*sl("g_LK"), lw=1.0, color="tab:green", label="$g_{LK}$")
    ax_k2.set_ylabel("$g_{LK}$", color="tab:green")
    ax_k.set_ylabel("cortical\nknobs")
    ax_k.set_xlabel("time (s)")
    ax_k.legend(loc="upper right", fontsize=8, ncol=2)

    time_axes = (ax_eeg, ax_vp, ax_vt, ax_ne, ax_k)
    for a in time_axes[:-1]:        # only the bottom panel keeps tick labels
        a.tick_params(labelbottom=False)
    ax_eeg.margins(x=0)             # shared, so this sets all five

    # --- the two phase-resolved panels, sharing the phase axis ------------
    deg = np.degrees(prof["phase"])
    ne = prof["NE"]

    def phase_panel(col, y, label, color, share=None):
        a = fig.add_subplot(gs[5, col], sharex=share)
        a.plot(deg, y, "o-", lw=1.4, ms=3, color=color)
        a.set_xlabel("ISO phase (deg)")
        a.set_ylabel(label)
        a.set_xticks([0, 90, 180, 270, 360])
        a.set_xlim(0, 360)
        a.axvspan(90, 270, color="0.9", zorder=0)
        b = a.twinx()
        b.plot(deg, ne, lw=0.8, color="tab:red", alpha=0.5)
        b.set_yticks([])
        return a

    ax_sp = phase_panel(0, prof["spindle_env"], "spindle envelope\n11-16 Hz ($V_t$)",
                        "tab:orange")
    phase_panel(1, prof["arousal_index"], "arousal index\n(-slow-wave power)",
                "tab:blue", share=ax_sp)
    a = fig.add_subplot(gs[5, 2], projection="polar")
    if prof["n_spindles"]:
        a.hist(prof["spindle_phase"], bins=18, color="tab:orange", alpha=0.8)
        a.arrow(prof["spindle_pref_phase"], 0, 0, prof["spindle_R"] * a.get_ylim()[1],
                color="k", lw=2, head_width=0.08)
    a.set_theta_zero_location("E")
    a.set_rlabel_position(135)
    a.tick_params(labelsize=7)
    # label below rather than above: a title here collides with the panel
    # in the row above, because a polar axes fills its whole cell
    a.set_xlabel("spindle events by ISO phase", fontsize=9, labelpad=10)

    if path:
        fig.savefig(path, dpi=140, bbox_inches="tight")
        print(f"wrote {path}")
    if show:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Thalamocortical model with an infraslow NE oscillation.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--seconds", type=int, help="recorded seconds")
    g.add_argument("--hours", type=float, help="recorded hours")
    p.add_argument("--start-from", type=str.lower, choices=sorted(START_STATES),
                   metavar="STATE",
                   help="warm-start the sleep switch in wake/nrem/rem "
                        "(case-insensitive); free-running, not clamped")
    p.add_argument("--clamp-sr-nrem", action="store_true",
                   help="freeze the sleep switch in stationary deep NREM - the "
                        "configuration used for the ISO phase analysis "
                        "(a different, deeper state than --start-from nrem)")
    p.add_argument("--param-sr", type=float, nargs=4, metavar=("F_W", "F_N", "F_R", "H"),
                   help="explicit sleep-switch initial condition; overrides the above")
    p.add_argument("--f-iso", type=float, default=F_ISO, help="ISO frequency, Hz")
    p.add_argument("--a-iso", type=float, default=A_ISO, help="NE modulation depth")
    p.add_argument("--b-iso", type=float, default=B_ISO, help="g_LK modulation depth")
    p.add_argument("--iso-jitter", type=float, default=ISO_JITTER,
                   help=f"cycle-period CV of the ISO drive (default {ISO_JITTER}; "
                        "0 = deterministic sinusoid)")
    p.add_argument("--iso-amp-cv", type=float, default=ISO_AMP_CV,
                   help=f"amplitude CV of the ISO drive (default {ISO_AMP_CV}; 0 = fixed)")
    p.add_argument("--no-iso-gate", action="store_true",
                   help="do not gate the ISO by the NREM marker C_G")
    p.add_argument("--iso-sr-feedback", action="store_true",
                   help="let the modulated NE re-enter the sleep switch")
    p.add_argument("--cortex", choices=["tc", "sr"], default="tc")
    p.add_argument("--g-lk", type=float, default=G_LK_ISO)
    p.add_argument("--g-h", type=float, default=G_H_DEFAULT)
    p.add_argument("--red", type=int, default=RED)
    p.add_argument("--red-slow", type=int, default=RED_SLOW)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-analyze", dest="analyze", action="store_false",
                   help="skip the ISO phase analysis (it runs by default)")
    p.add_argument("--load", metavar="NPZ",
                   help="skip the simulation and replot/re-analyse a saved run")
    p.add_argument("--plot", metavar="PNG")
    p.add_argument("--show", action="store_true",
                   help="display the figure interactively (plt.show)")
    p.add_argument("--save", metavar="NPZ")
    p.add_argument("--quiet", action="store_true",
                   help="suppress progress output")
    p.add_argument("--progress-every", type=float, default=2.0, metavar="SEC")
    p.add_argument("--no-jit", action="store_true")
    args = p.parse_args()

    if args.hours is not None:
        T = int(round(args.hours * 3600))
    elif args.seconds is not None:
        T = args.seconds
    else:
        T = 3000

    # Precedence: --param-sr > --start-from > --clamp-sr-nrem > wake.
    # The default is always wake at the start of a 24 h day, whatever the run
    # length - a duration-dependent default was too surprising, since a short
    # run would silently begin in clamped NREM.
    #
    # Clamping and the deep-NREM state are one option because they are only
    # useful together: that state is not a fixed point of the switch (the REM
    # population escapes on its 60 s time constant), so it has to be frozen to
    # stay put.  Everything else runs free.  To clamp some other state, call
    # simulate(param_sr=..., clamp_sr=True) directly.
    clamp = False
    if args.param_sr is not None:
        param_sr = tuple(args.param_sr)
    elif args.start_from is not None:
        param_sr = START_STATES[args.start_from]
    elif args.clamp_sr_nrem:
        param_sr = NREM_IC
        clamp = True
    else:
        param_sr = PARAM_SR_DEFAULT

    if not args.quiet:
        lab = stage_label(param_sr[0], param_sr[1], param_sr[2])
        print(f"[nm_tc_sr_iso] start state {lab} "
              f"f_W={param_sr[0]:g} f_N={param_sr[1]:g} f_R={param_sr[2]:g} "
              f"h={param_sr[3]:g}, clamp_sr={clamp}", file=sys.stderr, flush=True)
        if lab == "WAKE" and T < 12 * 3600:
            print("[nm_tc_sr_iso] note: the ISO is gated by C_G, so it stays "
                  "off during wake; sleep onset is ~13.5 h in. Use "
                  "--start-from nrem or --clamp-sr-nrem for a short ISO run.",
                  file=sys.stderr)

    if not HAVE_NUMBA:
        print("numba not found - running pure Python (very slow)", file=sys.stderr)

    if args.load:
        print(f"loading {args.load} "
              f"({os.path.getsize(args.load) / 1e6:.1f} MB) ...",
              file=sys.stderr, flush=True)
        out = load(args.load)
        print(f"loaded {args.load}: {out['Vp'].size:,} fast samples @ "
              f"{out['fs']:g} Hz, {out['f_W'].size:,} slow @ {out['fs_slow']:g} Hz "
              f"({out['t'][-1] / 3600:.2f} h)")
        print(f"  A_iso={out.get('A_iso', float('nan')):.2f} "
              f"B_iso={out.get('B_iso', float('nan')):.4f} "
              f"jitter={out.get('iso_jitter', float('nan')):.2f} "
              f"amp_cv={out.get('iso_amp_cv', float('nan')):.2f} "
              f"clamp_sr={bool(out.get('clamp_sr', 0))}")
        print("running ISO phase analysis ...", file=sys.stderr, flush=True)
        prof = iso_analysis(out)
        print_analysis(prof)
        if args.plot or args.show:
            _plot(out, prof, args.plot, show=args.show)
        return

    out = simulate(T=T, param_sr=param_sr, cortex=args.cortex,
                   g_LK=args.g_lk, g_h=args.g_h,
                   f_iso=args.f_iso, A_iso=args.a_iso, B_iso=args.b_iso,
                   iso_jitter=args.iso_jitter, iso_amp_cv=args.iso_amp_cv,
                   iso_gate=not args.no_iso_gate,
                   iso_sr_feedback=args.iso_sr_feedback, clamp_sr=clamp,
                   red=args.red, red_slow=args.red_slow, seed=args.seed,
                   use_jit=not args.no_jit, progress=not args.quiet,
                   progress_every=args.progress_every)

    print(f"  Vp   mean {out['Vp'].mean():7.2f} mV  range "
          f"[{out['Vp'].min():.1f}, {out['Vp'].max():.1f}]")
    print(f"  Vt   mean {out['Vt'].mean():7.2f} mV  range "
          f"[{out['Vt'].min():.1f}, {out['Vt'].max():.1f}]")
    print(f"  NE   range [{out['NE'].min():.3f}, {out['NE'].max():.3f}]")
    print(f"  g_LK range [{out['g_LK'].min():.4f}, {out['g_LK'].max():.4f}]")
    print(f"  g_KNa range [{out['g_KNa'].min():.3f}, {out['g_KNa'].max():.3f}]  "
          f"sigma_p range [{out['sigma_p'].min():.3f}, {out['sigma_p'].max():.3f}]")

    # the figure needs `prof`, so still compute it under --no-analyze --plot;
    # --no-analyze only suppresses the printed statistics
    prof = None
    if args.analyze or args.plot or args.show:
        if not args.quiet:
            print(f"running ISO phase analysis on {out['Vp'].size:,} samples ...",
                  file=sys.stderr, flush=True)
        prof = iso_analysis(out)
        if args.analyze:
            print_analysis(prof)

    if args.save:
        np.savez_compressed(args.save, **{k: v for k, v in out.items()})
        print(f"wrote {args.save}")
    if args.plot or args.show:
        print("rendering figure ...", file=sys.stderr, flush=True)
        _plot(out, prof, args.plot, show=args.show)


# ---------------------------------------------------------------------------
# Convenience wrapper for figure3_4.py
# ---------------------------------------------------------------------------
_STAGE_SR4 = {
    "Wake": (START_STATES["wake"], False),
    "N2":   (START_STATES["nrem"],  True),
    "N3":   (NREM_IC,               True),
    "REM":  (START_STATES["rem"],   False),
}


def run_stage_4(stage_name, T_s=200, seed=7):
    """Return (Vp, Vt) arrays at FS=100 Hz for the requested sleep stage.

    Uses simulate() with clamp_sr=True for NREM stages so the ISO alone
    gates K-complex generation.  Wake runs free (stable fixed point);
    REM runs free for T_s (holds ~540 s before transitioning to wake).
    """
    param_sr, clamp = _STAGE_SR4[stage_name]
    out = simulate(
        T=T_s, param_sr=param_sr, clamp_sr=clamp,
        seed=seed, progress=True, use_jit=HAVE_NUMBA,
    )
    return out["Vp"], out["Vt"]


if __name__ == "__main__":
    main()