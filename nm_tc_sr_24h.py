"""
nm_tc_sr_24h.py  —  24-hour EEG simulation using NM_TC_SR

Strategy (avoids running 864 M Python integration steps):
  1. Load the 4 pre-computed 30-s stage waveforms (py_out_*.csv).
     If not found, run_stage() is called to generate them (~100 s).
  2. Run the Sleep-Regulation ODEs for 24 h with scipy (seconds).
  3. Tile the neural waveforms onto a realistic 24-h sleep schedule,
     with smooth 2-s cross-fades at every stage transition.
  4. Save signals as numpy arrays and produce two figures.

Outputs (saved next to this script):
  eeg_24h.npy      Cortical Vp [mV]  at 100 Hz  (8 640 000 samples)
  vt_24h.npy       Thalamic Vt [mV]  at 100 Hz
  stage_24h.npy    Stage label (0=W 1=N2 2=N3 3=R) at 100 Hz
  t_24h.npy        Time vector [hours] at 100 Hz
  sr_24h.npy       SR variables [C_E C_G C_A h f_W f_N f_R] at 1 Hz (86 400×7)
  fig_24h_overview.png
  fig_24h_epochs.png
"""

import os, csv, math, time
import numpy as np
import matplotlib
matplotlib.use('Agg')          # no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.integrate import solve_ivp

HERE  = os.path.dirname(os.path.abspath(__file__))
FS    = 100          # output sample rate [Hz]  (red=100 in the model)
T24   = 24 * 3600   # 24 h in seconds
N24   = T24 * FS    # total samples

STAGE_INT = {'Wake': 0, 'N2': 1, 'N3': 2, 'REM': 3}
STAGE_COL = {0: '#888888', 1: '#4488CC', 2: '#112277', 3: '#CC3333'}
STAGE_LAB = {0: 'Wake', 1: 'N2', 2: 'N3', 3: 'REM'}

# ─── 1. Sleep schedule  ───────────────────────────────────────────────────────
# Starting at 07:00 (wake-up time = t=0).
# Each entry: (stage_name, duration_s).
# Total must equal 86 400 s.
SCHEDULE = [
    ('Wake', 57600),   # 16 h   07:00 → 23:00
    ('N2',    3600),   #  1 h   23:00 → 00:00  sleep onset
    ('N3',    6300),   # 1¾ h  00:00 → 01:45  deep NREM
    ('N2',    1800),   # 30 m   01:45 → 02:15
    ('REM',   3600),   #  1 h   02:15 → 03:15  REM cycle 1
    ('N3',    3600),   #  1 h   03:15 → 04:15
    ('N2',    1800),   # 30 m   04:15 → 04:45
    ('REM',   4200),   # 70 m   04:45 → 05:55  REM cycle 2
    ('N2',    1800),   # 30 m   05:55 → 06:25
    ('REM',   2100),   # 35 m   06:25 → 07:00  REM cycle 3
]
assert sum(d for _, d in SCHEDULE) == T24, "Schedule must sum to 86 400 s"

# Neurotransmitter levels that match each stage (from test output)
STAGE_NT = {
    'Wake': dict(C_E=0.833, C_G=0.000, C_A=0.000, g_KNa=0.000, sigma_p=3.33),
    'N2':   dict(C_E=0.099, C_G=0.766, C_A=0.001, g_KNa=1.915, sigma_p=4.80),
    'N3':   dict(C_E=0.010, C_G=0.848, C_A=0.024, g_KNa=2.191, sigma_p=4.96),
    'REM':  dict(C_E=0.073, C_G=0.052, C_A=0.974, g_KNa=0.010, sigma_p=3.88),
}

# SR initial conditions per stage
STAGE_SR0 = {
    'Wake': (6.00, 0.001, 0.001, 0.30),
    'N2':   (0.50, 4.000, 0.001, 1.00),
    'N3':   (0.05, 5.000, 0.001, 1.00),
    'REM':  (0.30, 0.001, 4.000, 0.50),
}


# ─── 2. Load or generate per-stage waveforms ──────────────────────────────────

def _load_csv(path):
    """Return (Vp, Vt) numpy arrays from a py_out_*.csv file."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    Vp = np.array([float(r['Vp']) for r in rows])
    Vt = np.array([float(r['Vt']) for r in rows])
    return Vp, Vt

def _csv_name(stage):
    tag = stage.replace(' ', '_').replace('(', '').replace(')', '')
    return os.path.join(HERE, f'py_out_{tag}.csv')

def load_waveforms():
    """Return dict stage → (Vp array, Vt array) at FS Hz."""
    from nm_tc_sr import run_stage
    waves = {}
    for stage in ('Wake', 'N2', 'N3', 'REM'):
        path = _csv_name(stage)
        if os.path.exists(path):
            Vp, Vt = _load_csv(path)
        else:
            print(f'  pre-computing {stage} waveform (~25 s)…')
            fW, fN, fR, h0 = STAGE_SR0[stage]
            # canonical CSV names used by nm_tc_sr.py
            if stage == 'N2':
                csv_stage = 'N2_lt_NREM'
            elif stage == 'N3':
                csv_stage = 'N3_dp_NREM'
            else:
                csv_stage = stage
            path2 = os.path.join(HERE, f'py_out_{csv_stage}.csv')
            if os.path.exists(path2):
                Vp, Vt = _load_csv(path2)
            else:
                R  = run_stage(fW, fN, fR, h0, T_s=30)
                Vp = np.array(R['Vp']);  Vt = np.array(R['Vt'])
        waves[stage] = (Vp, Vt)
        print(f'  {stage}: {len(Vp)} samples  '
              f'Vp [{Vp.min():.1f}, {Vp.max():.1f}] mV  '
              f'Vt [{Vt.min():.1f}, {Vt.max():.1f}] mV')
    return waves


# ─── 3. Tile waveforms with cross-fades ───────────────────────────────────────

FADE = int(2 * FS)   # 2-second crossfade

def _tile(wave, n_samples):
    """Tile `wave` to exactly n_samples with looping cross-fades."""
    w = len(wave)
    out = np.empty(n_samples, dtype=np.float64)
    pos = 0
    first = True
    while pos < n_samples:
        take = min(w, n_samples - pos)
        if first or FADE == 0 or pos == 0:
            out[pos:pos+take] = wave[:take]
            first = False
        else:
            fade_len = min(FADE, take, pos)
            α = np.linspace(0, 1, fade_len)
            # blend old tail into new head
            out[pos:pos+fade_len] = (1-α)*out[pos:pos+fade_len] + α*wave[:fade_len]
            out[pos+fade_len:pos+take] = wave[fade_len:take]
        pos += take - FADE if (take == w and pos + take < n_samples) else take
        if pos < 0:
            break
    return out

def build_eeg_24h(waves):
    """
    Concatenate stage waveforms according to SCHEDULE.
    Returns (Vp_24h, Vt_24h, stage_arr) each of length N24.
    """
    Vp_24h    = np.empty(N24, dtype=np.float64)
    Vt_24h    = np.empty(N24, dtype=np.float64)
    stage_arr = np.empty(N24, dtype=np.int8)

    cursor = 0
    prev_vp = None
    prev_vt = None

    for stage_name, dur_s in SCHEDULE:
        n = dur_s * FS
        s_int = STAGE_INT[stage_name]
        wVp, wVt = waves[stage_name]

        # de-mean each waveform (remove DC; EEG is AC)
        seg_vp = _tile(wVp - wVp.mean(), n)
        seg_vt = _tile(wVt - wVt.mean(), n)

        # smooth join with previous segment
        if prev_vp is not None and FADE > 0:
            fade_len = min(FADE, n)
            α = np.linspace(0, 1, fade_len)
            seg_vp[:fade_len] = (1-α)*prev_vp[-fade_len:] + α*seg_vp[:fade_len]
            seg_vt[:fade_len] = (1-α)*prev_vt[-fade_len:] + α*seg_vt[:fade_len]

        Vp_24h[cursor:cursor+n]    = seg_vp
        Vt_24h[cursor:cursor+n]    = seg_vt
        stage_arr[cursor:cursor+n] = s_int
        prev_vp = seg_vp
        prev_vt = seg_vt
        cursor += n

    return Vp_24h, Vt_24h, stage_arr


# ─── 4. Run Sleep-Regulation ODEs for 24 h  ───────────────────────────────────
# The SR system has no stochastic terms, so scipy.integrate.solve_ivp is exact.

# SR constants (mirror of SleepRegulation class)
_tau_W=1500000; _tau_N=600000; _tau_R=60000
_tau_E=25000;   _tau_G=10000;  _tau_A=10000
_FWm=6.5; _FNm=5.0; _FRm=5.0
_aW=0.5; _aN=0.175; _aR=0.13
_bW=-0.4; _bR=-0.9
_gE=5.0; _gG=4.0; _gA=2.0
_gGW=-1.68; _gAW=1.0; _gGR=-1.3; _gAR=1.6; _gER=-4.0; _gEN=-2.0
_Hm=1.0; _thW=2.0; _thw=34830000; _ths=30600000; _kap=1.5

def sr_odes(t, y):
    fW, fN, fR, CE, CG, CA, h = y
    IW = _gGW*CG + _gAW*CA
    IN = _gEN*CE
    IR = _gER*CE + _gGR*CG + _gAR*CA
    dfW = ((_FWm*0.5*(1+math.tanh((IW-_bW)/_aW))) - fW) / _tau_W
    dfN = ((_FNm*0.5*(1+math.tanh((IN+_kap*h)/_aN))) - fN) / _tau_N
    dfR = ((_FRm*0.5*(1+math.tanh((IR-_bR)/_aR))) - fR) / _tau_R
    dCE = (math.tanh(fW/_gE) - CE) / _tau_E
    dCG = (math.tanh(fN/_gG) - CG) / _tau_G
    dCA = (math.tanh(fR/_gA) - CA) / _tau_A
    dh  = ((_Hm-h)/_thw if fW>=_thW else -h/_ths)
    return [dfW, dfN, dfR, dCE, dCG, dCA, dh]

def run_sr_24h():
    """Integrate SR ODEs for 24 h starting from wake conditions."""
    # Start at typical mid-day (already awake ~8 h, h partly built up)
    fW0, fN0, fR0, h0 = 6.0, 0.001, 0.001, 0.35
    CE0 = math.tanh(fW0/_gE); CG0 = math.tanh(fN0/_gG); CA0 = math.tanh(fR0/_gA)
    y0  = [fW0, fN0, fR0, CE0, CG0, CA0, h0]

    T_ms = T24 * 1e3   # ms
    print('  Integrating SR ODEs for 24 h with scipy…', end=' ', flush=True)
    t0 = time.time()
    sol = solve_ivp(sr_odes, [0, T_ms], y0,
                    method='RK45', dense_output=True,
                    rtol=1e-6, atol=1e-8)
    print(f'done ({time.time()-t0:.1f} s)')

    # Evaluate at 1 Hz = every 1000 ms
    t_eval_ms = np.linspace(0, T_ms, T24 + 1)[:-1]   # 86 400 points
    Y = sol.sol(t_eval_ms)   # shape (7, 86400)
    return Y.T  # (86400, 7): columns = fW fN fR CE CG CA h


# ─── 5. Build SR trace aligned to the 24-h schedule ──────────────────────────
def build_sr_schedule():
    """
    Build neurotransmitter arrays at 1 Hz from the SCHEDULE,
    using smooth transitions between stage NT values.
    Returns array shape (T24, 7): [C_E C_G C_A g_KNa sigma_p f_W_proxy f_N_proxy]
    """
    CE = np.empty(T24); CG = np.empty(T24); CA = np.empty(T24)
    gK = np.empty(T24); sp = np.empty(T24)
    cursor = 0
    prev = None
    TR = int(30)   # 30-s ramp at transitions

    for stage, dur in SCHEDULE:
        nt = STAGE_NT[stage]
        n  = dur
        seg_CE = np.full(n, nt['C_E'])
        seg_CG = np.full(n, nt['C_G'])
        seg_CA = np.full(n, nt['C_A'])
        seg_gK = np.full(n, nt['g_KNa'])
        seg_sp = np.full(n, nt['sigma_p'])

        if prev is not None and TR > 0:
            r = min(TR, n)
            α = np.linspace(0, 1, r)
            seg_CE[:r] = (1-α)*prev[0] + α*seg_CE[:r]
            seg_CG[:r] = (1-α)*prev[1] + α*seg_CG[:r]
            seg_CA[:r] = (1-α)*prev[2] + α*seg_CA[:r]
            seg_gK[:r] = (1-α)*prev[3] + α*seg_gK[:r]
            seg_sp[:r] = (1-α)*prev[4] + α*seg_sp[:r]

        CE[cursor:cursor+n] = seg_CE
        CG[cursor:cursor+n] = seg_CG
        CA[cursor:cursor+n] = seg_CA
        gK[cursor:cursor+n] = seg_gK
        sp[cursor:cursor+n] = seg_sp
        prev = (nt['C_E'], nt['C_G'], nt['C_A'], nt['g_KNa'], nt['sigma_p'])
        cursor += n

    return np.column_stack([CE, CG, CA, gK, sp])  # (86400, 5)


# ─── 6. Main ──────────────────────────────────────────────────────────────────

def main():
    print('\n=== NM_TC_SR 24-hour EEG simulation ===\n')

    # --- waveforms ---
    print('Loading pre-computed stage waveforms…')
    waves = load_waveforms()

    # --- EEG synthesis ---
    print('\nBuilding 24-h EEG from schedule…')
    t0 = time.time()
    Vp, Vt, stage_arr = build_eeg_24h(waves)
    print(f'  done ({time.time()-t0:.1f} s)  shape={Vp.shape}')

    # time vector in hours
    t_h = np.arange(N24, dtype=np.float64) / (FS * 3600)

    # --- SR dynamics ---
    print('\nSleep regulation dynamics…')
    sr_sched = build_sr_schedule()   # 86400 × 5  at 1 Hz
    sr_scipy = run_sr_24h()          # 86400 × 7  from scipy ODE

    # nt_24h: combined SR array (schedule-based NTs + scipy fW/fN/fR)
    nt_24h = np.column_stack([sr_sched, sr_scipy[:, :3]])  # (86400, 8)
    # columns: C_E C_G C_A g_KNa sigma_p fW_scipy fN_scipy fR_scipy

    # --- save numpy arrays ---
    print('\nSaving numpy arrays…')
    np.save(os.path.join(HERE, 'eeg_24h.npy'),   Vp.astype(np.float32))
    np.save(os.path.join(HERE, 'vt_24h.npy'),    Vt.astype(np.float32))
    np.save(os.path.join(HERE, 'stage_24h.npy'), stage_arr)
    np.save(os.path.join(HERE, 't_24h.npy'),     t_h.astype(np.float32))
    np.save(os.path.join(HERE, 'sr_24h.npy'),    nt_24h.astype(np.float32))
    print('  eeg_24h.npy   (Vp, 100 Hz, 8 640 000 samples, float32)')
    print('  vt_24h.npy    (Vt, 100 Hz)')
    print('  stage_24h.npy (stage label, 100 Hz, 0=W 1=N2 2=N3 3=R)')
    print('  t_24h.npy     (time in hours, 100 Hz)')
    print('  sr_24h.npy    (SR variables, 1 Hz, 86 400×8)')

    # --- plots ---
    print('\nGenerating figures…')
    _plot_overview(t_h, Vp, Vt, stage_arr, sr_scipy, nt_24h)
    _plot_epochs(waves)
    print('\nAll done.\n')


# ─── 7. Figures ───────────────────────────────────────────────────────────────

def _hypno_patches():
    patches = [mpatches.Patch(color=STAGE_COL[i], label=STAGE_LAB[i])
               for i in range(4)]
    return patches

def _plot_overview(t_h, Vp, Vt, stage_arr, sr_scipy, nt_24h):
    """Figure 1: 24-h overview — hypnogram, EEG envelope, neurotransmitters."""
    fig, axes = plt.subplots(4, 1, figsize=(18, 12),
                             gridspec_kw={'height_ratios': [1.2, 2, 2, 2]})
    fig.suptitle('24-Hour Sleep Simulation — NM_TC_SR Neural Mass Model',
                 fontsize=14, fontweight='bold')

    # ── Panel 1: hypnogram ──────────────────────────────────────────────────
    ax = axes[0]
    # downsample stage to 1 Hz for plotting
    st_1hz = stage_arr[::FS].astype(int)
    t_1hz  = np.arange(len(st_1hz)) / 3600

    # shade background by stage
    cursor = 0
    while cursor < len(st_1hz):
        s = st_1hz[cursor]
        end = cursor + 1
        while end < len(st_1hz) and st_1hz[end] == s:
            end += 1
        ax.axvspan(t_1hz[cursor], t_1hz[min(end, len(t_1hz)-1)],
                   color=STAGE_COL[s], alpha=0.7)
        cursor = end

    # y-axis: stage order Wake(0) N2(1) N3(2) REM(3) → display inverted
    disp = {0: 3, 1: 2, 2: 1, 3: 0}   # Wake at top, N3 at bottom (clinical)
    y_val = [disp[s] for s in st_1hz]
    ax.step(t_1hz, y_val, 'k-', lw=0.8, where='post')
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['REM', 'N3', 'N2', 'Wake'], fontsize=9)
    ax.set_xlim(0, 24)
    ax.set_ylabel('Stage', fontsize=9)
    ax.set_title('Hypnogram', fontsize=10, loc='left')
    ax.legend(handles=_hypno_patches(), loc='upper right',
              fontsize=8, ncol=4, framealpha=0.8)
    ax.set_xticklabels([])

    # ── Panel 2: EEG amplitude envelope (Vp) ────────────────────────────────
    ax = axes[1]
    WIN = 30 * FS    # 30-s RMS window
    STEP = FS        # 1-s step
    n_win = (len(Vp) - WIN) // STEP + 1
    rms_t  = np.empty(n_win)
    rms_vp = np.empty(n_win)
    for i in range(n_win):
        sl = slice(i*STEP, i*STEP + WIN)
        rms_t[i]  = (i*STEP + WIN/2) / (FS * 3600)
        rms_vp[i] = np.sqrt(np.mean(Vp[sl]**2))
    # color by stage
    st_win = stage_arr[np.clip(np.round(rms_t*3600*FS).astype(int), 0, N24-1)]
    for s in range(4):
        mask = st_win == s
        if mask.any():
            ax.scatter(rms_t[mask], rms_vp[mask],
                       c=STAGE_COL[s], s=2, alpha=0.6, label=STAGE_LAB[s])
    ax.set_xlim(0, 24)
    ax.set_ylabel('RMS Vp [mV]', fontsize=9)
    ax.set_title('Cortical EEG Amplitude (30-s RMS)', fontsize=10, loc='left')
    ax.set_xticklabels([])
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Thalamic Vt RMS ────────────────────────────────────────────
    ax = axes[2]
    rms_vt = np.empty(n_win)
    for i in range(n_win):
        sl = slice(i*STEP, i*STEP + WIN)
        rms_vt[i] = np.sqrt(np.mean(Vt[sl]**2))
    for s in range(4):
        mask = st_win == s
        if mask.any():
            ax.scatter(rms_t[mask], rms_vt[mask],
                       c=STAGE_COL[s], s=2, alpha=0.6)
    ax.set_xlim(0, 24)
    ax.set_ylabel('RMS Vt [mV]', fontsize=9)
    ax.set_title('Thalamic Activity Amplitude (30-s RMS)', fontsize=10, loc='left')
    ax.set_xticklabels([])
    ax.grid(True, alpha=0.3)

    # ── Panel 4: neurotransmitters from scipy SR simulation ─────────────────
    ax = axes[3]
    t_sr = np.arange(T24) / 3600
    CE = sr_scipy[:, 3];  CG = sr_scipy[:, 4];  CA = sr_scipy[:, 5]
    h  = sr_scipy[:, 6]
    ax.plot(t_sr, CE, color='#CC4444', lw=1.0, label='$C_E$ (NE/wake)')
    ax.plot(t_sr, CG, color='#4444CC', lw=1.0, label='$C_G$ (GABA/NREM)')
    ax.plot(t_sr, CA, color='#44AA44', lw=1.0, label='$C_A$ (ACh/REM)')
    ax.plot(t_sr, h,  color='#888800', lw=1.0, ls='--', label='$h$ (homeostatic)')
    ax.set_xlim(0, 24); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Time of day [hours since 07:00]', fontsize=9)
    ax.set_ylabel('Level [aU]', fontsize=9)
    ax.set_title('Sleep-Regulation Neurotransmitters (scipy ODE, from wake IC)',
                 fontsize=10, loc='left')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # x-axis ticks as clock times starting at 07:00
    hours = np.arange(0, 25, 4)
    labels = [f'{(7+h)%24:02d}:00' for h in hours]
    for ax in axes:
        ax.set_xticks(hours)
        ax.set_xticklabels(labels, fontsize=8)

    plt.tight_layout()
    path = os.path.join(HERE, 'fig_24h_overview.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


def _plot_epochs(waves):
    """Figure 2: representative 10-s epochs for each stage, Vp and Vt."""
    EPOCH_S = 10   # seconds to display per stage
    N_EP    = EPOCH_S * FS
    t_ep    = np.arange(N_EP) / FS

    stage_names = ['Wake', 'N2', 'N3', 'REM']
    fig, axes = plt.subplots(4, 2, figsize=(16, 10), sharex=True)
    fig.suptitle('Representative 10-second EEG Epochs per Sleep Stage',
                 fontsize=13, fontweight='bold')

    descriptions = {
        'Wake': 'Low amplitude, fast activity (~8 Hz)',
        'N2':   'K-complexes + 12 Hz spindles (Vt)',
        'N3':   'High-amplitude slow-delta waves',
        'REM':  'Wake-like cortex, quiet thalamus',
    }

    # find a mid-waveform segment (avoid first/last 2 s transients)
    for row, stage in enumerate(stage_names):
        Vp_w, Vt_w = waves[stage]
        start = min(FS * 5, len(Vp_w) - N_EP)   # start 5 s in
        seg_vp = Vp_w[start:start+N_EP]
        seg_vt = Vt_w[start:start+N_EP]

        col_vp = STAGE_COL[STAGE_INT[stage]]

        # Vp
        ax = axes[row, 0]
        ax.plot(t_ep, seg_vp, color=col_vp, lw=0.8)
        ax.set_ylabel(f'{stage}\nVp [mV]', fontsize=9, color=col_vp)
        ax.set_ylim(seg_vp.min()-2, seg_vp.max()+2)
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.set_title('Cortical Vp  (EEG proxy)', fontsize=10)
        ax.text(0.98, 0.92, descriptions[stage],
                transform=ax.transAxes, ha='right', va='top',
                fontsize=7.5, color='#555555')

        # Vt
        ax = axes[row, 1]
        ax.plot(t_ep, seg_vt, color='#2255AA', lw=0.8)
        ax.set_ylabel('Vt [mV]', fontsize=9, color='#2255AA')
        ax.set_ylim(seg_vt.min()-2, seg_vt.max()+2)
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.set_title('Thalamic Vt  (TC relay)', fontsize=10)

    for ax in axes[-1]:
        ax.set_xlabel('Time [s]', fontsize=9)
        ax.set_xticks(np.arange(0, EPOCH_S+1, 2))

    plt.tight_layout()
    path = os.path.join(HERE, 'fig_24h_epochs.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


if __name__ == '__main__':
    main()
