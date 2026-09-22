"""
ViSuS — Music Signal Analysis (Full persona, thin client)
Step 7 of ESM_README.md: app53_patched.py stripped of all compute, calling
the FastAPI backend via api_client.py instead. Same UI/tab structure as the
original; every place that used to call a local compute.py function now
calls api_client and draws the returned arrays with the same matplotlib
code (drawing was never Streamlit- or compute-coupled, so it's unchanged).

Run: streamlit run app_full_client.py
Needs the backend running first: uvicorn main:app --port 8000 (from backend/)
Backend URL: reads st.secrets["BACKEND_URL"] if set, else localhost:8000.

Known, deliberate simplifications vs. the original (flagged, not silent):
  - Alignment is true pairwise now (each pair DTW'd against each other
    directly), not "everyone aligned onto recording #1". Confirmed
    decision, not a port bug -- see ESM_README.md conversation history.
  - Playback plays the full recording; the original could play just the
    zoomed window. Browser players already support seeking, so this
    trades a minor convenience for not needing a waveform-slicing
    endpoint. Can be added back if it's missed in practice.
  - The sidebar's quick-download shortcuts (report/MIDI buttons duplicated
    in the sidebar) are omitted; the main Downloads section at the bottom
    covers the same functionality once, not twice.
"""
import warnings; warnings.filterwarnings("ignore")
import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

import api_client as api
from api_client import BackendError

# ─────────────────────────────────────────────────────────────
# Constants (display/frame math only -- no analysis happens client-side)
# ─────────────────────────────────────────────────────────────
SR, HOP = 22050, 512
N_CQT = 84
MIDI_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
MODE_CHARACTER = {
    "Ionian (Major)":   "Bright, happy, resolved.",
    "Dorian":           "Minor with a raised 6th — jazzy, hopeful.",
    "Phrygian":         "Dark, tense, Spanish flavour.",
    "Lydian":           "Dreamy, floating, raised 4th.",
    "Mixolydian":       "Major but flattened 7th — bluesy, dominant.",
    "Aeolian (Minor)":  "Natural minor — sad, introspective.",
    "Locrian":          "Unstable, dissonant, rarely used.",
}
REC_COLORS = ['#2196F3','#FF9800','#4CAF50','#E91E63','#9C27B0','#00BCD4','#FF5722','#607D8B']
def rec_color(idx): return REC_COLORS[idx % len(REC_COLORS)]

MEL_BANDS_HZ = [
    (0,16,"Sub-bass\n<80Hz"), (16,32,"Bass\n80-300Hz"),
    (32,64,"Low-mids\n300Hz-1kHz"), (64,96,"Mids\n1-4kHz"),
    (96,112,"Presence\n4-8kHz"), (112,128,"Air\n>8kHz"),
]

def cqt_bin_to_note(b):
    """C1 + b semitones. CQT bins are fixed at 12/octave starting at C1
    (matches compute.py's CQT_FMIN/bins_per_octave), so this is plain
    arithmetic, not something requiring librosa client-side."""
    return f"{MIDI_NAMES[b % 12]}{1 + b // 12}"

def t2f(t): return int(np.clip(t*SR/HOP, 0, 999999))
def cl1(a, s, e): return a[s:e] if len(a) > s else a[:0]
def cl2(a, s, e): return a[:, s:e] if a.shape[1] > s else a[:, :0]

def backend_status_banner():
    if not api.health():
        st.error(
            f"⚠️ Can't reach the backend at {api.BASE_URL}. "
            f"Start it first: `uvicorn main:app --reload --port 8000` "
            f"(from the backend/ folder), then reload this page."
        )
        st.stop()

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="ViSuS", layout="wide", initial_sidebar_state="expanded")
st.title("ViSuS — Music Signal Analysis")
backend_status_banner()

# ─────────────────────────────────────────────────────────────
# Upload mode selection
# ─────────────────────────────────────────────────────────────
upload_mode = st.radio(
    "Comparison mode",
    ["2 Rehearsals", "N Rehearsals (batch)"],
    horizontal=True, key="upload_mode",
    help="2 Rehearsals: detailed pairwise analysis. N Rehearsals: compare multiple takes at once.")

# Broadened to everything librosa/soundfile/ffmpeg can realistically decode
# (see DEPLOYMENT.md's audio-format note for why some of these need ffmpeg
# on the backend, not just soundfile). This list is a client-side filter
# only -- the real gate is the backend's decode attempt, which fails with
# a clear error via BackendError if a file turns out to be unreadable.
AUDIO_TYPES = ["wav", "mp3", "flac", "ogg", "oga", "opus", "m4a", "mp4", "aac",
               "wma", "aiff", "aif", "au", "caf", "wv", "amr", "3gp", "3g2", "webm", "mka"]

score_mode = st.radio(
    "Score upload mode",
    ["One score for all recordings", "Different score per recording"],
    index=0, horizontal=True, key="score_upload_mode",
    help="Same setup / multiple takes → one score for all. "
         "Same song / different styles or arrangements → per-recording scores.")

shared_midi = None
if score_mode == "One score for all recordings":
    shared_midi = st.file_uploader(
        "Score for all recordings (optional)", type=["mid","midi"], key="fu_midi_shared",
        help="This single MIDI file will be used as the reference score for every recording uploaded below.")

if upload_mode == "2 Rehearsals":
    c1, c2 = st.columns(2)
    with c1:
        file_a = st.file_uploader("Rehearsal 1", type=AUDIO_TYPES, key="fu_a")
        name_a = st.text_input("Name", value="Rehearsal 1", key="name_a")
        midi_a = shared_midi if score_mode == "One score for all recordings" else \
            st.file_uploader("Score for Rehearsal 1 (optional)", type=["mid","midi"], key="fu_midi_a")
    with c2:
        file_b = st.file_uploader("Rehearsal 2", type=AUDIO_TYPES, key="fu_b")
        name_b = st.text_input("Name", value="Rehearsal 2", key="name_b")
        midi_b = shared_midi if score_mode == "One score for all recordings" else \
            st.file_uploader("Score for Rehearsal 2 (optional)", type=["mid","midi"], key="fu_midi_b")
    files, names, midi_files = [file_a, file_b], [name_a, name_b], [midi_a, midi_b]
    n_recs = 2
else:
    st.markdown("Upload between 2 and 8 rehearsal recordings. All pairwise comparisons will be computed.")
    n_recs_input = st.slider("Number of rehearsals", min_value=2, max_value=8, value=3, key="n_recs_slider")
    files, names, midi_files = [], [], []
    cols = st.columns(min(n_recs_input, 4))
    for i in range(n_recs_input):
        col = cols[i % len(cols)]
        with col:
            f = st.file_uploader(f"Rehearsal {i+1}", type=AUDIO_TYPES, key=f"fu_{i}")
            n = st.text_input("Name", value=f"Rehearsal {i+1}", key=f"name_{i}")
            m = shared_midi if score_mode == "One score for all recordings" else \
                st.file_uploader("Score (optional)", type=["mid","midi"], key=f"fu_midi_{i}")
            files.append(f); names.append(n); midi_files.append(m)
    n_recs = n_recs_input

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ViSuS")
    for f, n in zip(files, names):
        if f: st.caption(f"{n}: {f.name[:30]}")
    st.markdown("---")

    st.subheader("Features")
    with st.expander("Core", expanded=True):
        do_mel      = st.checkbox("Mel Spectrogram",     value=True)
        do_cqt      = st.checkbox("CQT + Key Detection", value=True)
        do_mfcc     = st.checkbox("MFCCs",               value=True)
        do_chroma   = st.checkbox("Chroma",              value=True)
        do_spectral = st.checkbox("Spectral Features",   value=True)
        do_onset    = st.checkbox("Onset & Beats",       value=True)
        do_sms      = st.checkbox("SMS / HPSS",          value=True)
    with st.expander("Optional", expanded=False):
        do_stft      = st.checkbox("STFT",               value=False)
        do_cwt       = st.checkbox("CWT / Scalogram",    value=False)
        do_gammatone = st.checkbox("Gammatone",          value=False)
        do_tonnetz   = st.checkbox("Tonnetz",            value=False)
        do_zcr       = st.checkbox("ZCR",                value=False)
        do_reverb    = st.checkbox("RT60 / Reverb",      value=False)

    st.markdown("---")
    st.markdown("**⏱ DTW Alignment**")
    dtw_strategy_choice = st.radio(
        "Strategy", ["Chroma + Onset (combo)", "Chroma only", "Onset only", "Off"],
        index=0, key="dtw_strategy_radio",
        help="Chroma aligns pitch. Onset aligns rhythm. Combo blends both 50/50.")
    do_dtw = dtw_strategy_choice != "Off"
    dtw_strategy = {"Chroma + Onset (combo)":"combo", "Chroma only":"chroma",
                     "Onset only":"onset", "Off":"none"}[dtw_strategy_choice]

    st.markdown("---")
    st.subheader("🔍 Zoom Window")
    zoom_mode = st.radio("Mode", ["Full recording","Time (seconds)","Musical bars"],
                          key="zoom_mode", label_visibility="collapsed")
    zoom_start_sb = 0.0; zoom_end_sb = None
    bar_start_sb = bar_end_sb = time_sig_sb = None
    if zoom_mode == "Time (seconds)":
        zoom_start_sb = st.number_input("Start (s)", min_value=0.0, value=0.0, step=0.5)
        zoom_end_sb   = st.number_input("End (s)",   min_value=0.0, value=30.0, step=0.5)
    elif zoom_mode == "Musical bars":
        bar_start_sb = st.number_input("From bar", min_value=1, value=1, step=1)
        bar_end_sb   = st.number_input("To bar",   min_value=1, value=4, step=1)
        time_sig_sb  = st.selectbox("Time sig", [4,3,6,5,7], index=0)

    st.markdown("---")
    diff_threshold = st.slider("Waveform diff threshold", 0.0, 0.3, 0.03, 0.01)
    st.markdown("---")
    run_btn = st.button("▶ Run Analysis", type="primary", use_container_width=True)
    st.markdown("---")
    st.caption("ViSuS · HIWI Research Demo · thin client")

FEATURE_FLAGS = dict(
    do_mel=do_mel, do_cqt=do_cqt, do_mfcc=do_mfcc, do_chroma=do_chroma,
    do_spectral=do_spectral, do_onset=do_onset, do_sms=do_sms, do_stft=do_stft,
    do_cwt=do_cwt, do_gammatone=do_gammatone, do_tonnetz=do_tonnetz,
    do_zcr=do_zcr, do_reverb=do_reverb,
)

def wav_bytes_for_playback(uploaded_file):
    """Playback uses the raw uploaded bytes directly -- the browser's
    native audio decode handles mp3/wav/flac/etc without a round trip.
    No zoom-window cropping (see module docstring)."""
    uploaded_file.seek(0)
    return uploaded_file.read()

# ─────────────────────────────────────────────────────────────
# Run Analysis — upload every file, store results in session_state
# ─────────────────────────────────────────────────────────────
ready = all(f is not None for f in files)
if run_btn:
    if not ready:
        st.warning(f"Upload all {n_recs} recordings before running analysis.")
    else:
        import json as _json
        recordings = []
        with st.spinner("Uploading and analysing..."):
            for f, n in zip(files, names):
                f.seek(0)
                try:
                    result = api.upload_recording(
                        f.read(), f.name, feature_flags=FEATURE_FLAGS)
                except BackendError as e:
                    st.error(f"Upload failed for {n}: {e}")
                    st.stop()
                recordings.append({"recording_id": result["recording_id"], "name": n, "R": result["R"]})
            # Attach uploaded scores, if any
            for rec, midi_f in zip(recordings, midi_files):
                if midi_f is not None:
                    midi_f.seek(0)
                    try:
                        api.upload_score(rec["recording_id"], midi_f.read())
                    except BackendError as e:
                        st.warning(f"Score upload failed for {rec['name']}: {e}")
        st.session_state["recordings"] = recordings
        st.session_state["names"] = names
        st.session_state["file_bytes"] = [f.getvalue() for f in files]
        st.session_state["durs"] = [r["R"].get("dur", 0.0) for r in recordings]
        st.success("Analysis complete.")

if "recordings" not in st.session_state:
    st.info("Upload recordings and click **▶ Run Analysis** to begin.")
    st.stop()

recordings = st.session_state["recordings"]
names      = st.session_state["names"]
durs       = st.session_state["durs"]
file_bytes = st.session_state["file_bytes"]
n_recs     = len(recordings)
rec_ids    = [r["recording_id"] for r in recordings]
Rs         = [r["R"] for r in recordings]  # unaligned, as uploaded

st.success(
    f"✓ {n_recs} recording{'s' if n_recs>1 else ''} analysed: "
    + ", ".join(f"**{n}** ({d:.1f}s)" for n, d in zip(names, durs))
    + "  ·  Change settings and click **▶ Run Analysis** to re-analyse.",
    icon="🎵")

# ─────────────────────────────────────────────────────────────
# Zoom window
# ─────────────────────────────────────────────────────────────
dur_max = max(durs) if durs else 1.0
if zoom_mode == "Time (seconds)":
    zoom_start = float(zoom_start_sb)
    zoom_end   = float(min(zoom_end_sb or dur_max, dur_max))
elif zoom_mode == "Musical bars":
    avg_t = float(np.mean([R.get("tempo", 120) for R in Rs]))
    spb = 60/avg_t; zoom_start = (bar_start_sb-1)*spb*time_sig_sb
    zoom_end = min(bar_end_sb*spb*time_sig_sb, dur_max)
else:
    zoom_start, zoom_end = 0.0, dur_max
zoom_start = max(0.0, zoom_start); zoom_end = min(zoom_end, dur_max)
if zoom_end <= zoom_start: zoom_end = zoom_start + 1.0
fs = t2f(zoom_start); fe = t2f(zoom_end)

# ─────────────────────────────────────────────────────────────
# Shared plot helpers (pure drawing -- unchanged from app53_patched.py,
# these never touched compute; they take already-computed arrays)
# ─────────────────────────────────────────────────────────────
def spec_pair(title, ma, mb, la, lb, cmap="magma", ylabel="", yticks=None,
              col_a=None, col_b=None, add_mel_bands=False,
              ref_label="reference", cmp_label="aligned"):
    import matplotlib.colors as mcolors
    nc = min(ma.shape[1], mb.shape[1]); ma, mb = ma[:,:nc], mb[:,:nc]
    diff = mb-ma; vmax = float(np.percentile(np.abs(diff),95)) or 1.0
    n_rows = ma.shape[0]; ext = [zoom_start, zoom_end, 0, n_rows]
    fig, axes = plt.subplots(1,3,figsize=(16,3.5))
    im_a = axes[0].imshow(ma, aspect='auto', origin='lower', cmap=cmap, interpolation='nearest', extent=ext)
    axes[0].set_title(f"{la}  [{ref_label}]", fontsize=8, color=col_a or 'white', fontweight='bold')
    axes[0].set_xlabel("Time (s)", fontsize=7); axes[0].set_ylabel(ylabel, fontsize=7)
    for sp in axes[0].spines.values(): sp.set_edgecolor(col_a or 'steelblue'); sp.set_linewidth(2)
    plt.colorbar(im_a, ax=axes[0], format="%+.0f", label="dB")
    im_b = axes[1].imshow(mb, aspect='auto', origin='lower', cmap=cmap, interpolation='nearest', extent=ext)
    axes[1].set_title(f"{lb}  [{cmp_label}]", fontsize=8, color=col_b or 'white', fontweight='bold')
    axes[1].set_xlabel("Time (s)", fontsize=7)
    for sp in axes[1].spines.values(): sp.set_edgecolor(col_b or 'darkorange'); sp.set_linewidth(2)
    plt.colorbar(im_b, ax=axes[1], format="%+.0f", label="dB")
    ca = mcolors.to_rgb(col_a or '#2196F3'); cb_ = mcolors.to_rgb(col_b or '#FF9800')
    diff_cmap = mcolors.LinearSegmentedColormap.from_list('rec_diff', [ca, (1,1,1), cb_], N=256)
    im3 = axes[2].imshow(diff, aspect='auto', origin='lower', cmap=diff_cmap,
                          interpolation='nearest', extent=ext, vmin=-vmax, vmax=vmax)
    cb3 = plt.colorbar(im3, ax=axes[2], format="%+.0f")
    cb3.set_label(f"(+) = {lb} louder  |  (−) = {la} louder", fontsize=6)
    axes[2].set_title(f"Δ = {lb}  MINUS  {la}\norange tint = {lb} has more energy · blue tint = {la} has more energy", fontsize=7)
    axes[2].set_xlabel("Time (s)", fontsize=7)
    if yticks:
        for ax in axes:
            ax.set_yticks(list(yticks.keys())); ax.set_yticklabels(list(yticks.values()), fontsize=5)
    if add_mel_bands:
        for ax in axes:
            for lo, hi, lbl in MEL_BANDS_HZ:
                mid = (lo+hi)/2
                ax.axhline(lo, color='white', lw=0.4, alpha=0.4)
                ax.text(zoom_start+(zoom_end-zoom_start)*0.01, mid, lbl, fontsize=4, color='white', va='center', alpha=0.8)
    fig.suptitle(title, fontsize=10, fontweight='bold'); plt.tight_layout()
    return fig

def line_pair(title, da, db, la, lb, ylabel="", yrange=None, yscale='linear', col_a=None, col_b=None):
    ca = col_a or rec_color(0); cb = col_b or rec_color(1)
    n = min(len(da), len(db)); da, db = np.array(da[:n]), np.array(db[:n])
    t = np.linspace(zoom_start, zoom_end, n); diff = db-da
    fig, axes = plt.subplots(3,1,figsize=(14,6), sharex=True)
    for ax, d, lbl, col in zip(axes[:2], [da,db], [la,lb], [ca,cb]):
        ax.plot(t, d, lw=0.8, color=col); ax.set_ylabel(ylabel, fontsize=7)
        ax.set_title(lbl, fontsize=8, color=col, fontweight='bold')
        if yrange: ax.set_ylim(yrange)
        if yscale == 'log': ax.set_yscale('log')
    diff_max = max(abs(diff.max()), abs(diff.min()), 1e-8)
    axes[2].fill_between(t, diff, where=diff>=0, color=cb, alpha=0.55, label=f'{lb} higher')
    axes[2].fill_between(t, diff, where=diff<0,  color=ca, alpha=0.55, label=f'{la} higher')
    axes[2].axhline(0, color='black', lw=1.2, zorder=5)
    axes[2].set_ylim(-diff_max*1.15, diff_max*1.15)
    axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:+.3g}"))
    axes[2].set_ylabel(f"Δ {ylabel}", fontsize=7); axes[2].set_xlabel("Time (s)", fontsize=7)
    axes[2].legend(fontsize=7)
    axes[2].set_title(f"Δ = {lb}  MINUS  {la}  (above zero = {lb} higher · below zero = {la} higher)", fontsize=8)
    if diff.max() > 0:
        peak_pos = t[int(np.argmax(diff))]
        axes[2].annotate(f"+{diff.max():.3g}", xy=(peak_pos, diff.max()), xytext=(peak_pos, diff.max()*1.05), fontsize=6, ha='center', color=cb)
    if diff.min() < 0:
        peak_neg = t[int(np.argmin(diff))]
        axes[2].annotate(f"{diff.min():.3g}", xy=(peak_neg, diff.min()), xytext=(peak_neg, diff.min()*1.05), fontsize=6, ha='center', color=ca)
    fig.suptitle(title, fontsize=10, fontweight='bold'); plt.tight_layout(); return fig

def diff_note(label, va, vb, unit, math_txt, musical_txt):
    d = vb-va; pct = d/(abs(va)+1e-10)*100
    st.markdown(f"**{label}:** 1=`{va:.4g} {unit}` · 2=`{vb:.4g} {unit}` · Δ=`{d:+.4g}` ({pct:+.1f}%)  \n"
                f"*Math:* {math_txt}  \n*Musical:* {musical_txt}")

def diff_formula(name_a, name_b, col_a, col_b):
    st.markdown(
        f"<div style='background:#111;padding:6px 12px;border-radius:6px;font-size:0.8rem;margin-bottom:4px'>"
        f"<b>Difference =</b> <span style='color:{col_b};font-weight:bold'>{name_b}</span> <b>minus</b> "
        f"<span style='color:{col_a};font-weight:bold'>{name_a}</span> &nbsp;·&nbsp; "
        f"<span style='color:{col_b}'>■</span> tint = {name_b} has more energy &nbsp;·&nbsp; "
        f"<span style='color:{col_a}'>■</span> tint = {name_a} has more energy</div>",
        unsafe_allow_html=True)

def insight_card(icon, headline, detail):
    st.markdown(
        f"<div style='background:#1a1a2e;border-left:4px solid #4a9eff;border-radius:6px;"
        f"padding:10px 14px;margin-bottom:12px'><span style='font-size:1.2rem'>{icon}</span> "
        f"<strong>{headline}</strong><br><span style='color:#aaa;font-size:0.82rem'>{detail}</span></div>",
        unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# N-recording matrix view (backend-driven)
# ─────────────────────────────────────────────────────────────
def render_matrix():
    st.subheader("Pairwise Similarity Matrix")
    st.caption("Green = similar · Red = different.")
    try:
        sim_matrix = api.matrix_similarity(rec_ids)
    except BackendError as e:
        st.error(f"Matrix unavailable: {e}"); return

    def cell_bg(sim):
        r=int(233*(1-sim)+76*sim); g=int(30*(1-sim)+175*sim); b=int(99*(1-sim)+80*sim)
        return f"rgb({r},{g},{b})"
    def text_col(sim): return "#000" if sim>0.55 else "#fff"

    n = n_recs
    cell_size = max(90, min(150, 640 // n))
    css = f"""<style>
      *{{box-sizing:border-box;margin:0;padding:0;}}
      body{{background:#0f0f1a;font-family:sans-serif;}}
      .sim-grid{{display:grid;grid-template-columns:100px repeat({n},{cell_size}px);gap:3px;margin-bottom:6px;}}
      .sim-header{{background:#1a1a2e;color:#aaa;font-size:10px;padding:4px;text-align:center;border-radius:4px;
        display:flex;align-items:center;justify-content:center;font-weight:bold;word-break:break-word;}}
      .sim-row-label{{background:#1a1a2e;color:#aaa;font-size:10px;padding:4px 6px;border-radius:4px;
        display:flex;align-items:center;font-weight:bold;word-break:break-word;}}
      .sim-cell{{border-radius:6px;padding:4px;text-align:center;height:{cell_size}px;
        display:flex;flex-direction:column;align-items:center;justify-content:center;}}
      .sim-cell .pct{{font-size:13px;font-weight:bold;}}
      .sim-diag{{background:#1a1a2e;color:#444;font-size:20px;border-radius:6px;
        display:flex;align-items:center;justify-content:center;height:{cell_size}px;}}
    </style>"""
    grid = '<div class="sim-grid"><div class="sim-header"></div>'
    for j in range(n):
        short = names[j] if len(names[j])<=10 else names[j][:9]+"…"
        grid += f'<div class="sim-header">{short}</div>'
    for i in range(n):
        short_row = names[i] if len(names[i])<=10 else names[i][:9]+"…"
        grid += f'<div class="sim-row-label">{short_row}</div>'
        for j in range(n):
            if i==j:
                grid += '<div class="sim-diag">—</div>'
            else:
                sim = sim_matrix[i,j]; bg = cell_bg(sim); tc = text_col(sim)
                grid += f'<div class="sim-cell" style="background:{bg};color:{tc}"><span class="pct">{sim*100:.0f}%</span></div>'
    grid += '</div>'
    import streamlit.components.v1 as components
    components.html(css+grid, height=44+n*(cell_size+4), scrolling=False)

    if n >= 3:
        st.markdown("#### Recording Map (MDS)")
        try:
            mds = api.matrix_mds(rec_ids)
        except BackendError as e:
            st.caption(f"MDS unavailable: {e}")
        else:
            green_thresh = mds["green_thresh_pct"]/100.0
            red_thresh   = mds["red_thresh_pct"]/100.0
            with st.expander("⚖️ Feature groups + weights", expanded=False):
                selected_groups = st.multiselect(
                    "Include feature groups", options=mds["available_groups"],
                    default=mds["available_groups"], key="mds_feature_groups")
                weights = {}
                if selected_groups:
                    w_cols = st.columns(min(3, len(selected_groups)))
                    for idx, g in enumerate(selected_groups):
                        with w_cols[idx % len(w_cols)]:
                            weights[g] = st.slider(g, 0.1, 3.0, 1.0, 0.1, key=f"mds_weight_{g}")
                if selected_groups != mds["available_groups"] or any(w != 1.0 for w in weights.values()):
                    try:
                        mds = api.matrix_mds(rec_ids, selected_groups=selected_groups or None, weights=weights or None)
                        green_thresh = mds["green_thresh_pct"]/100.0
                        red_thresh   = mds["red_thresh_pct"]/100.0
                    except BackendError as e:
                        st.caption(f"MDS re-fetch failed: {e}")

            coords = mds["display_coords"]; dist = mds["distance_matrix"]
            fig2, ax2 = plt.subplots(figsize=(max(6.5,5.0+0.6*n), max(5.2,4.2+0.5*n)))
            fig2.patch.set_facecolor('#0f0f1a'); ax2.set_facecolor('#0f0f1a')
            x_range = coords[:,0].max()-coords[:,0].min()+1e-8
            y_range = coords[:,1].max()-coords[:,1].min()+1e-8
            for i in range(n):
                for j in range(i+1, n):
                    x0,y0 = coords[i]; x1,y1 = coords[j]; d = dist[i,j]; sim_val = 1.0-d
                    line_col = '#4CAF50' if sim_val>=green_thresh else '#FF9800' if sim_val>=red_thresh else '#E91E63'
                    ax2.plot([x0,x1],[y0,y1], color=line_col, lw=1.6, alpha=0.45, zorder=1)
                    if n <= 4:
                        ax2.text((x0+x1)/2,(y0+y1)/2, f"sim: {sim_val*100:.0f}%", fontsize=6, color='#ccc',
                                  ha='center', va='center', bbox=dict(boxstyle='round,pad=0.15', fc='#1a1a2e', ec='none', alpha=0.8), zorder=2)
            for i, (nm,(x,y)) in enumerate(zip(names, coords)):
                ax2.scatter(x,y,s=200,color=rec_color(i),zorder=3,edgecolors='white',linewidths=1.5)
                short_nm = nm if len(nm)<=14 else nm[:13]+"…"
                ax2.text(x, y+0.06*y_range, short_nm, fontsize=7.5, color='white', ha='center', va='bottom', fontweight='bold', zorder=4)
            pad_x = 0.20*x_range+0.05*max(x_range,y_range); pad_y = 0.15*y_range+0.05*max(x_range,y_range)
            ax2.set_xlim(coords[:,0].min()-pad_x, coords[:,0].max()+pad_x)
            ax2.set_ylim(coords[:,1].min()-pad_y, coords[:,1].max()+pad_y*1.3)
            ax2.set_xticks([]); ax2.set_yticks([])
            ax2.set_title("All-features distance", fontsize=9, fontweight='bold', color='white', pad=6)
            for sp in ax2.spines.values(): sp.set_visible(False)
            if mds["overlap_groups"]:
                lines = [f"({', '.join(names[k] for k in grp)}) are near-identical — spread in a ring" for grp in mds["overlap_groups"]]
                ax2.text(0.5, -0.07, "\n".join(lines), transform=ax2.transAxes, fontsize=6.5, color='#888', ha='center', va='top')
            plt.tight_layout(); st.pyplot(fig2, use_container_width=True); plt.close()
            st.caption(f"Distance = normalised Euclidean distance in a {mds['n_dims']}D feature space "
                       f"({', '.join(mds['selected_groups'])}). Closer dots = more similar.")

    st.subheader("Feature Comparison — All Rehearsals")
    metrics = []
    if all("tempo" in R for R in Rs):
        metrics.append(("Tempo (BPM)", [R["tempo"] for R in Rs]))
    if all("beat_reg" in R for R in Rs):
        metrics.append(("Beat Regularity (lower=tighter)", [R["beat_reg"] for R in Rs]))
    if all("rms" in R for R in Rs):
        metrics.append(("Average Loudness (RMS)", [float(np.mean(R["rms"])) for R in Rs]))
    if all("centroid" in R for R in Rs):
        metrics.append(("Brightness (Hz)", [float(np.mean(R["centroid"])) for R in Rs]))
    if all("sms_ratio" in R for R in Rs):
        metrics.append(("Harmonic Ratio", [float(np.mean(R["sms_ratio"])) for R in Rs]))
    if metrics:
        n_m = len(metrics)
        fig_feat, axes_feat = plt.subplots(1, n_m, figsize=(min(16, n_m*3.5), 4))
        if n_m == 1: axes_feat = [axes_feat]
        colors = plt.cm.tab10(np.linspace(0, 0.9, len(names)))
        for ax, (feat_name, vals) in zip(axes_feat, metrics):
            bars = ax.bar(range(len(names)), vals, color=colors, alpha=0.85)
            ax.set_xticks(range(len(names))); ax.set_xticklabels([n[:12] for n in names], rotation=30, ha='right', fontsize=8)
            ax.set_title(feat_name, fontsize=8, fontweight='bold')
            best = int(np.argmin(vals)) if "Regularity" in feat_name else int(np.argmax(vals))
            bars[best].set_edgecolor('gold'); bars[best].set_linewidth(2.5)
        plt.tight_layout(); st.pyplot(fig_feat, use_container_width=True); plt.close()
        st.caption("Gold outline = best performing rehearsal for each metric.")

    if all("key" in R for R in Rs):
        st.markdown("**Key & Mode per rehearsal:**")
        cols = st.columns(len(names))
        for col, name, R in zip(cols, names, Rs):
            col.markdown(f"**{name}**  \n{R.get('key','—')}  \n{R.get('mode','—')}")

    st.markdown("---")
    st.subheader("All Rehearsals — Group View")
    try:
        gv = api.matrix_group_view(rec_ids, t0=zoom_start, t1=zoom_end)
    except BackendError as e:
        st.info(f"Group view unavailable: {e}"); return
    if not gv:
        st.info("Enable Spectral Features and Onset to see overlay plots."); return
    ov_tab1, ov_tab2 = st.tabs(["📈 All recordings overlaid", "📊 Group mean ± spread"])
    with ov_tab1:
        n_feats = len(gv)
        fig_ov, axes_ov = plt.subplots(n_feats, 1, figsize=(14, 2.5*n_feats), sharex=True)
        fig_ov.patch.set_facecolor('#0f0f1a')
        if n_feats == 1: axes_ov = [axes_ov]
        for ax, (feat, s) in zip(axes_ov, gv.items()):
            ax.set_facecolor('#0f0f1a')
            for idx, nm in enumerate(names):
                arr = s["per_recording"].get(nm)
                if arr is None: continue
                ax.plot(s["time"], arr, lw=0.9, color=rec_color(idx), label=nm, alpha=0.85)
            ax.set_ylabel(s["label"], fontsize=7, color='white'); ax.tick_params(colors='white', labelsize=6)
            for sp in ax.spines.values(): sp.set_edgecolor('#333')
            ax.legend(fontsize=7, loc="upper right", facecolor='#1a1a2e', labelcolor='white', edgecolor='#333')
        axes_ov[-1].set_xlabel("Time (s)", fontsize=8, color='white')
        plt.tight_layout(); st.pyplot(fig_ov, use_container_width=True); plt.close()
    with ov_tab2:
        n_feats = len(gv)
        fig_ms, axes_ms = plt.subplots(n_feats, 1, figsize=(14, 3.0*n_feats), sharex=True)
        fig_ms.patch.set_facecolor('#0f0f1a')
        if n_feats == 1: axes_ms = [axes_ms]
        for ax, (feat, s) in zip(axes_ms, gv.items()):
            ax.set_facecolor('#0f0f1a')
            ax.fill_between(s["time"], s["mean"]-s["std"], s["mean"]+s["std"], alpha=0.25, color='white', label='±1 std')
            ax.plot(s["time"], s["mean"], lw=2.0, color='white', label='Group mean', zorder=4)
            for idx, nm in enumerate(names):
                arr = s["per_recording"].get(nm)
                if arr is None: continue
                ax.plot(s["time"], arr, lw=0.8, color=rec_color(idx), label=nm, alpha=0.75, zorder=3)
                outliers = s["outlier_frames"].get(nm, [])
                if outliers:
                    idx_arr = np.array(outliers)
                    idx_arr = idx_arr[idx_arr < len(arr)]
                    if len(idx_arr): ax.scatter(s["time"][idx_arr], arr[idx_arr], s=12, color=rec_color(idx), zorder=5, alpha=0.6)
            ax.set_ylabel(s["label"], fontsize=7, color='white'); ax.tick_params(colors='white', labelsize=6)
            for sp in ax.spines.values(): sp.set_edgecolor('#333')
            ax.legend(fontsize=7, loc="upper right", facecolor='#1a1a2e', labelcolor='white', edgecolor='#333', ncol=2)
        axes_ms[-1].set_xlabel("Time (s)", fontsize=8, color='white')
        plt.tight_layout(); st.pyplot(fig_ms, use_container_width=True); plt.close()
        st.markdown("**Outlier summary:**")
        for feat, s in gv.items():
            if s["outlier_summary"]:
                o = s["outlier_summary"]
                st.markdown(f"- **{s['label']}:** *{o['name']}* deviated most (mean Δ = {o['mean_deviation']:.3f}, others avg {o['others_avg']:.3f})")


def apply_warp_local(arr, wp, n_a):
    """Apply an already-computed DTW warping path to a raw feature array.
    This is display-side application of a precomputed alignment (the path
    itself came from api.get_alignment()) -- not new analysis, so it's
    fine to keep here rather than adding another backend round trip for
    every before/after panel. Verbatim port of compute.apply_warp()."""
    pa, pb = wp[:, 0], wp[:, 1]
    if arr.ndim == 2:
        out = np.zeros((arr.shape[0], n_a), dtype=np.float32)
        for i in range(n_a):
            idx = pb[pa == i]
            if len(idx): out[:, i] = arr[:, np.clip(idx, 0, arr.shape[1]-1)].mean(1)
    else:
        out = np.zeros(n_a, dtype=np.float32)
        for i in range(n_a):
            idx = pb[pa == i]
            if len(idx): out[i] = arr[np.clip(idx, 0, len(arr)-1)].mean()
    return out


# ─────────────────────────────────────────────────────────────
# Song Map — backend-driven (api.get_song_map + api.get_sections)
# ─────────────────────────────────────────────────────────────
def render_song_map(a_id, b_id, name_a, name_b):
    try:
        sm = api.get_song_map(a_id, b_id)
    except BackendError as e:
        st.info(f"Song Map unavailable for this pair: {e}")
        return
    secs = api.get_sections(a_id, b_id)
    time_full = sm["time"]

    st.markdown(f"**Song Map — {name_a} vs {name_b}**")
    st.caption(f"{len(secs)} sections detected")

    map_tabs = st.tabs(["📋 Section breakdown", "🌊 Stream graph", "📊 Stacked bars"])
    VERDICT_ICON = {"same": "🟢", "slight": "🟡", "noticeable": "🔴"}
    VERDICT_WORD = {"same": "Same", "slight": "Slight diff", "noticeable": "Notable diff"}

    # ── View 1: per-section framing ─────────────────────────
    with map_tabs[0]:
        st.caption("Each section is analysed in isolation. "
                   "Differences are flagged neutrally — no judgement of 'better'.")
        if not secs:
            st.info("No sections detected.")
        else:
            for b0, b1, t0, t1, lbl, cmp in secs:
                if not cmp:
                    continue
                verdicts = [v for v, _ in cmp.values()]
                if all(v == "same" for v in verdicts):
                    overall_icon, overall_word = "🟢", "Very similar"
                elif any(v == "noticeable" for v in verdicts):
                    overall_icon, overall_word = "🔴", "Notable differences"
                else:
                    overall_icon, overall_word = "🟡", "Minor differences"
                with st.expander(
                    f"{overall_icon} **{lbl}** — bars {b0}–{b1} "
                    f"· {t0:.0f}–{t1:.0f}s · {overall_word}",
                    expanded=(overall_icon == "🔴"),
                ):
                    for dim, (verdict, detail) in cmp.items():
                        icon = VERDICT_ICON.get(verdict, "⚪")
                        st.markdown(
                            f"{icon} **{dim}:** {VERDICT_WORD.get(verdict, verdict)}  \n"
                            f"<span style='color:#888;font-size:0.8rem'>{detail}</span>",
                            unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("**Section summary:**")
            summary_cols = st.columns(len(secs))
            for col, (b0, b1, t0, t1, lbl, cmp) in zip(summary_cols, secs):
                verdicts = [v for v, _ in cmp.values()]
                icon = ("🟢" if all(v == "same" for v in verdicts)
                         else "🔴" if any(v == "noticeable" for v in verdicts) else "🟡")
                col.markdown(f"**{lbl}**  \n{icon} bars {b0}–{b1}")

    # ── View 2: stream graph ─────────────────────────────────
    with map_tabs[1]:
        st.caption(
            "Stream graph — each coloured band shows how much one feature "
            "contributes to the total difference at each moment. "
            "Wider band = that feature is driving the divergence more.")
        bands, colors = sm["bands"], sm["colors"]
        if not bands:
            st.info("Enable Spectral Features and Onset for the stream graph.")
        else:
            labels_s = list(bands.keys())
            min_len = min(len(v) for v in bands.values())
            times_s = np.array(time_full[:min_len])
            vals_s = np.array([bands[l][:min_len] for l in labels_s])
            cols_s = [colors[l] for l in labels_s]

            fig_sg, ax_sg = plt.subplots(figsize=(14, 4))
            ax_sg.set_facecolor('#0f0f1a'); fig_sg.patch.set_facecolor('#0f0f1a')
            ax_sg.axhline(0, color='white', lw=1.0, alpha=0.5, zorder=2)
            legend_handles = []
            cum_pos = np.zeros(min_len)
            for label, val_sc, col in zip(labels_s, vals_s, cols_s):
                upper = cum_pos + val_sc
                patch = ax_sg.fill_between(times_s, cum_pos, upper, color=col, alpha=0.82, label=label)
                legend_handles.append(patch); cum_pos = upper
            cum_neg = np.zeros(min_len)
            for val_sc, col in zip(vals_s, cols_s):
                lower = cum_neg - val_sc
                ax_sg.fill_between(times_s, lower, cum_neg, color=col, alpha=0.82)
                cum_neg = lower
            total_s = vals_s.sum(axis=0)
            mid_t = int(np.argmax(total_s))
            cum_lbl = np.zeros(min_len)
            for label, val_sc, col in zip(labels_s, vals_s, cols_s):
                mid_y = (cum_lbl[mid_t] + cum_lbl[mid_t] + val_sc[mid_t]) / 2
                if val_sc[mid_t] > 0.02:
                    ax_sg.text(times_s[mid_t], mid_y, label, fontsize=8, color='white',
                               ha='center', va='center', fontweight='bold')
                cum_lbl += val_sc
            for b0, b1, t0_s, t1_s, lbl, cmp in secs:
                ax_sg.axvline(t0_s, color='white', lw=0.6, alpha=0.4, ls='--')
                ax_sg.text(t0_s+0.3, 0.98, lbl, fontsize=6, color='white',
                           va='top', transform=ax_sg.get_xaxis_transform())
            if zoom_mode != "Full recording":
                ax_sg.axvspan(zoom_start, zoom_end, alpha=0.15, color='yellow', zorder=0)
            ax_sg.set_xlim(0, float(times_s[-1]) if min_len else 1.0)
            ax_sg.set_ylim(-0.55, 0.55)
            ax_sg.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
            ax_sg.set_yticklabels(["-0.5", "-0.25", "0", "+0.25", "+0.5"], fontsize=7, color='white')
            ax_sg.set_xlabel("Time (s)", fontsize=8, color='white')
            ax_sg.set_ylabel("Divergence", fontsize=8, color='white')
            ax_sg.set_title(f"What drives the difference — {name_a} vs {name_b}",
                            fontweight='bold', fontsize=10, color='white')
            ax_sg.tick_params(colors='white')
            for sp in ax_sg.spines.values(): sp.set_edgecolor('#333')
            ax_sg.legend(handles=legend_handles[::-1], labels=labels_s[::-1], fontsize=8,
                        loc='upper right', facecolor='#1a1a2e', labelcolor='white', edgecolor='#333')
            plt.tight_layout(); st.pyplot(fig_sg, use_container_width=True); plt.close()

            moments = sm["top_divergent_moments"]
            if moments:
                st.markdown("**Top 3 most different moments:**")
                for rank, m in enumerate(moments):
                    pt2 = m["time"]
                    st.markdown(
                        f"{rank+1}. **{int(pt2//60)}:{int(pt2%60):02d}** — "
                        f"divergence {m['divergence']:.2f} · driven by **{m['dominant_band']}**")
                st.caption("Use the zoom window in the sidebar to focus on any of these moments.")

    # ── View 3: stacked bars per section ─────────────────────
    with map_tabs[2]:
        st.caption(
            "Each bar = one detected section. "
            "Bar is split into coloured segments showing different types of difference. "
            "Taller total bar = more overall divergence.")
        rows = sm["section_bars"]
        if not rows:
            st.info("No sections detected.")
        else:
            dim_keys = ["rms", "centroid", "onset", "sms_ratio"]
            dim_labels = ["Volume", "Brightness", "Rhythm", "Harmonic"]
            dim_colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]
            sec_labels = [f"{r['label']}\n{r['start_time']:.0f}–{r['end_time']:.0f}s" for r in rows]
            x = np.arange(len(rows))
            fig_sb, ax_sb = plt.subplots(figsize=(max(8, len(rows)*1.6), 4))
            bottoms = np.zeros(len(rows))
            for dk, dlbl, dcol in zip(dim_keys, dim_labels, dim_colors):
                vals = np.array([r["divergence"].get(dk, 0.0) for r in rows])
                ax_sb.bar(x, vals, bottom=bottoms, label=dlbl, color=dcol, alpha=0.82, width=0.6)
                bottoms += vals
            ax_sb.set_xticks(x); ax_sb.set_xticklabels(sec_labels, fontsize=8)
            ax_sb.set_ylabel("Divergence (stacked by dimension)", fontsize=9)
            ax_sb.set_title(f"Section divergence — {name_a} vs {name_b}", fontweight="bold", fontsize=10)
            ax_sb.legend(fontsize=8, loc="upper right"); ax_sb.axhline(0, color="black", lw=0.5)
            totals = bottoms
            best_i = int(np.argmin(totals)); worst_i = int(np.argmax(totals))
            ax_sb.annotate("most similar", xy=(best_i, totals[best_i]), xytext=(best_i, totals[best_i]+0.05),
                           ha="center", fontsize=7, color="green", arrowprops=dict(arrowstyle="-", color="green", lw=0.8))
            ax_sb.annotate("most different", xy=(worst_i, totals[worst_i]), xytext=(worst_i, totals[worst_i]+0.05),
                           ha="center", fontsize=7, color="red", arrowprops=dict(arrowstyle="-", color="red", lw=0.8))
            plt.tight_layout(); st.pyplot(fig_sb, use_container_width=True); plt.close()

            st.markdown("**Section breakdown table:**")
            header_cols = st.columns([2]+[1]*len(dim_labels)+[1])
            header_cols[0].markdown("**Section**")
            for i, dl in enumerate(dim_labels): header_cols[i+1].markdown(f"**{dl}**")
            header_cols[-1].markdown("**Total**")
            for r in rows:
                total = sum(r["divergence"].get(dk, 0) for dk in dim_keys)
                icon = "🟢" if total < 0.15 else "🟡" if total < 0.35 else "🔴"
                row_cols = st.columns([2]+[1]*len(dim_labels)+[1])
                row_cols[0].markdown(f"**{r['label']}** {icon}")
                for i, dk in enumerate(dim_keys):
                    row_cols[i+1].markdown(f"{r['divergence'].get(dk, 0):.2f}")
                row_cols[-1].markdown(f"**{total:.2f}**")


# ─────────────────────────────────────────────────────────────
# Score / MIDI Alignment — backend-driven (api.score_alignment)
# ─────────────────────────────────────────────────────────────
def render_score_alignment(rec_ids_sa, names_sa):
    st.markdown("### Score / MIDI Alignment")
    st.caption(
        "Each recording is transcribed to MIDI via **pYIN** pitch estimation "
        "(monophonic instrument assumed). If a recording has its own uploaded "
        "score, it's aligned against that score directly. Recordings without "
        "their own score are aligned against a consensus (the average of the "
        "other auto-transcriptions).")
    with st.spinner("Generating MIDI transcriptions..."):
        try:
            sa = api.score_alignment(rec_ids_sa)
        except BackendError as e:
            st.error(f"Score alignment failed: {e}"); return

    names_l = names_sa
    voiced_pcts = sa["voiced_pcts"]
    alignments = sa["alignments"]
    used_uploaded_score = sa["any_uploaded_score"]
    n = len(names_l)

    if used_uploaded_score:
        n_own = sum(1 for a in alignments if a and a.get("used_own_score"))
        st.info(f"📄 {n_own} of {n} recording(s) using their own uploaded score. "
                f"The rest use the auto-generated consensus.")

    if voiced_pcts is not None:
        st.markdown("**pYIN transcription confidence:**")
        st.caption(
            "**Voiced %** = proportion of frames where a clear pitch was detected. "
            "Low values mean the recording has many silent, noisy, or unpitched frames. "
            "Best results: >70% voiced.")
        conf_cols = st.columns(n)
        for col, nm, vp in zip(conf_cols, names_l, voiced_pcts):
            icon = "✅" if vp >= 70 else "⚠️" if vp >= 40 else "❌"
            col.metric(f"{icon} {nm}", f"{vp:.0f}% voiced",
                       delta="reliable" if vp >= 70 else "low confidence",
                       delta_color="normal" if vp >= 70 else "inverse")
        low_conf = [nm for nm, vp in zip(names_l, voiced_pcts) if vp < 40]
        if low_conf:
            st.warning(f"⚠️ Low pitch detection confidence for: **{', '.join(low_conf)}**. "
                       "The alignment score for these recordings should be interpreted with caution.")

    if any(a and a.get("used_own_score") for a in alignments if a):
        st.markdown("**Transcription accuracy — auto-generated MIDI vs uploaded score:**")
        acc_cols = st.columns(n)
        for col, nm, aln in zip(acc_cols, names_l, alignments):
            if aln is None:
                col.caption(f"{nm}: —"); continue
            if aln.get("used_own_score"):
                acc = aln.get("transcription_accuracy")
                if acc is not None:
                    icon = "✅" if acc >= 0.75 else "⚠️" if acc >= 0.5 else "❌"
                    col.metric(f"{icon} {nm}", f"{acc*100:.0f}% match")
                else:
                    col.caption(f"{nm}: own score, accuracy N/A")
            else:
                col.caption(f"{nm}: using consensus\n(no own score uploaded)")

    with st.expander("📄 Consensus MIDI Score (chroma)", expanded=False):
        cons = sa["consensus_chroma"]
        fs_sc, fe_sc = t2f(zoom_start), t2f(zoom_end)
        cons_w = cons[:, fs_sc:fe_sc]
        if cons_w.shape[1] > 0:
            fig_c, ax_c = plt.subplots(figsize=(14, 2.5))
            fig_c.patch.set_facecolor('#0f0f1a'); ax_c.set_facecolor('#0f0f1a')
            ax_c.imshow(cons_w, aspect='auto', origin='lower', cmap='YlOrRd',
                       extent=[zoom_start, zoom_end, 0, 12])
            ax_c.set_yticks(range(12)); ax_c.set_yticklabels(MIDI_NAMES, fontsize=6, color='white')
            ax_c.set_xlabel("Time (s)", fontsize=7, color='white')
            ax_c.set_title("Consensus MIDI chroma — average across all pYIN transcriptions",
                           fontsize=8, color='white')
            ax_c.tick_params(colors='white', labelsize=6)
            for sp in ax_c.spines.values(): sp.set_edgecolor('#333')
            plt.tight_layout(); st.pyplot(fig_c, use_container_width=True); plt.close()

    with st.expander("🔊 Listen — sanity check the MIDI transcriptions", expanded=False):
        st.caption(
            "Synthesized playback (simple sine-wave synth) of each pYIN transcription. "
            "If the MIDI sounds musically unrelated to the recording, the transcription — "
            "and therefore its alignment score — should not be trusted for that recording. "
            "(Reference-score playback is omitted here — only per-recording transcriptions "
            "are available for now; see api_client.download_transcription_audio.)")
        for idx, (rid, nm) in enumerate(zip(rec_ids_sa, names_l)):
            try:
                audio_bytes = api.download_transcription_audio(rid)
            except BackendError as e:
                st.caption(f"{nm}: playback unavailable ({e})"); continue
            st.caption(f"{nm} — pYIN MIDI transcription")
            st.audio(audio_bytes, format='audio/wav')

    st.markdown("**Alignment scores** (each vs its own score, or the consensus if none was uploaded):")
    score_cols = st.columns(n)
    for col, nm, aln in zip(score_cols, names_l, alignments):
        if aln is None:
            col.metric(nm, "N/A", help="Alignment failed"); continue
        col.metric(nm, f"{aln['score']*100:.0f}%", f"mean offset {aln['mean_offset_s']:.2f}s",
                   delta_color="inverse")

    st.markdown("**Temporal offset vs consensus score over time:**")
    st.caption("How many seconds each recording drifts from the consensus score at each moment. "
               "Closer to 0 = more on-score.")
    valid_alns = [(nm, aln) for nm, aln in zip(names_l, alignments) if aln is not None]
    if valid_alns:
        fig_off, ax_off = plt.subplots(figsize=(14, 3))
        fig_off.patch.set_facecolor('#0f0f1a'); ax_off.set_facecolor('#0f0f1a')
        ax_off.axhline(0, color='white', lw=0.8, alpha=0.4)
        for idx, (nm, aln) in enumerate(valid_alns):
            offsets = aln["offsets"]
            step = max(1, len(offsets)//2000)
            t_ds = np.arange(len(offsets))[::step] * HOP / SR
            off_ds = offsets[::step]
            n_plot = min(len(t_ds), len(off_ds))
            ax_off.plot(t_ds[:n_plot], off_ds[:n_plot], lw=1.0, color=rec_color(idx),
                       label=f"{nm} ({aln['score']*100:.0f}%)", alpha=0.85)
        ax_off.set_xlabel("Time (s)", fontsize=8, color='white')
        ax_off.set_ylabel("Offset from score (s)", fontsize=8, color='white')
        ax_off.tick_params(colors='white', labelsize=7)
        for sp in ax_off.spines.values(): sp.set_edgecolor('#333')
        ax_off.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white', edgecolor='#333')
        plt.tight_layout(); st.pyplot(fig_off, use_container_width=True); plt.close()

    st.markdown("**MIDI transcription vs audio chroma — per recording:**")
    st.caption("Top row = pYIN MIDI chroma. Bottom row = audio chroma. "
               "Similar patterns = the recording follows the score closely.")
    fs_sc, fe_sc = t2f(zoom_start), t2f(zoom_end)
    for idx, (rid, nm, mc, aln) in enumerate(zip(rec_ids_sa, names_l, sa["midi_chromas"], alignments)):
        with st.expander(f"{nm}  —  score {aln['score']*100:.0f}%" if aln else nm, expanded=False):
            mc_w = mc[:, fs_sc:min(fe_sc, mc.shape[1])]
            R_idx = [r for r in st.session_state["recordings"] if r["recording_id"] == rid]
            aud_c = R_idx[0]["R"].get("chroma") if R_idx else None
            if aud_c is not None:
                aud_w = aud_c[:, fs_sc:min(fe_sc, aud_c.shape[1])]
                nc = min(mc_w.shape[1], aud_w.shape[1])
                if nc > 0:
                    fig_cmp, axes_cmp = plt.subplots(2, 1, figsize=(14, 4), sharex=True)
                    fig_cmp.patch.set_facecolor('#0f0f1a')
                    ext = [zoom_start, zoom_end, 0, 12]
                    for ax, mat, title in [
                        (axes_cmp[0], mc_w[:, :nc], f"pYIN MIDI chroma — {nm}"),
                        (axes_cmp[1], aud_w[:, :nc], f"Audio chroma — {nm}"),
                    ]:
                        ax.set_facecolor('#0f0f1a')
                        ax.imshow(mat, aspect='auto', origin='lower', cmap='YlOrRd', extent=ext)
                        ax.set_yticks(range(12)); ax.set_yticklabels(MIDI_NAMES, fontsize=6, color='white')
                        ax.set_title(title, fontsize=8, color=rec_color(idx), fontweight='bold')
                        for sp in ax.spines.values(): sp.set_edgecolor(rec_color(idx)); sp.set_linewidth(1.5)
                        ax.tick_params(colors='white', labelsize=6)
                    axes_cmp[-1].set_xlabel("Time (s)", fontsize=7, color='white')
                    plt.tight_layout(); st.pyplot(fig_cmp, use_container_width=True); plt.close()


# ─────────────────────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────────────────────
main_col, side_col = st.columns([3, 1])

# ── Floating side panel ─────────────────────────────────────
with side_col:
    st.markdown("#### 🎧 Playback")
    for fb, name in zip(file_bytes, names):
        st.caption(f"**{name}**")
        st.audio(fb)

    # Mini waveforms (first 3 only if N>2) -- approximated from
    # api.get_waveform()'s fixed-length downsample, cropped to the zoom
    # window by index proportion (see module docstring: no raw audio
    # decoding happens client-side, so this is coarser than the
    # original's exact-sample-rate crop, but visually equivalent at
    # thumbnail size).
    n_ds = 300
    try:
        wave_previews = [api.get_waveform(rid, n_points=n_ds) for rid in rec_ids[:3]]
        fig_w, axes_w = plt.subplots(min(n_recs, 3), 1, figsize=(3, min(n_recs, 3)*1.0), sharex=True)
        if min(n_recs, 3) == 1: axes_w = [axes_w]
        colors_w = [rec_color(i) for i in range(len(names))]
        for ax, wp, name, col, d in zip(axes_w, wave_previews, names[:3], colors_w, durs[:3]):
            samp = wp["samples"]; dur_i = max(wp["duration"], 1e-6)
            i0 = int(np.clip(zoom_start/dur_i*len(samp), 0, len(samp)))
            i1 = int(np.clip(zoom_end/dur_i*len(samp), 0, len(samp)))
            ds = samp[i0:i1] if i1 > i0 else samp
            tw = np.linspace(zoom_start, zoom_end, len(ds)) if len(ds) else np.array([zoom_start])
            ax.plot(tw, ds, lw=0.5, color=col); ax.set_title(name[:18], fontsize=6); ax.set_yticks([])
        axes_w[-1].set_xlabel("Time (s)", fontsize=6)
        plt.tight_layout(pad=0.2); st.pyplot(fig_w, use_container_width=True); plt.close()
    except BackendError:
        pass

    st.markdown("---")
    for name, R in zip(names, Rs):
        if "tempo" in R:
            st.markdown(f"**{name}**  \n{R['tempo']} BPM · {R.get('key', '—')}")
            if "mode" in R: st.caption(R["mode"])
    st.markdown("---")
    st.caption(f"Window: {zoom_start:.1f}–{zoom_end:.1f}s")
    st.caption(f"DTW: {'✓' if do_dtw else '✗'}")

# ── Main panel ───────────────────────────────────────────────
with main_col:
    if n_recs > 2:
        render_matrix()
        st.markdown("---")

        pairs = [(i, j) for i in range(n_recs) for j in range(i+1, n_recs)]
        pair_labels = [f"{names[i]} vs {names[j]}" for i, j in pairs]
        try:
            sim_mat_full = api.matrix_similarity(rec_ids)
        except BackendError as e:
            st.error(f"Could not compute similarity matrix: {e}"); st.stop()
        pair_scores = [float(sim_mat_full[i, j]) for i, j in pairs]

        best_idx = int(np.argmax(pair_scores))
        worst_idx = int(np.argmin(pair_scores))

        st.subheader("Pairwise Similarity")
        st.caption("Computed from timbre, harmony, tempo, rhythm, and brightness. "
                   "Higher = more similar. Auto-selects the closest pair for detailed comparison.")

        score_cols = st.columns(len(pairs))
        for col, (i, j), score, label in zip(score_cols, pairs, pair_scores, pair_labels):
            is_best = (i, j) == pairs[best_idx]
            is_worst = (i, j) == pairs[worst_idx]
            tag = " ✅ auto-selected" if is_best else (" ⚠️ most different" if is_worst else "")
            bg = "#1a3a1a" if is_best else ("#3a1a1a" if is_worst else "#1a1a2e")
            col.markdown(
                f"<div style='background:{bg};border-radius:8px;padding:10px;text-align:center'>"
                f"<div style='font-size:0.8rem;color:#aaa'>{label}</div>"
                f"<div style='font-size:2rem;font-weight:bold;color:{'#4CAF50' if is_best else ('#E91E63' if is_worst else '#fff')}'>"
                f"{score*100:.0f}%</div>"
                f"<div style='font-size:0.7rem;color:#888'>{tag}</div></div>",
                unsafe_allow_html=True)

        with st.expander("Why these scores? — feature breakdown"):
            breakdown_rows = []
            for (i, j), label in zip(pairs, pair_labels):
                try:
                    sim = api.get_similarity(rec_ids[i], rec_ids[j], names[i], names[j])
                except BackendError:
                    continue
                row = {"Pair": label}
                for k, v in sim["sub_scores"].items():
                    row[k] = f"{v*100:.0f}%"
                breakdown_rows.append(row)
            if breakdown_rows:
                cols_b = list(breakdown_rows[0].keys())
                header = "| " + " | ".join(cols_b) + " |"
                sep = "| " + " | ".join(["---"]*len(cols_b)) + " |"
                rows_md = [header, sep]
                for row in breakdown_rows:
                    rows_md.append("| " + " | ".join(str(row.get(c, "—")) for c in cols_b) + " |")
                st.markdown("\n".join(rows_md))

        outlier_counts = {i: 0 for i in range(n_recs)}
        for (i, j), score in zip(pairs, pair_scores):
            if score == min(pair_scores):
                outlier_counts[i] += 1; outlier_counts[j] += 1
        outlier_idx = max(outlier_counts, key=outlier_counts.get)
        best_pi, best_pj = pairs[best_idx]
        outlier_name = names[outlier_idx]
        best_name_a, best_name_b = names[best_pi], names[best_pj]

        st.info(
            f"**Auto-selected:** {best_name_a} vs {best_name_b} "
            f"({pair_scores[best_idx]*100:.0f}% similar) — the closest pair.  \n"
            f"**Outlier:** {outlier_name} differs most from the others. "
            f"It may represent a session that went differently.")

        st.subheader("Detailed Comparison — All Pairs")
        st.caption("Each tab is one pair. Switch freely — no rerun needed. ⭐ = auto-selected (most similar).")
        n_pairs = len(pairs)

        def pair_label(i, j, score, is_best):
            star = "⭐" if is_best else ""
            sim_s = f"{score*100:.0f}%"
            if n_pairs <= 3:
                return f"{star}{names[i]} vs {names[j]} {sim_s}"
            return f"{star}R{i+1}·R{j+1} {sim_s}"

        tab_labels_pairs = [pair_label(i, j, score, idx == best_idx)
                             for idx, ((i, j), score) in enumerate(zip(pairs, pair_scores))]
        if n_pairs > 15:
            st.warning(f"⚠️ {n_recs} recordings produce {n_pairs} pairs. "
                      "Consider using the N-recording overview above to identify the most relevant pairs first.")
        pair_tabs = st.tabs(tab_labels_pairs)
        if n_pairs > 3:
            legend_cols = st.columns(min(n_recs, 6))
            for col, (k, nm) in zip(legend_cols, enumerate(names)):
                col.caption(f"**R{k+1}** = {nm}")
    else:
        pair_tabs = None
        pairs = [(0, 1)]
        best_idx = 0

    # ── Render comparison for each pair ─────────────────────
    def _render_pair(pi, pj):
        a_id, b_id = rec_ids[pi], rec_ids[pj]
        name_a_sel, name_b_sel = names[pi], names[pj]
        try:
            Ra_sel, Rb_sel = api.get_aligned_pair(a_id, b_id)
        except BackendError as e:
            st.error(f"Could not align {name_a_sel} vs {name_b_sel}: {e}"); return
        try:
            sim = api.get_similarity(a_id, b_id, name_a_sel, name_b_sel)
        except BackendError:
            sim = None
        try:
            aln_data = api.get_alignment(a_id, b_id)
        except BackendError:
            aln_data = None

        ALL_TABS = ["Overview", "Song Map", "Alignment"]
        if do_chroma:   ALL_TABS.append("Harmony")
        if do_onset:    ALL_TABS.append("Rhythm")
        if do_mfcc:     ALL_TABS.append("Timbre")
        if do_spectral: ALL_TABS.append("Dynamics")
        has_specs = do_mel or do_cqt
        if has_specs:   ALL_TABS.append("Spectrograms")
        if do_chroma:   ALL_TABS.append("Score Alignment")
        adv = [k for k, v in [("SMS/HPSS", do_sms), ("Reverb", do_reverb), ("STFT", do_stft),
                               ("CWT", do_cwt), ("Gammatone", do_gammatone), ("Tonnetz", do_tonnetz)] if v]
        if adv: ALL_TABS.append("Advanced")

        tabs = st.tabs(ALL_TABS)
        tab_map = {n: t for n, t in zip(ALL_TABS, tabs)}

        has_dtw = aln_data is not None and any(
            aln_data.get(k) is not None for k in ("wp_chroma", "wp_onset", "wp_combo"))

        # ── Overview ─────────────────────────────────────────
        with tab_map["Overview"]:
            st.markdown(f"### {name_a_sel} vs {name_b_sel}")
            st.caption(f"Window: {zoom_start:.1f}s → {zoom_end:.1f}s  |  "
                      f"A: {durs[pi]:.1f}s  B: {durs[pj]:.1f}s  |  DTW: {'chroma+onset+RMS' if has_dtw else 'off'}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"Tempo {name_a_sel}", f"{Ra_sel.get('tempo', 120)} BPM")
            c2.metric(f"Tempo {name_b_sel}", f"{Rb_sel.get('tempo', 120)} BPM",
                      f"{Rb_sel.get('tempo', 120)-Ra_sel.get('tempo', 120):+.1f}")
            if "key" in Ra_sel:
                c3.metric(f"Key {name_a_sel}", Ra_sel["key"])
                c4.metric(f"Key {name_b_sel}", Rb_sel.get("key", "—"))
            if sim is not None:
                st.metric("Timbre Similarity", f"{sim['sub_scores'].get('Timbre', 0)*100:.1f}%")
                for ins in sim["insights"]:
                    insight_card(ins["icon"], ins["headline"], ins["detail"])
            try:
                wv_a = api.get_waveform(a_id, n_points=2000)
                wv_b = api.get_waveform(b_id, n_points=2000)
                wa, wb = wv_a["samples"], wv_b["samples"]
                n_wv = min(len(wa), len(wb))
                wa, wb = wa[:n_wv], wb[:n_wv]
                t_wv = np.linspace(0, min(durs[pi], durs[pj]), n_wv)
                diff_w = wb - wa; mag = np.abs(diff_w)
                norm_mag = np.clip(mag/(mag.max()+1e-8), 0, 1); mask = mag > diff_threshold
                fig_wv, axes_wv = plt.subplots(3, 1, figsize=(14, 6), sharex=True)
                axes_wv[0].plot(t_wv, wa, lw=0.4, color=rec_color(pi))
                axes_wv[0].set_title(name_a_sel, fontsize=8, color=rec_color(pi), fontweight='bold')
                axes_wv[1].plot(t_wv, wb, lw=0.4, color=rec_color(pj))
                axes_wv[1].set_title(name_b_sel, fontsize=8, color=rec_color(pj), fontweight='bold')
                for i in range(len(t_wv)-1):
                    if not mask[i]: continue
                    col_i = (1, 0, 0, float(norm_mag[i])) if diff_w[i] > 0 else (0, 0, 1, float(norm_mag[i]))
                    axes_wv[2].fill_between(t_wv[i:i+2], [diff_w[i], diff_w[i+1]], color=col_i)
                axes_wv[2].axhline(0, color='black', lw=0.8)
                dw_max = max(abs(diff_w.max()), abs(diff_w.min()), 1e-8)
                axes_wv[2].set_ylim(-dw_max*1.15, dw_max*1.15)
                axes_wv[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.3g}"))
                axes_wv[2].set_title(f"Δ = {name_b_sel} minus {name_a_sel}", fontsize=8)
                axes_wv[2].set_xlabel("Time (s)", fontsize=7)
                plt.tight_layout(); st.pyplot(fig_wv, use_container_width=True); plt.close()
            except BackendError as e:
                st.caption(f"Waveform diff unavailable: {e}")

        # ── Song Map ─────────────────────────────────────────
        with tab_map["Song Map"]:
            render_song_map(a_id, b_id, name_a_sel, name_b_sel)

        # ── Alignment ────────────────────────────────────────
        with tab_map["Alignment"]:
            if not has_dtw or aln_data is None:
                st.info("Enable DTW Alignment in the sidebar to see alignment analysis.")
            else:
                st.markdown("### DTW Alignment — Three Strategies")
                st.caption(
                    "Each strategy aligns the recordings using a different musical signal. "
                    "**Chroma**: matches harmonic/melodic content. **Onset**: matches rhythmic "
                    "attack positions. **RMS**: coarse offset applied before all strategies.")
                rms_off_s = aln_data.get("rms_offset_seconds", 0)
                if rms_off_s:
                    st.info(f"Coarse RMS offset detected: **{rms_off_s:+.2f}s** "
                           f"({'B starts later' if rms_off_s > 0 else 'B starts earlier'}). "
                           "Applied before all DTW strategies.")

                st.subheader("Warping paths — all three strategies")
                fig_cmp, axes_cmp = plt.subplots(1, 3, figsize=(16, 4))
                strategy_info = [
                    (aln_data.get("wp_chroma"), "Chroma DTW", rec_color(0)),
                    (aln_data.get("wp_onset"), "Onset DTW", rec_color(1)),
                    (aln_data.get("wp_combo"), "Chroma+Onset Combo", rec_color(2)),
                ]
                for ax, (wp, label, col) in zip(axes_cmp, strategy_info):
                    if wp is None:
                        ax.text(0.5, 0.5, "Not available", ha='center', va='center',
                               transform=ax.transAxes, fontsize=9, color='gray')
                        ax.set_title(label, fontsize=9); continue
                    step = max(1, len(wp)//1000); wp_ds = wp[::step]
                    ax.plot(wp_ds[:, 0]*HOP/SR, wp_ds[:, 1]*HOP/SR, lw=1.0, color=col)
                    max_t = max(durs[pi], durs[pj])
                    ax.plot([0, max_t], [0, max_t], '--', color='gray', lw=0.7, alpha=0.6, label='perfect')
                    avg_off = float(np.mean(np.abs(wp[:, 0]-wp[:, 1]))*HOP/SR)
                    ax.set_title(f"{label} — mean offset ±{avg_off:.2f}s", fontsize=8, color=col, fontweight='bold')
                    ax.set_xlabel(f"{name_a_sel} (s)", fontsize=7); ax.set_ylabel(f"{name_b_sel} (s)", fontsize=7)
                    ax.legend(fontsize=6)
                plt.suptitle("DTW Warping Paths — diagonal = perfect alignment", fontsize=10, fontweight='bold')
                plt.tight_layout(); st.pyplot(fig_cmp, use_container_width=True); plt.close()

                st.subheader("Temporal offset over time")
                fig_off, ax_off = plt.subplots(figsize=(14, 3))
                for wp, label, col in strategy_info:
                    if wp is None: continue
                    offset = (wp[:, 0]-wp[:, 1]).astype(float)*HOP/SR
                    step = max(1, len(offset)//2000)
                    t_p = wp[::step, 0]*HOP/SR; off_ds = offset[::step]
                    ax_off.plot(t_p, off_ds, lw=1.0, color=col, label=label, alpha=0.85)
                ax_off.axhline(0, color='black', lw=0.8)
                ax_off.set_xlabel("Time (s)", fontsize=8); ax_off.set_ylabel("Offset (s)", fontsize=8)
                ax_off.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.2f}s"))
                ax_off.legend(fontsize=8)
                ax_off.set_title("How much each strategy shifts B to match A at each moment", fontsize=9, fontweight='bold')
                plt.tight_layout(); st.pyplot(fig_off, use_container_width=True); plt.close()
                st.caption("All three curves close together = recordings are well-aligned by any measure.")

                st.subheader("Before vs After alignment — feature comparison")
                active_label = {"chroma": "Chroma DTW", "onset": "Onset DTW", "combo": "Chroma+Onset Combo"}.get(dtw_strategy, "Combo")
                st.caption(f"Active strategy for feature alignment: **{active_label}**. "
                          "All three paths computed and shown for comparison above.")
                strat_tab_names = [s[1] for s in strategy_info if s[0] is not None]
                if strat_tab_names:
                    strat_tabs = st.tabs(strat_tab_names)
                    for stab, (wp, label, col) in zip(strat_tabs, strategy_info):
                        if wp is None: continue
                        with stab:
                            st.caption(f"**{label}** — comparing raw B vs B warped using {label.lower()}")
                            if "rms" in Ra_sel and "rms" in Rs[pj]:
                                n_a_warp = Ra_sel["rms"].shape[0] if "chroma" not in Ra_sel else Ra_sel["chroma"].shape[1]
                                rms_b_warp = apply_warp_local(Rs[pj]["rms"].astype(np.float32), wp, n_a_warp)
                                rms_a, rms_b_raw = Ra_sel["rms"], Rs[pj]["rms"]
                                n_ = min(len(rms_a), len(rms_b_raw), len(rms_b_warp))
                                t_ax = np.linspace(zoom_start, zoom_end, min(n_, fe-fs) if fe > fs else n_)
                                n_w = len(t_ax)
                                col_a, col_b = rec_color(pi), rec_color(pj)
                                fig_ba, axes_ba = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
                                for ax, db, sfx in [(axes_ba[0], rms_b_raw[:n_w], "Before alignment"),
                                                     (axes_ba[1], rms_b_warp[:n_w], "After alignment")]:
                                    ax.plot(t_ax, rms_a[:n_w], lw=0.9, color=col_a, label=name_a_sel)
                                    ax.plot(t_ax, db, lw=0.9, color=col_b, label=f"{name_b_sel} ({sfx})", alpha=0.8)
                                    n_plt = min(len(t_ax), len(rms_a), len(db))
                                    diff_rms = db[:n_plt] - rms_a[:n_plt]
                                    ax.fill_between(t_ax[:n_plt], rms_a[:n_plt], db[:n_plt], where=diff_rms >= 0, alpha=0.18, color=col_b)
                                    ax.fill_between(t_ax[:n_plt], rms_a[:n_plt], db[:n_plt], where=diff_rms < 0, alpha=0.18, color=col_a)
                                    ax.set_ylabel("RMS Energy", fontsize=7); ax.legend(fontsize=7)
                                    ax.set_title(sfx, fontsize=8, color='#aaa' if sfx == "Before alignment" else '#4CAF50')
                                axes_ba[1].set_xlabel("Time (s)", fontsize=8)
                                plt.suptitle(f"RMS Energy — {label}", fontsize=10, fontweight='bold')
                                plt.tight_layout(); st.pyplot(fig_ba, use_container_width=True); plt.close()
                            if "chroma" in Ra_sel and "chroma" in Rs[pj]:
                                n_ac = Ra_sel["chroma"].shape[1]
                                ch_b_warp = apply_warp_local(Rs[pj]["chroma"].astype(np.float32), wp, n_ac)
                                ca = cl2(Ra_sel["chroma"], fs, fe); cb_raw = cl2(Rs[pj]["chroma"], fs, fe); cb_aln = cl2(ch_b_warp, fs, fe)
                                nc = min(ca.shape[1], cb_raw.shape[1], cb_aln.shape[1])
                                import matplotlib.colors as mcol
                                col_a_ch, col_b_ch = rec_color(pi), rec_color(pj)
                                dcmap = mcol.LinearSegmentedColormap.from_list('d', [mcol.to_rgb(col_a_ch), (1, 1, 1), mcol.to_rgb(col_b_ch)], N=256)
                                fig_ch, axes_ch = plt.subplots(2, 3, figsize=(16, 5))
                                for row, (cb, sfx) in enumerate([(cb_raw[:, :nc], "Before"), (cb_aln[:, :nc], "After")]):
                                    diff = cb - ca[:, :nc]; vmax = float(np.percentile(np.abs(diff), 95)) or 1.0
                                    ext = [zoom_start, zoom_end, 0, 12]
                                    for col_j, (mat, nm, c_) in enumerate([(ca[:, :nc], name_a_sel, col_a_ch),
                                                                            (cb, f"{name_b_sel} ({sfx})", col_b_ch),
                                                                            (diff, "Difference", None)]):
                                        if col_j < 2:
                                            axes_ch[row, col_j].imshow(mat, aspect='auto', origin='lower', cmap='YlOrRd', extent=ext)
                                            axes_ch[row, col_j].set_title(nm, fontsize=8, color=c_, fontweight='bold')
                                            for sp in axes_ch[row, col_j].spines.values(): sp.set_edgecolor(c_); sp.set_linewidth(2)
                                        else:
                                            axes_ch[row, col_j].imshow(diff, aspect='auto', origin='lower', cmap=dcmap, extent=ext, vmin=-vmax, vmax=vmax)
                                            axes_ch[row, col_j].set_title("Difference", fontsize=8)
                                        axes_ch[row, col_j].set_yticks(range(12)); axes_ch[row, col_j].set_yticklabels(MIDI_NAMES, fontsize=6)
                                plt.suptitle(f"Chroma — {label}", fontsize=10, fontweight='bold')
                                plt.tight_layout(); st.pyplot(fig_ch, use_container_width=True); plt.close()

        # ── Harmony ──────────────────────────────────────────
        if do_chroma and "chroma" in Ra_sel and "Harmony" in tab_map:
            with tab_map["Harmony"]:
                if "key" in Ra_sel:
                    same_key = Ra_sel.get("key") == Rb_sel.get("key")
                    insight_card("🎹",
                        f"Key & mode: {'both in '+Ra_sel['key'] if same_key else Ra_sel['key']+' vs '+Rb_sel.get('key','—')}",
                        MODE_CHARACTER.get(Ra_sel.get("mode_name", ""), ""))
                c1, c2 = st.columns(2)
                for col, R_src, nm in [(c1, Ra_sel, name_a_sel), (c2, Rb_sel, name_b_sel)]:
                    with col:
                        st.markdown(f"**{nm}**")
                        st.markdown(f"Key: {R_src.get('key', '—')} · Mode: {R_src.get('mode', '—')}")
                        st.caption(MODE_CHARACTER.get(R_src.get("mode_name", ""), ""))
                        ch_w = np.array(cl2(np.array(R_src["chroma"]), fs, fe))
                        cm = ch_w.mean(1) if ch_w.shape[1] > 0 else np.array(R_src["chroma"]).mean(1)
                        top3 = sorted(range(12), key=lambda i: -cm[i])[:3]
                        st.markdown(f"Dominant notes: **{', '.join(MIDI_NAMES[i] for i in top3)}**")
                ca_w = cl2(Ra_sel["chroma"], fs, fe); cb_w = cl2(Rb_sel["chroma"], fs, fe)
                nc = min(ca_w.shape[1], cb_w.shape[1])
                if nc > 0:
                    ca_m = ca_w[:, :nc].mean(1); cb_m = cb_w[:, :nc].mean(1)
                    cc = float(np.dot(ca_m, cb_m)/(np.linalg.norm(ca_m)*np.linalg.norm(cb_m)+1e-10))
                    st.markdown(f"**Harmonic similarity: {int((cc+1)/2*100)}%**")
                    st.progress((cc+1)/2)
                    x = np.arange(12); dc = cb_m - ca_m
                    diff_formula(name_a_sel, name_b_sel, rec_color(pi), rec_color(pj))
                    fig_cb, axes_cb = plt.subplots(2, 1, figsize=(12, 5))
                    axes_cb[0].bar(x-0.2, ca_m, 0.35, label=name_a_sel, color=rec_color(pi), alpha=0.85)
                    axes_cb[0].bar(x+0.2, cb_m, 0.35, label=name_b_sel, color=rec_color(pj), alpha=0.85)
                    axes_cb[0].set_xticks(x); axes_cb[0].set_xticklabels(MIDI_NAMES)
                    axes_cb[0].set_title("Pitch class energy", fontweight='bold'); axes_cb[0].legend()
                    axes_cb[1].bar(x, dc, color=[rec_color(pj) if d > 0 else rec_color(pi) for d in dc], alpha=0.8)
                    axes_cb[1].axhline(0, color='black', lw=1.2, zorder=5)
                    dc_max = max(abs(dc.max()), abs(dc.min()), 1e-8)
                    axes_cb[1].set_ylim(-dc_max*1.2, dc_max*1.2)
                    axes_cb[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.2f}"))
                    axes_cb[1].set_xticks(x); axes_cb[1].set_xticklabels(MIDI_NAMES)
                    axes_cb[1].set_title(f"Δ = {name_b_sel} minus {name_a_sel}", fontsize=9)
                    plt.tight_layout(); st.pyplot(fig_cb, use_container_width=True); plt.close()

        # ── Rhythm ───────────────────────────────────────────
        if do_onset and "onset" in Ra_sel and "Rhythm" in tab_map:
            with tab_map["Rhythm"]:
                ta_v = Ra_sel.get("tempo", 120); tb_v = Rb_sel.get("tempo", 120)
                ra = Ra_sel.get("beat_reg", 0); rb = Rb_sel.get("beat_reg", 0)
                td = tb_v - ta_v
                if abs(td) < 1: insight_card("🎯", "Tempo: identical", f"Both at {ta_v:.1f} BPM.")
                elif abs(td) < 3: insight_card("🎯", f"Tempo: nearly same (Δ{td:+.1f} BPM)", f"A:{ta_v:.1f} B:{tb_v:.1f}")
                else:
                    faster = name_b_sel if td > 0 else name_a_sel
                    insight_card("⚡", f"Tempo: {faster} is {abs(td):.1f} BPM faster", f"A:{ta_v:.1f} · B:{tb_v:.1f} BPM")
                c1, c2 = st.columns(2)
                with c1:
                    st.metric(f"Tempo {name_a_sel}", f"{ta_v} BPM")
                    feel_a = "Locked in 🟢" if ra < 1.5 else "Fairly steady 🟡" if ra < 3 else "Loose 🔴"
                    st.markdown(f"**Feel:** {feel_a}  \nBeat regularity: `{ra:.3f}`")
                with c2:
                    st.metric(f"Tempo {name_b_sel}", f"{tb_v} BPM", delta=f"{td:+.1f} BPM")
                    feel_b = "Locked in 🟢" if rb < 1.5 else "Fairly steady 🟡" if rb < 3 else "Loose 🔴"
                    st.markdown(f"**Feel:** {feel_b}  \nBeat regularity: `{rb:.3f}`")
                if ra > 0 and rb > 0:
                    max_reg = max(ra, rb, 0.001)
                    ta_t = int((1-ra/max_reg)*10); tb_t = int((1-rb/max_reg)*10)
                    st.markdown(f"**Tightness:** {name_a_sel}: `{'█'*ta_t+'░'*(10-ta_t)}` {ta_t}/10 · "
                               f"{name_b_sel}: `{'█'*tb_t+'░'*(10-tb_t)}` {tb_t}/10")
                oa = cl1(Ra_sel["onset"], fs, fe); ob = cl1(Rb_sel["onset"], fs, fe)
                n_on = min(len(oa), len(ob)); oa, ob = oa[:n_on], ob[:n_on]
                t_on = np.linspace(zoom_start, zoom_end, n_on)
                diff_formula(name_a_sel, name_b_sel, rec_color(pi), rec_color(pj))
                fig_on, axes_on = plt.subplots(3, 1, figsize=(14, 6), sharex=True)
                for ax, onset, R_src, nm, col in [(axes_on[0], oa, Ra_sel, name_a_sel, rec_color(pi)),
                                                    (axes_on[1], ob, Rb_sel, name_b_sel, rec_color(pj))]:
                    ax.fill_between(t_on, onset, alpha=0.3, color=col)
                    ax.set_title(f"{nm} — {R_src.get('tempo', 120)} BPM", fontsize=8, color=col, fontweight='bold')
                    ax.set_ylabel("Onset", fontsize=7); ax.set_yticks([])
                    if "beats" in R_src:
                        bt = np.array(R_src["beats"])*HOP/SR; bt = bt[(bt >= zoom_start) & (bt <= zoom_end)]
                        if len(bt): ax.scatter(bt, np.interp(bt, t_on, onset), color=col, s=30, zorder=5)
                diff_on = ob - oa
                axes_on[2].fill_between(t_on, diff_on, where=diff_on >= 0, color=rec_color(pj), alpha=0.55)
                axes_on[2].fill_between(t_on, diff_on, where=diff_on < 0, color=rec_color(pi), alpha=0.55)
                axes_on[2].axhline(0, color='black', lw=1.2, zorder=5)
                don_max = max(abs(diff_on.max()), abs(diff_on.min()), 1e-8)
                axes_on[2].set_ylim(-don_max*1.15, don_max*1.15)
                axes_on[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.2g}"))
                axes_on[2].set_xlabel("Time (s)", fontsize=7)
                axes_on[2].set_title(f"Δ = {name_b_sel} minus {name_a_sel}", fontsize=8)
                plt.tight_layout(); st.pyplot(fig_on, use_container_width=True); plt.close()

        # ── Timbre ───────────────────────────────────────────
        if do_mfcc and "mfcc_means" in Ra_sel and "Timbre" in tab_map:
            with tab_map["Timbre"]:
                ma_raw = np.array(Ra_sel["mfcc_means"]); mb_raw = np.array(Rb_sel.get("mfcc_means", ma_raw))
                mc_ = float(np.dot(ma_raw, mb_raw)/(np.linalg.norm(ma_raw)*np.linalg.norm(mb_raw)+1e-10))
                insight_card("🎨", f"Timbre similarity: {(mc_+1)/2*100:.1f}%",
                    "MFCCs capture the timbral fingerprint. Coeff 1–4 = broad spectral shape. 5–12 = fine texture.")
                mfcc_view = st.radio("View", ["Raw values", "Normalized per coefficient"], horizontal=True, key=f"mfcc_view_{pi}_{pj}")
                if mfcc_view == "Raw values":
                    ma_m = ma_raw[1:]; mb_m = mb_raw[1:]; ylabel = "Mean MFCC value"
                    caption = "Raw values. Coefficients 1–4 are large (broad spectral envelope). 5–12 are small — expected."
                else:
                    ma_m = np.array(Ra_sel.get("mfcc_means_z", ma_raw))[1:]
                    mb_m = np.array(Rb_sel.get("mfcc_means_z", mb_raw))[1:]; ylabel = "Normalized [-1,+1]"
                    caption = "Each coefficient scaled to its own [-1,+1] range. All 12 now visually comparable."
                x = np.arange(1, len(ma_m)+1)
                fig_m, axes_m = plt.subplots(2, 1, figsize=(14, 5))
                axes_m[0].bar(x-0.2, ma_m, 0.35, label=name_a_sel, color=rec_color(pi), alpha=0.85)
                axes_m[0].bar(x+0.2, mb_m, 0.35, label=name_b_sel, color=rec_color(pj), alpha=0.85)
                axes_m[0].axhline(0, color='black', lw=1.2, zorder=5)
                axes_m[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.2f}"))
                axes_m[0].set_xticks(x); axes_m[0].set_xlabel("Coefficient index", fontsize=8)
                axes_m[0].set_title(f"MFCC Profile — {mfcc_view}", fontweight='bold', fontsize=9); axes_m[0].legend()
                dm = mb_m - ma_m
                axes_m[1].bar(x, dm, color=[rec_color(pj) if d > 0 else rec_color(pi) for d in dm], alpha=0.85)
                axes_m[1].axhline(0, color='black', lw=1.2, zorder=5)
                dm_max = max(abs(dm.max()), abs(dm.min()), 1e-8)
                axes_m[1].set_ylim(-dm_max*1.2, dm_max*1.2)
                axes_m[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.2f}"))
                axes_m[1].set_xticks(x); axes_m[1].set_xlabel("Coefficient index", fontsize=8)
                n_b_h = sum(1 for d in dm if d > 0); n_a_h = sum(1 for d in dm if d < 0)
                dominant = name_b_sel if n_b_h > n_a_h else name_a_sel
                top_c = int(np.argmax(np.abs(dm)))+1
                axes_m[1].set_title(f"Δ = {name_b_sel} minus {name_a_sel} — above zero = {name_b_sel} higher", fontsize=8)
                axes_m[1].annotate(f"largest diff (coeff {top_c})", xy=(top_c, dm[top_c-1]),
                    xytext=(min(top_c+1.5, x[-1]), dm[top_c-1]*1.1 if dm[top_c-1] != 0 else dm_max*0.5),
                    fontsize=6, arrowprops=dict(arrowstyle='->', lw=0.8))
                plt.tight_layout(); st.pyplot(fig_m, use_container_width=True); plt.close()
                st.caption(caption)
                st.caption(f"**{dominant}** dominates {max(n_a_h, n_b_h)}/12 coefficients.")

        # ── Dynamics ─────────────────────────────────────────
        if do_spectral and "centroid" in Ra_sel and "Dynamics" in tab_map:
            with tab_map["Dynamics"]:
                if "rms" in Ra_sel and "rms" in Rb_sel:
                    rms_a_w = cl1(Ra_sel["rms"], fs, fe); rms_b_w = cl1(Rb_sel["rms"], fs, fe)
                    n_r = min(len(rms_a_w), len(rms_b_w)); rms_a_w, rms_b_w = rms_a_w[:n_r], rms_b_w[:n_r]
                    t_r = np.linspace(zoom_start, zoom_end, n_r)
                    mean_a = float(rms_a_w.mean()); mean_b = float(rms_b_w.mean())
                    loud_pct = (mean_b-mean_a)/(mean_a+1e-10)*100
                    if abs(loud_pct) < 5: loud_s = "Both at similar volume."
                    elif loud_pct > 0: loud_s = f"{name_b_sel} is {abs(loud_pct):.0f}% louder."
                    else: loud_s = f"{name_a_sel} is {abs(loud_pct):.0f}% louder."
                    insight_card("🔊", loud_s, "Volume difference in the selected window.")
                    peak = max(rms_a_w.max(), rms_b_w.max(), 1e-8)
                    rms_a_n = rms_a_w/peak; rms_b_n = rms_b_w/peak
                    fig_e, axes_e = plt.subplots(2, 1, figsize=(14, 4), sharex=True)
                    axes_e[0].fill_between(t_r, rms_a_n, alpha=0.35, color=rec_color(pi), label=name_a_sel)
                    axes_e[0].fill_between(t_r, rms_b_n, alpha=0.35, color=rec_color(pj), label=name_b_sel)
                    axes_e[0].set_ylim(0, 1.05); axes_e[0].set_yticks([0, 0.5, 1.0])
                    axes_e[0].set_yticklabels(["Quiet", "", "Loud"], fontsize=8)
                    axes_e[0].set_title("Volume over time", fontsize=9, fontweight='bold'); axes_e[0].legend(fontsize=8)
                    diff_e = rms_b_n - rms_a_n
                    axes_e[1].fill_between(t_r, diff_e, where=diff_e >= 0, color=rec_color(pj), alpha=0.6, label=f"{name_b_sel} louder")
                    axes_e[1].fill_between(t_r, diff_e, where=diff_e < 0, color=rec_color(pi), alpha=0.6, label=f"{name_a_sel} louder")
                    axes_e[1].axhline(0, color='black', lw=1.2, zorder=5)
                    de_max = max(abs(diff_e.max()), abs(diff_e.min()), 1e-8)
                    axes_e[1].set_ylim(-de_max*1.15, de_max*1.15)
                    axes_e[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.2f}"))
                    axes_e[1].set_xlabel("Time (s)", fontsize=8)
                    axes_e[1].set_title(f"Δ = {name_b_sel} minus {name_a_sel}", fontsize=9); axes_e[1].legend(fontsize=8)
                    plt.tight_layout(); st.pyplot(fig_e, use_container_width=True); plt.close()
                st.markdown("---"); st.markdown("**Spectral descriptors:**")
                for feat, ylabel, yscale, note in [
                    ("centroid", "Hz", "linear", "Brightness — centre of spectral mass."),
                    ("bandwidth", "Hz", "linear", "Bandwidth — harmonic richness."),
                    ("rolloff", "Hz", "linear", "Rolloff — high-frequency content."),
                    ("flatness", "", "log", "Flatness — 0=tonal, 1=noise-like."),
                    ("rms", "", "linear", "RMS energy — loudness."),
                ]:
                    if feat not in Ra_sel or feat not in Rb_sel: continue
                    da = cl1(Ra_sel[feat], fs, fe); db = cl1(Rb_sel[feat], fs, fe)
                    yr = Ra_sel.get(f"{feat}_range")
                    yr_b = Rb_sel.get(f"{feat}_range")
                    if yr is not None and yr_b is not None:
                        yr = (min(yr[0], yr_b[0]), max(yr[1], yr_b[1]))
                    diff_formula(name_a_sel, name_b_sel, rec_color(pi), rec_color(pj))
                    st.pyplot(line_pair(note, da, db, name_a_sel, name_b_sel, ylabel=ylabel, yrange=yr,
                                        yscale=yscale, col_a=rec_color(pi), col_b=rec_color(pj))); plt.close()

        # ── Spectrograms ─────────────────────────────────────
        if has_specs and "Spectrograms" in tab_map:
            with tab_map["Spectrograms"]:
                spec_sub_names = []
                if do_mel and "mel" in Ra_sel: spec_sub_names.append("Mel")
                if do_cqt and "cqt" in Ra_sel: spec_sub_names.append("CQT")
                if spec_sub_names:
                    spec_sub_tabs = st.tabs(spec_sub_names)
                    spec_sub_map = {n: t for n, t in zip(spec_sub_names, spec_sub_tabs)}
                    if "Mel" in spec_sub_map:
                        with spec_sub_map["Mel"]:
                            insight_card("🌊", "Mel Spectrogram", "128 Mel bands, x-axis in seconds. "
                                        "Bottom=bass, top=treble. Orange=B louder, blue=A louder.")
                            diff_formula(name_a_sel, name_b_sel, rec_color(pi), rec_color(pj))
                            ma = cl2(Ra_sel["mel"], fs, fe); mb = cl2(Rb_sel.get("mel", Ra_sel["mel"]), fs, fe)
                            nc = min(ma.shape[1], mb.shape[1])
                            st.pyplot(spec_pair("Mel Spectrogram", ma[:, :nc], mb[:, :nc], name_a_sel, name_b_sel,
                                cmap="inferno", ylabel="Mel band", add_mel_bands=True, col_a=rec_color(pi),
                                col_b=rec_color(pj), cmp_label="DTW-aligned")); plt.close()
                    if "CQT" in spec_sub_map:
                        with spec_sub_map["CQT"]:
                            insight_card("🎵", "CQT", "Semitone-aligned. Y-axis = MIDI note names. "
                                        "Differences at specific note rows show which pitches diverge.")
                            diff_formula(name_a_sel, name_b_sel, rec_color(pi), rec_color(pj))
                            ma = cl2(Ra_sel["cqt"], fs, fe); mb = cl2(Rb_sel.get("cqt", Ra_sel["cqt"]), fs, fe)
                            nc = min(ma.shape[1], mb.shape[1])
                            yticks = {b: cqt_bin_to_note(b) for b in range(0, N_CQT, 12)}
                            st.pyplot(spec_pair("CQT", ma[:, :nc], mb[:, :nc], name_a_sel, name_b_sel, cmap="viridis",
                                ylabel="Note", yticks=yticks, col_a=rec_color(pi), col_b=rec_color(pj),
                                cmp_label="DTW-aligned")); plt.close()

        # ── Score Alignment ──────────────────────────────────
        if do_chroma and "Score Alignment" in tab_map:
            with tab_map["Score Alignment"]:
                render_score_alignment(rec_ids, names)

        # ── Advanced ─────────────────────────────────────────
        if adv and "Advanced" in tab_map:
            with tab_map["Advanced"]:
                adv_tabs = st.tabs(adv); adv_map = {n: t for n, t in zip(adv, adv_tabs)}
                if "SMS/HPSS" in adv_map and "sms_h" in Ra_sel:
                    with adv_map["SMS/HPSS"]:
                        ha = float(np.mean(Ra_sel["sms_ratio"])) if "sms_ratio" in Ra_sel else 0
                        hb = float(np.mean(Rb_sel["sms_ratio"])) if "sms_ratio" in Rb_sel else 0
                        insight_card("🎼", "Harmonic/Percussive separation",
                            f"Ratio: {name_a_sel}={ha:.2f} · {name_b_sel}={hb:.2f}  (0=rhythmic, 1=melodic)")
                        for key, title in [("sms_h", "Harmonic"), ("sms_p", "Percussive")]:
                            if key in Ra_sel and key in Rb_sel:
                                diff_formula(name_a_sel, name_b_sel, rec_color(pi), rec_color(pj))
                                ma = cl2(Ra_sel[key], fs, fe); mb = cl2(Rb_sel[key], fs, fe)
                                nc = min(ma.shape[1], mb.shape[1])
                                st.pyplot(spec_pair(title, ma[:, :nc], mb[:, :nc], name_a_sel, name_b_sel,
                                    cmap="magma", ylabel="Freq bin", col_a=rec_color(pi), col_b=rec_color(pj),
                                    cmp_label="DTW-aligned")); plt.close()
                        st.caption("Harmonic/Percussive playback omitted in the thin client — "
                                  "only the pYIN transcription and original-file playback round trip "
                                  "through the backend today (see Score Alignment tab).")
                if "Reverb" in adv_map and "rt60" in Ra_sel and "rt60" in Rb_sel:
                    with adv_map["Reverb"]:
                        diff_rt = Rb_sel["rt60"]-Ra_sel["rt60"]
                        insight_card("🏠", "Room acoustics",
                            f"RT60: {name_a_sel}={Ra_sel['rt60']:.3f}s · {name_b_sel}={Rb_sel['rt60']:.3f}s · Δ={diff_rt:+.3f}s")
                        fig_r, ax_r = plt.subplots(figsize=(14, 3))
                        ax_r.plot(np.linspace(0, durs[pi], len(Ra_sel["decay"])), Ra_sel["decay"], label=name_a_sel, lw=0.8, color=rec_color(pi))
                        ax_r.plot(np.linspace(0, durs[pj], len(Rb_sel["decay"])), Rb_sel["decay"], label=name_b_sel, lw=0.8, color=rec_color(pj))
                        ax_r.axhline(-60, color='red', ls='--', lw=0.8, label='−60 dB')
                        ax_r.legend(fontsize=7); ax_r.set_xlabel("Time (s)"); ax_r.set_ylabel("dB")
                        plt.tight_layout(); st.pyplot(fig_r, use_container_width=True); plt.close()
                if "STFT" in adv_map and "stft" in Ra_sel:
                    with adv_map["STFT"]:
                        diff_formula(name_a_sel, name_b_sel, rec_color(pi), rec_color(pj))
                        ma = cl2(Ra_sel["stft"], fs, fe); mb = cl2(Rb_sel.get("stft", Ra_sel["stft"]), fs, fe)
                        nc = min(ma.shape[1], mb.shape[1])
                        st.pyplot(spec_pair("STFT", ma[:, :nc], mb[:, :nc], name_a_sel, name_b_sel, cmap="magma",
                            ylabel="Freq bin", col_a=rec_color(pi), col_b=rec_color(pj))); plt.close()
                if "CWT" in adv_map and "cwt" in Ra_sel:
                    with adv_map["CWT"]:
                        from scipy.ndimage import zoom as nd_zoom
                        ca_cw = Ra_sel["cwt"]; cb_cw = Rb_sel.get("cwt", ca_cw)
                        if cb_cw is not None and cb_cw.shape[1] != ca_cw.shape[1]:
                            cb_cw = nd_zoom(cb_cw.astype(np.float32), (1, ca_cw.shape[1]/cb_cw.shape[1]), order=1)
                        n_c = ca_cw.shape[1]; cz_s = int(zoom_start/durs[pi]*n_c); cz_e = int(zoom_end/durs[pi]*n_c)
                        ma = ca_cw[:, cz_s:cz_e]; mb = (cb_cw[:, cz_s:cz_e] if cb_cw is not None else ma)
                        nc = min(ma.shape[1], mb.shape[1])
                        diff_formula(name_a_sel, name_b_sel, rec_color(pi), rec_color(pj))
                        st.pyplot(spec_pair("CWT", ma[:, :nc], mb[:, :nc], name_a_sel, name_b_sel, cmap="plasma",
                            ylabel="Scale", col_a=rec_color(pi), col_b=rec_color(pj))); plt.close()
                if "Gammatone" in adv_map and "gam" in Ra_sel:
                    with adv_map["Gammatone"]:
                        ma = cl2(Ra_sel["gam"], fs, fe); mb = cl2(Rb_sel.get("gam", Ra_sel["gam"]), fs, fe)
                        nc = min(ma.shape[1], mb.shape[1])
                        diff_formula(name_a_sel, name_b_sel, rec_color(pi), rec_color(pj))
                        st.pyplot(spec_pair("Gammatone", ma[:, :nc], mb[:, :nc], name_a_sel, name_b_sel, cmap="hot",
                            ylabel="ERB filter", col_a=rec_color(pi), col_b=rec_color(pj))); plt.close()
                if "Tonnetz" in adv_map and "tonnetz" in Ra_sel:
                    with adv_map["Tonnetz"]:
                        ta_w = cl2(Ra_sel["tonnetz"], fs, fe); tb_w = cl2(Rb_sel.get("tonnetz", Ra_sel["tonnetz"]), fs, fe)
                        nc = min(ta_w.shape[1], tb_w.shape[1]); ta_m = ta_w[:, :nc].mean(1); tb_m = tb_w[:, :nc].mean(1)
                        dist = float(np.linalg.norm(ta_m-tb_m))
                        insight_card("🔮", "Tonnetz", f"L2 distance: {dist:.4f}. "
                                    f"{'Same harmonic region.' if dist < 0.2 else 'Different tonal centre.'}")
                        dim_labels = ["Fifths 1", "Fifths 2", "Maj 3rd 1", "Maj 3rd 2", "Min 3rd 1", "Min 3rd 2"]
                        fig_t, ax_t = plt.subplots(figsize=(10, 3)); x = np.arange(6)
                        ax_t.bar(x-0.2, ta_m, 0.35, label=name_a_sel, color=rec_color(pi), alpha=0.85)
                        ax_t.bar(x+0.2, tb_m, 0.35, label=name_b_sel, color=rec_color(pj), alpha=0.85)
                        ax_t.axhline(0, color='black', lw=1.2, zorder=5)
                        ax_t.set_xticks(x); ax_t.set_xticklabels(dim_labels, fontsize=8); ax_t.legend()
                        plt.tight_layout(); st.pyplot(fig_t, use_container_width=True); plt.close()

        # ── N-recording all-features overlay ──────────────────
        if n_recs > 2:
            st.markdown("---")
            st.subheader("All Rehearsals — Feature Overlay")
            st.caption("All recordings on the same axes for each feature. Each recording keeps its color throughout.")
            overlay_feats = [("rms", "Volume (RMS)"), ("centroid", "Brightness (Centroid Hz)"),
                             ("onset", "Onset Strength"), ("sms_ratio", "Harmonic Ratio")]
            try:
                all_aligned = [api.get_aligned_pair(rec_ids[0], rid)[1] if k > 0 else Ra_sel
                              for k, rid in enumerate(rec_ids)]
            except BackendError:
                all_aligned = None
            if all_aligned:
                avail = [f for f in overlay_feats if all(f[0] in R for R in all_aligned)]
                if avail:
                    n_f = len(avail)
                    fig_ov, axes_ov = plt.subplots(n_f, 1, figsize=(14, 2.5*n_f), sharex=True)
                    if n_f == 1: axes_ov = [axes_ov]
                    for ax, (feat, label) in zip(axes_ov, avail):
                        for idx, (R, nm) in enumerate(zip(all_aligned, names)):
                            arr = cl1(np.array(R[feat]), fs, fe)
                            if len(arr) == 0: continue
                            ax.plot(np.linspace(zoom_start, zoom_end, len(arr)), arr, lw=0.9, color=rec_color(idx), label=nm, alpha=0.85)
                        ax.set_ylabel(label, fontsize=7); ax.legend(fontsize=7, loc="upper right")
                    axes_ov[-1].set_xlabel("Time (s)", fontsize=8)
                    plt.suptitle(f"All {n_recs} Rehearsals — {zoom_start:.1f}–{zoom_end:.1f}s", fontsize=10, fontweight='bold')
                    plt.tight_layout(); st.pyplot(fig_ov, use_container_width=True); plt.close()
                else:
                    st.info("Enable Spectral Features and Onset for overlay plots.")

    if pair_tabs is not None:
        for pair_tab, (pi, pj) in zip(pair_tabs, pairs):
            with pair_tab:
                _render_pair(pi, pj)
    else:
        _render_pair(0, 1)

    # ── What Changed card (bottom) ──────────────────────────
    best_pi, best_pj = pairs[best_idx]
    try:
        best_sim = api.get_similarity(rec_ids[best_pi], rec_ids[best_pj], names[best_pi], names[best_pj])
        st.markdown("---")
        with st.expander("💡 What Changed? — Quick Summary", expanded=False):
            st.caption(f"Based on auto-selected pair: **{names[best_pi]} vs {names[best_pj]}**")
            for ins in best_sim["insights"]:
                st.markdown(f"**{ins['icon']} {ins['headline']}**")
                st.caption(ins["detail"])
            st.caption("_Rule-based · swap `generate_summary()` (compute.py) for an LLM call_")
    except BackendError:
        pass

# ─────────────────────────────────────────────────────────────
# Downloads section
# ─────────────────────────────────────────────────────────────
with main_col:
    st.markdown("---")
    st.subheader("⬇️ Downloads")

    st.markdown("**Progress Report**")
    st.caption(
        "Generates a markdown report from the same data already fetched for the "
        "interactive tabs (similarity, key/mode/tempo, insights) — a lighter-weight "
        "summary than the original's fully-illustrated report, which re-rendered "
        "every figure as an embedded image. Flagged simplification: figure-heavy "
        "PDF/HTML export isn't wired up in the thin client yet; this covers the "
        "narrative content losslessly.")

    if st.button("Generate Report", key="gen_report"):
        with st.spinner("Assembling report..."):
            best_pi, best_pj = pairs[best_idx]
            lines = [f"# ViSuS Report", "", f"Recordings: {', '.join(names)}", ""]
            for nm, R in zip(names, Rs):
                lines.append(f"- **{nm}**: {R.get('tempo','—')} BPM · key {R.get('key','—')} · "
                             f"mode {R.get('mode','—')} · {R.get('dur',0):.1f}s")
            lines.append("")
            lines.append("## Pairwise similarity")
            for (i, j) in pairs:
                try:
                    sim = api.get_similarity(rec_ids[i], rec_ids[j], names[i], names[j])
                except BackendError:
                    continue
                lines.append(f"### {names[i]} vs {names[j]} — {sim['overall_similarity']*100:.0f}% similar")
                lines.append(f"*ESM verdict: {sim['esm_verdict']['text']}*")
                for k, v in sim["sub_scores"].items():
                    lines.append(f"- {k}: {v*100:.0f}%")
                for ins in sim["insights"]:
                    lines.append(f"- {ins['icon']} **{ins['headline']}** — {ins['detail']}")
                lines.append("")
            st.session_state["report_md"] = "\n".join(lines)
        st.success("Report ready — download below.")

    if st.session_state.get("report_md"):
        best_pi, best_pj = pairs[best_idx]
        fname_base = f"visus_report_{names[best_pi][:8]}_{names[best_pj][:8]}"
        st.download_button(
            label="⬇️ Download as Markdown (.md)", data=st.session_state["report_md"],
            file_name=f"{fname_base}.md", mime="text/markdown", key="dl_report_md", use_container_width=True)
        with st.expander("Preview"):
            st.markdown(st.session_state["report_md"])

    st.markdown("---")
    st.markdown("**MIDI Files**")
    st.caption("Generate MIDI transcriptions from each recording using pYIN pitch detection. "
              "Download individual files or the consensus score (average of all transcriptions).")

    if st.button("Generate MIDI Files", key="gen_midi"):
        with st.spinner("Transcribing recordings to MIDI via pYIN..."):
            try:
                sa_dl = api.score_alignment(rec_ids)
                st.session_state["midi_dl"] = sa_dl
                st.session_state["midi_ready"] = True
            except BackendError as e:
                st.error(f"MIDI generation failed: {e}")
                st.session_state["midi_ready"] = False

    if st.session_state.get("midi_ready"):
        sa_dl = st.session_state["midi_dl"]
        midi_cols = st.columns(min(n_recs, 4))
        for col, rid, nm in zip(midi_cols, rec_ids, names):
            try:
                midi_bytes = api.download_transcription_midi(rid)
                fname_midi = nm.replace(" ", "_")[:20] + ".mid"
                col.download_button(label=f"⬇️ {nm[:15]}", data=midi_bytes, file_name=fname_midi,
                                    mime="audio/midi", key=f"dl_midi_{rid}", use_container_width=True)
            except BackendError as e:
                col.caption(f"Failed: {e}")
        try:
            avg_tempo = float(np.mean([R.get("tempo", 120) for R in Rs]))
            cons_bytes = api.download_chroma_as_midi(sa_dl["consensus_chroma"], tempo=avg_tempo)
            st.download_button(label="⬇️ Download Consensus Score (.mid)", data=cons_bytes,
                               file_name="visus_consensus_score.mid", mime="audio/midi", key="dl_midi_consensus")
            st.caption(f"Consensus = average of all {n_recs} MIDI transcriptions · {avg_tempo:.0f} BPM avg. "
                      "Import into any DAW or notation software.")
        except BackendError as e:
            st.caption(f"Consensus MIDI failed: {e}")

st.caption("ViSuS · Music Signal Analysis · HIWI Research Demo · thin client")
