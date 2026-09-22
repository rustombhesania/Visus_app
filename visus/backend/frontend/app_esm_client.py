"""
ViSuS — Everybody Speaks Music (ESM), thin client.

The limited-feature persona for musicians, not audio engineers. Per
Rustom's direction: show simple percentages and musician-relevant
visuals (the stream graph, harmony/rhythm/timbre/dynamics summaries),
but never a raw spectrogram/heatmap image -- "musicians don't
understand anything, I want stuff that musicians actually want."
Every percentage badge is tap/click-to-expand for the full plain-
language explanation (from compute.generate_summary()'s (icon,
headline, detail) triples, and short doc-strings for anything else).

What's shown, and why each thing cleared or missed the bar:
  - Overview insight cards        -- kept (plain language by construction)
  - Song Map stream graph          -- kept, explicitly requested
  - Song Map section breakdown     -- kept (traffic-light verdicts, no numbers forced)
  - Harmony: % + pitch-class bars  -- kept (a bar chart is not a spectrogram)
  - Rhythm: % + onset fill plot    -- kept
  - Timbre: % + MFCC bar chart     -- kept
  - Dynamics: % + volume fill plot -- kept
  - Mel/CQT spectrograms           -- DROPPED, this is the thing being avoided
  - DTW warping-path plots         -- DROPPED (technical alignment internals)
  - Advanced tab (STFT/CWT/etc.)   -- DROPPED (all spectrogram-family)
  - Score/MIDI alignment tab       -- DROPPED (chroma heatmaps)
  - Song Map stacked-bar view      -- DROPPED (redundant with stream graph +
    section breakdown; flagged simplification, easy to add back if missed)

N takes: upload 2-8 recordings. With exactly 2, the detail view (verdict,
stream graph, harmony/rhythm/timbre/dynamics) renders directly -- same
as before. With more than 2, every pairwise verdict is computed (cheap:
just api.get_similarity(), no charts) and shown as a plain vertical list
sorted best-to-worst, with an auto-flagged closest pair and outlier take
at the top. A dropdown picks which single pair to open the full detail
view for -- the chart-heavy detail work only ever runs for one pair at a
time, never for all of them at once, so an 8-take / 28-pair batch stays
fast instead of rendering ~140 matplotlib figures on every run.

Note on the dropdown-vs-tabs call: decisions.md flags "tabs, not
dropdowns" for the Full client, because dropdown reruns used to
re-trigger expensive *local* computation. That constraint doesn't apply
here -- a dropdown change now just re-fetches one pair's data from the
backend and redraws a handful of small charts, which is cheap. Tabs
were also dropped deliberately for mobile: N pairs of horizontally-
scrolling tabs is an awkward touch target; a single native dropdown
isn't.

Mobile-friendly by construction: single centered column, no sidebar,
no multi-column layouts. Small matplotlib figures are used for the
four dimension charts (bar/fill plots only, never a 2D heatmap), which
render as one full-width image per section -- no pinch-zoom needed.

Shares api_client.py with app_full_client.py -- same backend.
Run: streamlit run app_esm_client.py
Needs the backend running first: uvicorn main:app --port 8000 (from backend/)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

import api_client as api
from api_client import BackendError

st.set_page_config(page_title="ViSuS — ESM", layout="centered", initial_sidebar_state="collapsed")

MIDI_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
COL_A, COL_B = "#2196F3", "#FF9800"

TIER_STYLE = {
    "close":     {"color": "#22C55E", "icon": "🟢", "label": "Mostly matched"},
    "drift":     {"color": "#FBBF24", "icon": "🟡", "label": "Some drift"},
    "different": {"color": "#EF4444", "icon": "🔴", "label": "Different feel"},
}


def backend_status_banner():
    if not api.health():
        st.error(
            f"⚠️ Can't reach the backend at {api.BASE_URL}. "
            f"Start it first: `uvicorn main:app --reload --port 8000` "
            f"(from the backend/ folder), then reload this page."
        )
        st.stop()


def pct_badge(label, value, color="#4a9eff"):
    st.markdown(
        f"<div style='display:flex;justify-content:space-between;align-items:center;"
        f"background:#1a1a2e;border-radius:8px;padding:10px 14px;margin-bottom:6px'>"
        f"<span style='font-weight:600'>{label}</span>"
        f"<span style='font-size:1.3rem;font-weight:700;color:{color}'>{value*100:.0f}%</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


st.title("🎵 ViSuS")
st.caption("Everybody Speaks Music — compare your takes in plain language.")
backend_status_banner()

# ─────────────────────────────────────────────────────────────
# Upload — N takes, stacked vertically (no columns -- phone width)
# ─────────────────────────────────────────────────────────────
# Broadened to everything librosa/soundfile/ffmpeg can realistically decode
# (see DEPLOYMENT.md's audio-format note for why some of these need ffmpeg
# on the backend, not just soundfile). Client-side filter only -- the real
# gate is the backend's decode attempt.
AUDIO_TYPES = ["wav", "mp3", "flac", "ogg", "oga", "opus", "m4a", "mp4", "aac",
               "wma", "aiff", "aif", "au", "caf", "wv", "amr", "3gp", "3g2", "webm", "mka"]

n_takes = st.slider("Number of takes", min_value=2, max_value=8, value=2, key="n_takes")
files, names_in = [], []
for i in range(n_takes):
    st.markdown(f"#### Take {i+1}")
    f = st.file_uploader("Upload", type=AUDIO_TYPES, key=f"fu_{i}", label_visibility="collapsed")
    n = st.text_input("Name", value=f"Take {i+1}", key=f"name_{i}", label_visibility="collapsed")
    files.append(f); names_in.append(n)

run_btn = st.button("▶ Compare", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────
# Run — upload all N, then compute every pairwise verdict up front
# (cheap: no charts, just api.get_similarity per pair). Detail charts
# for any one pair are fetched lazily, only when that pair is opened.
# ─────────────────────────────────────────────────────────────
if run_btn:
    if any(f is None for f in files):
        st.warning(f"Upload all {n_takes} takes before comparing.")
    else:
        with st.spinner("Listening..."):
            try:
                recs = []
                for f, nm in zip(files, names_in):
                    f.seek(0)
                    res = api.upload_recording(f.read(), f.name)
                    recs.append({"recording_id": res["recording_id"], "name": nm, "file_bytes": f.getvalue()})

                pairs = [(i, j) for i in range(n_takes) for j in range(i+1, n_takes)]
                pair_sims = []
                for i, j in pairs:
                    sim = api.get_similarity(recs[i]["recording_id"], recs[j]["recording_id"],
                                             recs[i]["name"], recs[j]["name"])
                    pair_sims.append(sim)
            except BackendError as e:
                st.error(f"Comparison failed: {e}")
                st.stop()
        st.session_state["esm_recs"] = recs
        st.session_state["esm_pairs"] = pairs
        st.session_state["esm_pair_sims"] = pair_sims

# ─────────────────────────────────────────────────────────────
# Detail view for one pair — fetches chart-level data on demand.
# ─────────────────────────────────────────────────────────────
def render_pair_detail(sim, a_id, b_id, na, nb, file_a, file_b):
    tier = sim["esm_verdict"]["tier"]; text = sim["esm_verdict"]["text"]
    style = TIER_STYLE.get(tier, {"color": "#888", "icon": "⚪", "label": text})

    st.markdown(
        f"<div style='background:{style['color']}22;border:2px solid {style['color']};"
        f"border-radius:14px;padding:28px 20px;text-align:center;margin-bottom:16px'>"
        f"<div style='font-size:2.6rem'>{style['icon']}</div>"
        f"<div style='font-size:1.4rem;font-weight:700;color:{style['color']};margin-top:4px'>"
        f"{style['label']}</div>"
        f"<div style='font-size:1rem;color:#888;margin-top:6px'>{text}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    for nm, info, fb in [(na, sim["recording_a"], file_a), (nb, sim["recording_b"], file_b)]:
        st.markdown(
            f"<div style='background:#1a1a2e;border-radius:10px;padding:12px 16px;margin-bottom:8px'>"
            f"<div style='font-weight:700'>{nm}</div>"
            f"<div style='color:#aaa;font-size:0.9rem'>{info.get('tempo','—')} BPM · {info.get('mode','—')}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.audio(fb)

    st.markdown("### 💡 What stands out")
    for ins in sim["insights"]:
        with st.expander(f"{ins['icon']} {ins['headline']}"):
            st.write(ins["detail"])

    try:
        Ra, Rb = api.get_aligned_pair(a_id, b_id)
    except BackendError as e:
        st.error(f"Could not load detail: {e}"); return

    sm, secs = None, None
    try:
        sm = api.get_song_map(a_id, b_id)
        secs = api.get_sections(a_id, b_id)
    except BackendError:
        pass  # not enough overlapping features for this pair -- skip Song Map silently

    if sm is not None:
        st.markdown("### 🌊 Where they diverge")
        bands, colors = sm["bands"], sm["colors"]
        if bands:
            min_len = min(len(v) for v in bands.values())
            times_s = np.array(sm["time"][:min_len])
            labels_s = list(bands.keys())
            vals_s = np.array([bands[l][:min_len] for l in labels_s])
            cols_s = [colors[l] for l in labels_s]
            fig, ax = plt.subplots(figsize=(7, 3.2))
            ax.axhline(0, color='#888', lw=0.8, alpha=0.5)
            cum_pos = np.zeros(min_len)
            for label, val_sc, col in zip(labels_s, vals_s, cols_s):
                upper = cum_pos + val_sc
                ax.fill_between(times_s, cum_pos, upper, color=col, alpha=0.85, label=label)
                cum_pos = upper
            cum_neg = np.zeros(min_len)
            for val_sc, col in zip(vals_s, cols_s):
                lower = cum_neg - val_sc
                ax.fill_between(times_s, lower, cum_neg, color=col, alpha=0.85)
                cum_neg = lower
            ax.set_ylim(-0.55, 0.55); ax.set_yticks([])
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.legend(fontsize=7, loc='upper right', ncol=2)
            for sp in ax.spines.values(): sp.set_visible(False)
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
            with st.expander("ℹ️ What does this show?"):
                st.write(
                    "Each coloured band is one musical ingredient — Volume, Brightness, "
                    "Rhythm, Harmony. A wider band at a given moment means that ingredient "
                    "is driving the difference between the two takes more at that point. "
                    "A thin, flat ribbon means the takes agreed closely there.")
                moments = sm.get("top_divergent_moments") or []
                if moments:
                    st.write("**Most different moments:**")
                    for m in moments:
                        pt = m["time"]
                        st.write(f"- {int(pt//60)}:{int(pt%60):02d} — driven by **{m['dominant_band']}**")

        if secs:
            st.markdown("### 📋 Section by section")
            VERDICT_ICON = {"same": "🟢", "slight": "🟡", "noticeable": "🔴"}
            for b0, b1, t0, t1, lbl, cmp in secs:
                if not cmp: continue
                verdicts = [v for v, _ in cmp.values()]
                overall = ("🟢" if all(v == "same" for v in verdicts)
                           else "🔴" if any(v == "noticeable" for v in verdicts) else "🟡")
                with st.expander(f"{overall} **{lbl}** — {t0:.0f}–{t1:.0f}s"):
                    for dim, (verdict, detail) in cmp.items():
                        st.write(f"{VERDICT_ICON.get(verdict,'⚪')} **{dim}** — {detail}")

    if "chroma" in Ra and "chroma" in Rb:
        st.markdown("### 🎹 Harmony")
        pct_badge("Harmonic similarity", sim["sub_scores"].get("Harmony", 0), COL_A)
        ca = np.array(Ra["chroma"]).mean(1); cb = np.array(Rb["chroma"]).mean(1)
        x = np.arange(12)
        fig, ax = plt.subplots(figsize=(7, 2.6))
        ax.bar(x-0.2, ca, 0.35, label=na, color=COL_A, alpha=0.85)
        ax.bar(x+0.2, cb, 0.35, label=nb, color=COL_B, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(MIDI_NAMES, fontsize=8); ax.legend(fontsize=8)
        for sp in ax.spines.values(): sp.set_visible(False)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
        with st.expander("ℹ️ What does this mean?"):
            top3_a = sorted(range(12), key=lambda i: -ca[i])[:3]
            top3_b = sorted(range(12), key=lambda i: -cb[i])[:3]
            st.write(f"**{na}** leans on: {', '.join(MIDI_NAMES[i] for i in top3_a)}")
            st.write(f"**{nb}** leans on: {', '.join(MIDI_NAMES[i] for i in top3_b)}")
            st.write(f"Key: {Ra.get('key','—')} ({Ra.get('mode','—')}) vs {Rb.get('key','—')} ({Rb.get('mode','—')})")
            st.write(
                "This compares which notes each take emphasizes overall. Similar bar "
                "heights across the same notes means the two takes agree harmonically, "
                "even if the exact timing or tone differs.")

    if "onset" in Ra and "onset" in Rb:
        st.markdown("### 🥁 Rhythm")
        pct_badge("Rhythm similarity", sim["sub_scores"].get("Rhythm", 0), COL_A)
        ta_v, tb_v = Ra.get("tempo", 120), Rb.get("tempo", 120)
        c1, c2 = st.columns(2)
        c1.metric(na, f"{ta_v} BPM")
        c2.metric(nb, f"{tb_v} BPM", f"{tb_v-ta_v:+.1f}")
        oa = np.array(Ra["onset"]); ob = np.array(Rb["onset"])
        n = min(len(oa), len(ob))
        t = np.linspace(0, min(Ra.get("dur", 0), Rb.get("dur", 0)), n)
        fig, ax = plt.subplots(figsize=(7, 2.2))
        ax.fill_between(t, oa[:n], alpha=0.4, color=COL_A, label=na)
        ax.fill_between(t, ob[:n], alpha=0.4, color=COL_B, label=nb)
        ax.set_yticks([]); ax.set_xlabel("Time (s)", fontsize=8); ax.legend(fontsize=8)
        for sp in ax.spines.values(): sp.set_visible(False)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
        with st.expander("ℹ️ What does this mean?"):
            ra_r, rb_r = Ra.get("beat_reg", 0), Rb.get("beat_reg", 0)
            feel_a = "locked in" if ra_r < 1.5 else "fairly steady" if ra_r < 3 else "loose"
            feel_b = "locked in" if rb_r < 1.5 else "fairly steady" if rb_r < 3 else "loose"
            st.write(f"**{na}** feels **{feel_a}**. **{nb}** feels **{feel_b}**.")
            st.write(
                "The shaded shape shows where each take has a strong rhythmic attack "
                "(a note or hit starting). Peaks lining up in time means the two takes "
                "are rhythmically together at that moment.")

    if "mfcc_means" in Ra and "mfcc_means" in Rb:
        st.markdown("### 🎨 Tone / Timbre")
        pct_badge("Timbre similarity", sim["sub_scores"].get("Timbre", 0), COL_A)
        ma = np.array(Ra["mfcc_means"])[1:]; mb = np.array(Rb["mfcc_means"])[1:]
        x = np.arange(1, len(ma)+1)
        fig, ax = plt.subplots(figsize=(7, 2.4))
        ax.bar(x-0.2, ma, 0.35, label=na, color=COL_A, alpha=0.85)
        ax.bar(x+0.2, mb, 0.35, label=nb, color=COL_B, alpha=0.85)
        ax.set_xticks(x); ax.set_xlabel("Tone quality (coarse → fine)", fontsize=8); ax.legend(fontsize=8)
        for sp in ax.spines.values(): sp.set_visible(False)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
        with st.expander("ℹ️ What does this mean?"):
            st.write(
                "This is a fingerprint of each take's overall tone — the instrument's "
                "colour and texture, not the notes being played. Similar bar patterns "
                "mean the two takes sound like the same instrument/setup recorded "
                "similarly; different patterns can mean a different mic position, "
                "instrument, or room.")

    if "rms" in Ra and "rms" in Rb:
        st.markdown("### 🔊 Dynamics")
        pct_badge("Brightness similarity", sim["sub_scores"].get("Brightness", 0), COL_A)
        ra_rms = np.array(Ra["rms"]); rb_rms = np.array(Rb["rms"])
        n = min(len(ra_rms), len(rb_rms))
        peak = max(ra_rms.max(), rb_rms.max(), 1e-8)
        t = np.linspace(0, min(Ra.get("dur", 0), Rb.get("dur", 0)), n)
        fig, ax = plt.subplots(figsize=(7, 2.2))
        ax.fill_between(t, ra_rms[:n]/peak, alpha=0.4, color=COL_A, label=na)
        ax.fill_between(t, rb_rms[:n]/peak, alpha=0.4, color=COL_B, label=nb)
        ax.set_ylim(0, 1.05); ax.set_yticks([0, 1]); ax.set_yticklabels(["Quiet", "Loud"], fontsize=8)
        ax.set_xlabel("Time (s)", fontsize=8); ax.legend(fontsize=8)
        for sp in ax.spines.values(): sp.set_visible(False)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
        with st.expander("ℹ️ What does this mean?"):
            mean_a, mean_b = float(ra_rms.mean()), float(rb_rms.mean())
            loud_pct = (mean_b-mean_a)/(mean_a+1e-10)*100
            if abs(loud_pct) < 5:
                st.write(f"**{na}** and **{nb}** are at similar overall volume.")
            else:
                louder = nb if loud_pct > 0 else na
                st.write(f"**{louder}** is noticeably louder overall.")
            st.write(
                "This tracks loudness over time — where each take swells or pulls back. "
                "Shapes that rise and fall together mean the two performances share the "
                "same dynamic arc, even if the exact volume differs.")


# ─────────────────────────────────────────────────────────────
# Result — 2 takes go straight to detail; N>2 gets a summary list
# and a picker first.
# ─────────────────────────────────────────────────────────────
recs = st.session_state.get("esm_recs")
if recs:
    pairs = st.session_state["esm_pairs"]
    pair_sims = st.session_state["esm_pair_sims"]
    st.markdown("---")

    if len(pairs) == 1:
        (i, j) = pairs[0]
        render_pair_detail(pair_sims[0], recs[i]["recording_id"], recs[j]["recording_id"],
                           recs[i]["name"], recs[j]["name"], recs[i]["file_bytes"], recs[j]["file_bytes"])
    else:
        # ── Auto-callout: closest pair + outlier take ─────────
        scores = [s["overall_similarity"] for s in pair_sims]
        best_idx = int(np.argmax(scores)); worst_idx = int(np.argmin(scores))
        bi, bj = pairs[best_idx]; wi, wj = pairs[worst_idx]
        outlier_counts = {k: 0 for k in range(len(recs))}
        for (i, j), s in zip(pairs, scores):
            if s == min(scores):
                outlier_counts[i] += 1; outlier_counts[j] += 1
        outlier_idx = max(outlier_counts, key=outlier_counts.get)
        st.info(
            f"**Closest match:** {recs[bi]['name']} & {recs[bj]['name']}  \n"
            f"**Stands out most:** {recs[outlier_idx]['name']} — differs most from the others."
        )

        # ── Vertical list of all pairs, sorted best → worst ───
        st.markdown("### All comparisons")
        order = sorted(range(len(pairs)), key=lambda k: -scores[k])
        pair_labels = []
        for k in order:
            i, j = pairs[k]
            tier = pair_sims[k]["esm_verdict"]["tier"]
            style = TIER_STYLE.get(tier, {"color": "#888", "icon": "⚪", "label": tier})
            label = f"{recs[i]['name']} vs {recs[j]['name']}"
            pair_labels.append(label)
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;align-items:center;"
                f"background:#1a1a2e;border-radius:8px;padding:10px 14px;margin-bottom:6px'>"
                f"<span>{style['icon']} {label}</span>"
                f"<span style='color:{style['color']};font-weight:600;font-size:0.85rem'>{style['label']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # ── Picker: open the full detail view for one pair ────
        st.markdown("### 🔍 Look closer")
        default_label = f"{recs[bi]['name']} vs {recs[bj]['name']}"
        default_idx = order.index(best_idx)
        chosen_label = st.selectbox("Pick a pair to compare in detail", pair_labels, index=default_idx)
        chosen_k = order[pair_labels.index(chosen_label)]
        i, j = pairs[chosen_k]
        render_pair_detail(pair_sims[chosen_k], recs[i]["recording_id"], recs[j]["recording_id"],
                           recs[i]["name"], recs[j]["name"], recs[i]["file_bytes"], recs[j]["file_bytes"])

    st.markdown("---")
    if st.button("↺ Compare different takes"):
        del st.session_state["esm_recs"]
        del st.session_state["esm_pairs"]
        del st.session_state["esm_pair_sims"]
        st.rerun()
