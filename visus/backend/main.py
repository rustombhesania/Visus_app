"""
ViSuS FastAPI backend -- step 2 of ESM_README.md.

Scope of this file: endpoints 1-5 from VISUS_BACKEND_API_DESIGN.md section 4,
which cover everything the ESM verdict (verdict.py) and the Full persona's
core pairwise views need. Endpoints 6-11 (matrix, MDS, group-view, score
alignment) are stubbed at the bottom -- same shape, not yet wired to compute.

Storage: in-memory only (RECORDINGS dict). Postgres is ESM_README.md step 3,
not done yet -- this file is deliberately just "the API code, separate from
the frontend," per the instruction to get that working before touching
Render or the database. Swapping the in-memory dict for real persistence
later should not require changing any endpoint signature below: every
handler already goes through get_recording()/store_recording() rather than
touching RECORDINGS directly, so that's the one place a DB-backed version
would change.

Run locally: uvicorn main:app --reload --port 8000
"""
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))
import compute
from verdict import generate_esm_verdict

app = FastAPI(title="ViSuS API", version="0.1.0")

# ---------------------------------------------------------------------------
# CORS. Worth being precise about why this is here, since it's easy to
# cargo-cult: api_client.py's requests all happen server-side, inside
# Streamlit's own Python process (Community Cloud runs the whole app in a
# container, not in the visitor's browser), so CORS -- a browser-enforced
# policy on JS-initiated cross-origin fetches -- doesn't actually gate
# anything in the current architecture. It's added anyway because it's
# harmless, and it stops this from being a landmine later: the FastAPI
# /docs Swagger UI (hosted on the backend's own origin, so same-origin --
# fine either way), a future browser-side fetch from a hand-built page, or
# anyone testing the API directly from a different host would otherwise
# hit an opaque browser error with no hint that CORS was the cause.
# ALLOWED_ORIGINS should list the two Streamlit Community Cloud app URLs
# once deployed (see DEPLOYMENT.md) -- "*" is fine for local dev only.
import os
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory store. Keys are recording_id (str). Each value is the raw R dict
# returned by compute.run_analysis(), plus the fields we add ourselves
# (name, uploaded_at). Arrays stay as numpy inside this dict; they're only
# converted to JSON-safe lists at the response boundary (serialize_R), so
# in-process endpoints (alignment, similarity, etc.) keep working with real
# ndarrays and don't pay a round-trip serialization cost internally.
# ---------------------------------------------------------------------------
RECORDINGS: dict[str, dict] = {}

# Default feature set: everything on. Both personas' endpoints (2-5) all
# depend on some subset of these; ESM's own needs (key, mode, tempo,
# similarity) are the cheapest subset, but Full needs the rest, and nothing
# in the API design doc says to compute two different R dicts per recording,
# so one recording is analyzed once, in full, and both personas read from
# the same stored result.
DEFAULT_FEATURE_FLAGS = dict(
    do_mel=True, do_cqt=True, do_mfcc=True, do_chroma=True, do_spectral=True,
    do_onset=True, do_sms=True, do_stft=True, do_cwt=False, do_gammatone=False,
    do_tonnetz=False, do_zcr=True, do_reverb=False,
)


def get_recording(recording_id: str) -> dict:
    R = RECORDINGS.get(recording_id)
    if R is None:
        raise HTTPException(status_code=404, detail=f"Unknown recording_id: {recording_id}")
    return R


def store_recording(name: str, R: dict, y_bytes: bytes) -> str:
    recording_id = str(uuid.uuid4())
    # _audio_bytes: raw float32 PCM (post-decode, pre-analysis) kept alongside
    # the analysis dict so detect_structure() -- and anything else that needs
    # the waveform itself, not just derived features -- works without a
    # database. This is in-memory only, same as everything else here; it's
    # not the Postgres "raw audio as durable source of truth" decision, just
    # what's needed to stop /sections and the song-map endpoint from 501ing
    # while the database step is deferred.
    RECORDINGS[recording_id] = {
        **R, "_name": name, "_uploaded_at": time.time(), "_audio_bytes": y_bytes,
    }
    return recording_id


def serialize_R(R: dict) -> dict:
    """
    ndarray -> list for JSON. Internal-only keys (leading underscore, e.g.
    _wp_chroma, _rms_offset_frames) are dropped from the response -- they're
    alignment bookkeeping, not analysis results a frontend renders.
    """
    out = {}
    for k, v in R.items():
        if k.startswith("_"):
            continue
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        elif isinstance(v, tuple):
            out[k] = list(v)
        else:
            out[k] = v
    return out


def get_pair(a: str, b: str) -> tuple[dict, dict]:
    """
    Fetch both recordings and align b onto a, exactly as align_to() is used
    in app53_patched.py: the first recording of a pair is the reference,
    the second is warped onto it. DTW strategy fixed to 'combo' here since
    none of endpoints 2-5 expose a strategy choice yet -- app53_patched.py's
    UI lets the user pick, but that's a frontend concern (state-classification
    rule in VISUS_BACKEND_API_DESIGN.md section 2: a real UI *input* like
    this belongs as a request parameter once the frontend needs to control
    it, not hardcoded -- flagging this as a known gap, not silently deciding
    it's final).
    """
    Ra = get_recording(a)
    Rb_raw = get_recording(b)
    Rb = compute.align_to(Ra, Rb_raw, do_dtw_flag=True, dtw_strategy="combo")
    return Ra, Rb


# ---------------------------------------------------------------------------
# 1. POST /recordings
# ---------------------------------------------------------------------------
@app.post("/recordings")
async def upload_recording(file: UploadFile = File(...), feature_flags: Optional[str] = Form(None)):
    raw = await file.read()
    try:
        y = compute.load_audio(raw, file.filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode audio: {e}")

    # feature_flags: optional JSON string (form field alongside the file)
    # overriding which of run_analysis()'s do_* computations actually run.
    # Without this, the sidebar's feature checkboxes (CWT, Gammatone,
    # Tonnetz, RT60/Reverb -- all off by default, meaningfully expensive)
    # would have no effect once compute moved server-side: unchecking a
    # box needs to actually skip the computation, not just hide a tab.
    flags = dict(DEFAULT_FEATURE_FLAGS)
    if feature_flags:
        import json as _json
        try:
            overrides = _json.loads(feature_flags)
        except _json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid feature_flags JSON: {e}")
        unknown = set(overrides) - set(flags)
        if unknown:
            raise HTTPException(status_code=400, detail=f"Unknown feature flags: {sorted(unknown)}")
        flags.update(overrides)

    # run_analysis() decodes raw PCM float32 bytes via np.frombuffer() --
    # matches app53_patched.py line 1948 exactly (y.astype(np.float32).tobytes()).
    # Passing the ndarray itself here worked by accident (ndarrays support the
    # buffer protocol, and librosa.load() happens to already return float32,
    # C-contiguous arrays) but silently depended on that instead of stating it.
    y_bytes = y.astype(np.float32).tobytes()
    R = compute.run_analysis(y_bytes, **flags)
    recording_id = store_recording(file.filename, R, y_bytes)

    return {
        "recording_id": recording_id,
        "name": file.filename,
        "features": serialize_R(R),
    }


# ---------------------------------------------------------------------------
# Aligned pair -- full raw feature arrays for both recordings in a pair,
# with b's arrays already warped onto a (compute.align_to()'s output).
#
# Real gap found only once the frontend was actually being written against
# this API: the per-feature tabs (Harmony, Rhythm, Timbre, Dynamics,
# Spectrograms -- app53_patched.py's _render_pair()) all read Ra_sel/Rb_sel
# directly (chroma, mfcc_means, rms, centroid, ...), not a derived score.
# Every other pair endpoint (similarity, divergence, song-map, sections)
# computes align_to() internally and only returns its own derived slice of
# the result -- none of them expose the aligned arrays themselves. Without
# this endpoint the frontend would have to re-implement apply_warp() and
# re-run DTW client-side just to render a bar chart of two already-computed
# chroma vectors, which is exactly the "backend returns raw arrays" rule
# this whole migration is built around, violated in the other direction.
# ---------------------------------------------------------------------------
@app.get("/pairs/{a}/{b}/aligned")
def get_aligned_pair(a: str, b: str):
    Ra, Rb = get_pair(a, b)
    return {
        "recording_a": serialize_R(Ra),
        "recording_b_aligned": serialize_R(Rb),
    }


# ---------------------------------------------------------------------------
# 2. GET /pairs/{a}/{b}/alignment
# ---------------------------------------------------------------------------
@app.get("/pairs/{a}/{b}/alignment")
def get_alignment(a: str, b: str):
    Ra = get_recording(a)
    Rb_raw = get_recording(b)
    Rb_aligned = compute.align_to(Ra, Rb_raw, do_dtw_flag=True, dtw_strategy="combo")

    def wp_to_list(wp):
        return wp.tolist() if isinstance(wp, np.ndarray) else wp

    return {
        "rms_offset_frames": Rb_aligned.get("_rms_offset_frames", 0),
        "rms_offset_seconds": Rb_aligned.get("_rms_offset_seconds", 0.0),
        "warping_paths": {
            "chroma": wp_to_list(Rb_aligned.get("_wp_chroma")),
            "onset": wp_to_list(Rb_aligned.get("_wp_onset")),
            "combo": wp_to_list(Rb_aligned.get("_wp_combo")),
        },
    }


# ---------------------------------------------------------------------------
# 3. GET /pairs/{a}/{b}/similarity
#    Includes the ESM verdict here rather than a separate endpoint: verdict
#    is derived entirely from pair_similarity() plus fields already in this
#    response (tempo, mode), so a second round trip for it would just be the
#    same data twice. Flagging this as a deliberate placement choice, not
#    something the API design doc specified -- worth confirming it's the
#    right call before ESM's frontend is built against it.
# ---------------------------------------------------------------------------
@app.get("/pairs/{a}/{b}/similarity")
def get_similarity(a: str, b: str, name_a: str = "A", name_b: str = "B"):
    Ra, Rb = get_pair(a, b)
    score = compute.pair_similarity(Ra, Rb)
    tier, verdict_text = generate_esm_verdict(Ra, Rb, compute.pair_similarity)

    # generate_summary() (the "What Changed" insight cards) is called from
    # three places in app53_patched.py (lines 1555, 3496, 4250) but isn't in
    # any of the 11 endpoints VISUS_BACKEND_API_DESIGN.md planned -- a gap
    # in that design doc, not just the port. Folded in here rather than a
    # separate endpoint: same Ra/Rb inputs as the similarity score above,
    # so a second round trip would just refetch data this response already
    # has. name_a/name_b are display names (not part of the R dict, purely
    # for the generated text) -- default to generic A/B if the caller
    # doesn't have real names yet.
    insights = compute.generate_summary(Ra, Rb, name_a, name_b)
    breakdown = compute.pair_similarity_breakdown(Ra, Rb)

    return {
        "overall_similarity": score,
        "sub_scores": breakdown,
        "esm_verdict": {"tier": tier, "text": verdict_text},
        "recording_a": {"tempo": Ra.get("tempo"), "mode": Ra.get("mode")},
        "recording_b": {"tempo": Rb.get("tempo"), "mode": Rb.get("mode")},
        "insights": [
            {"icon": icon, "headline": headline, "detail": detail}
            for icon, headline, detail in insights
        ],
    }


# ---------------------------------------------------------------------------
# 4. GET /pairs/{a}/{b}/divergence
# ---------------------------------------------------------------------------
@app.get("/pairs/{a}/{b}/divergence")
def get_divergence(a: str, b: str):
    Ra, Rb = get_pair(a, b)
    t, combined = compute.divergence_1d(Ra, Rb)
    if t is None:
        raise HTTPException(
            status_code=422,
            detail="Not enough overlapping features between these two recordings to compute divergence.",
        )
    return {"time": t.tolist(), "divergence": combined.tolist()}


# ---------------------------------------------------------------------------
# 5. GET /pairs/{a}/{b}/sections
# ---------------------------------------------------------------------------
class SectionOut(BaseModel):
    start_bar: int
    end_bar: int
    start_time: float
    end_time: float
    label: str
    comparison: dict


def _sections_for(a: str) -> list:
    """Shared by /sections and /song-map -- detect_structure() is keyed off
    recording a's audio + tempo only (matches app53_patched.py: sections are
    detected on ys[0], the first/reference recording of the batch)."""
    Ra_raw = get_recording(a)
    return compute.detect_structure(Ra_raw["_audio_bytes"], Ra_raw.get("tempo", 120))


@app.get("/pairs/{a}/{b}/sections")
def get_sections(a: str, b: str):
    Ra, Rb = get_pair(a, b)
    secs = _sections_for(a)

    sections = []
    for b0, b1, t0, t1, lbl in secs:
        cmp = compute.section_comparison(Ra, Rb, t0, t1, "A", "B")
        sections.append({
            "start_bar": b0, "end_bar": b1,
            "start_time": t0, "end_time": t1,
            "label": lbl,
            # comparison: dim name -> {verdict, detail}. verdict is
            # "same" | "slight" | "noticeable" -- Simeon's neutral framing,
            # no better/worse language, per decisions.md.
            "comparison": {dim: {"verdict": v, "detail": d} for dim, (v, d) in cmp.items()},
        })
    return {"sections": sections}


# ---------------------------------------------------------------------------
# Song map -- stream-graph bands, top-3-divergent-moments, and per-section
# stacked-bar divergence. Not one of the original 11 endpoints (the design
# doc's /divergence endpoint covers the single-combined-line case); this is
# the 4-named-band version render_song_map() View 2/3 and
# make_stream_graph_png() actually need. Kept separate from /sections
# (which covers View 1's per-section verdicts) since the two views were
# already split that way in compute.py.
# ---------------------------------------------------------------------------
@app.get("/pairs/{a}/{b}/song-map")
def get_song_map(a: str, b: str):
    Ra, Rb = get_pair(a, b)

    time, bands, colors = compute.stream_divergence_bands(Ra, Rb)
    if time is None:
        raise HTTPException(
            status_code=422,
            detail="Not enough overlapping features between these two recordings for a stream graph.",
        )
    moments = compute.top_divergent_moments(Ra, Rb, n=3)
    secs = _sections_for(a)
    section_bars = compute.section_divergence_breakdown(Ra, Rb, secs)

    return {
        "stream_graph": {
            "time": time.tolist(),
            "bands": {label: vals.tolist() for label, vals in bands.items()},
            "colors": colors,
        },
        "top_divergent_moments": moments,
        "section_bars": section_bars,
    }


# ---------------------------------------------------------------------------
# 6. POST /matrix/similarity
# ---------------------------------------------------------------------------
class RecordingIdsIn(BaseModel):
    recording_ids: list[str]


@app.post("/matrix/similarity")
def matrix_similarity(body: RecordingIdsIn):
    Rs = [get_recording(rid) for rid in body.recording_ids]
    mat = compute.similarity_matrix(Rs)
    return {"recording_ids": body.recording_ids, "matrix": mat.tolist()}


# ---------------------------------------------------------------------------
# 7. POST /matrix/mds
# ---------------------------------------------------------------------------
class MdsIn(BaseModel):
    recording_ids: list[str]
    selected_groups: Optional[list[str]] = None
    weights: Optional[dict] = None


@app.post("/matrix/mds")
def matrix_mds(body: MdsIn):
    Rs = [get_recording(rid) for rid in body.recording_ids]
    try:
        result = compute.compute_mds(
            Rs, body.recording_ids,
            selected_groups=body.selected_groups, weights=body.weights,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {
        "coords": result["coords"].tolist(),
        "display_coords": result["display_coords"].tolist(),
        "overlap_groups": result["overlap_groups"],
        "distance_matrix": result["distance_matrix"].tolist(),
        "selected_groups": result["selected_groups"],
        "available_groups": result["available_groups"],
        "n_dims": result["n_dims"],
        "green_thresh_pct": result["green_thresh_pct"],
        "red_thresh_pct": result["red_thresh_pct"],
    }


# ---------------------------------------------------------------------------
# 8. POST /matrix/group-view
# ---------------------------------------------------------------------------
class GroupViewIn(BaseModel):
    recording_ids: list[str]
    t0: Optional[float] = None  # zoom-window start (s) -- UI value, real param
    t1: Optional[float] = None  # per the state-classification rule


@app.post("/matrix/group-view")
def matrix_group_view(body: GroupViewIn):
    Rs = [get_recording(rid) for rid in body.recording_ids]
    stats = compute.group_view_stats(Rs, body.recording_ids, t0=body.t0, t1=body.t1)

    out = {}
    for key, s in stats.items():
        out[key] = {
            "label": s["label"],
            "time": s["time"].tolist(),
            "mean": s["mean"].tolist(),
            "std": s["std"].tolist(),
            "per_recording": {nm: arr.tolist() for nm, arr in s["per_recording"].items()},
            "outlier_frames": s["outlier_frames"],
            "outlier_summary": s["outlier_summary"],
        }
    return {"features": out}


# ---------------------------------------------------------------------------
# 9. POST /score-alignment
#
# Supports each recording optionally having its own uploaded ground-truth
# score: compute_midi_alignment() uses it instead of the consensus for that
# recording and computes a transcription-accuracy sanity check against it.
# midi_file_to_chroma() (compute.py) does the MIDI-bytes -> chroma
# conversion. Scores are uploaded separately via POST /recordings/{id}/score
# below (base64 body, since this is a plain JSON endpoint, not multipart)
# and cached on the recording so alignment can be re-run without
# re-uploading.
# ---------------------------------------------------------------------------
class ScoreAlignmentIn(BaseModel):
    recording_ids: list[str]


class ScoreUploadIn(BaseModel):
    midi_base64: str


@app.post("/recordings/{recording_id}/score")
def upload_score(recording_id: str, body: ScoreUploadIn):
    import base64
    R = get_recording(recording_id)
    try:
        midi_bytes = base64.b64decode(body.midi_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64: {e}")

    # n_frames_target: match this recording's own chroma length so the
    # uploaded score's chroma array is directly comparable frame-for-frame,
    # same as app53_patched.py's call site.
    n_frames_target = R["chroma"].shape[1] if "chroma" in R else 0
    try:
        score_chroma = compute.midi_file_to_chroma(midi_bytes, n_frames_target)
    except (ValueError, ImportError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    R["_uploaded_score_chroma"] = score_chroma
    return {"recording_id": recording_id, "status": "score attached"}


@app.post("/score-alignment")
def score_alignment(body: ScoreAlignmentIn):
    Rs_aligned = [get_recording(rid) for rid in body.recording_ids]
    ys = [np.frombuffer(R["_audio_bytes"], dtype=np.float32) for R in Rs_aligned]
    uploaded_scores = [R.get("_uploaded_score_chroma") for R in Rs_aligned]
    if not any(s is not None for s in uploaded_scores):
        uploaded_scores = None  # matches compute_midi_alignment()'s "no scores at all" shortcut

    midi_chromas, consensus, alignments, voiced_pcts, any_uploaded = compute.compute_midi_alignment(
        Rs_aligned, ys, body.recording_ids, dtw_strategy="combo",
        uploaded_score_chromas=uploaded_scores,
    )

    def aln_out(a):
        if a is None:
            return None
        return {
            "mean_offset_s": a["mean_offset_s"],
            "max_offset_s": a["max_offset_s"],
            "score": a["score"],
            "offsets": a["offsets"].tolist(),
            "used_own_score": a["used_own_score"],
            "transcription_accuracy": a["transcription_accuracy"],
        }

    return {
        "recording_ids": body.recording_ids,
        "voiced_pcts": voiced_pcts,
        "any_uploaded_score": any_uploaded,
        "consensus_chroma": consensus.tolist(),
        "midi_chromas": [mc.tolist() for mc in midi_chromas],
        "alignments": [aln_out(a) for a in alignments],
    }


# ---------------------------------------------------------------------------
# 10. GET /score-alignment/{id}/midi -- per-recording pYIN transcription,
# rendered to a .mid file. Consensus download (the design doc's "+
# consensus" note) is a real gap: consensus is only meaningful for a group
# of recordings, not a single id, so it needs a different route shape
# (e.g. a POST body of recording_ids) -- not implemented here, flagged
# rather than faked.
# ---------------------------------------------------------------------------
@app.get("/score-alignment/{recording_id}/midi")
def score_alignment_midi(recording_id: str):
    R = get_recording(recording_id)
    y = np.frombuffer(R["_audio_bytes"], dtype=np.float32)
    mc, _, _ = compute.audio_to_midi_chroma(y)
    pm = compute.midi_chroma_to_pretty_midi(mc, tempo=float(R.get("tempo", 120)))

    import tempfile
    from fastapi.responses import FileResponse
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as f:
        pm.write(f.name)
        path = f.name
    return FileResponse(path, media_type="audio/midi",
                         filename=f"{R.get('_name', recording_id)}_transcription.mid")


# ---------------------------------------------------------------------------
# Consensus MIDI download -- closes the design doc's "+ consensus" note on
# endpoint 10 that the single-recording-id route shape can't express.
# Takes the chroma array directly (the frontend already has it from
# POST /score-alignment's consensus_chroma field) rather than recomputing
# alignment just to get back a value already in hand.
# ---------------------------------------------------------------------------
class ChromaToMidiIn(BaseModel):
    chroma: list  # (12, T) nested list, as returned by /score-alignment
    tempo: float = 120.0


@app.post("/midi/from-chroma")
def midi_from_chroma(body: ChromaToMidiIn):
    chroma = np.array(body.chroma, dtype=np.float32)
    pm = compute.midi_chroma_to_pretty_midi(chroma, tempo=body.tempo)

    import tempfile
    from fastapi.responses import FileResponse
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as f:
        pm.write(f.name)
        path = f.name
    return FileResponse(path, media_type="audio/midi", filename="score.mid")


# ---------------------------------------------------------------------------
# 11. GET /score-alignment/{id}/audio -- synthesized sine-wave playback of
# the pYIN transcription, for the "Listen" sanity-check UI.
# ---------------------------------------------------------------------------
@app.get("/score-alignment/{recording_id}/audio")
def score_alignment_audio(recording_id: str):
    R = get_recording(recording_id)
    y = np.frombuffer(R["_audio_bytes"], dtype=np.float32)
    mc, _, _ = compute.audio_to_midi_chroma(y)
    pm = compute.midi_chroma_to_pretty_midi(mc, tempo=float(R.get("tempo", 120)))
    synth_y = compute.synthesize_midi_audio(pm)

    import io
    import soundfile as sf
    from fastapi.responses import StreamingResponse
    buf = io.BytesIO()
    sf.write(buf, synth_y, compute.SR, format="WAV")
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/wav")


# ---------------------------------------------------------------------------
# GET /presets -- from ESM_README.md step 4, called out early because the
# ESM verdict endpoint above needs *a* preset concept to gate against
# eventually. Hardcoded here (not the feature_presets table yet -- that's
# step 4 with Postgres) so ESM_README.md's step ordering isn't jumped ahead
# of; this is a placeholder matching the two rows step 4 specifies, nothing
# more.
# ---------------------------------------------------------------------------
@app.get("/presets")
def list_presets():
    return [
        {"id": "full", "name": "Full", "fields": list(DEFAULT_FEATURE_FLAGS.keys())},
        {
            "id": "everybody_speaks_music",
            "name": "Everybody Speaks Music",
            "fields": ["key", "mode", "tempo", "esm_verdict"],
        },
    ]


# ---------------------------------------------------------------------------
# Waveform preview -- downsampled sample array for the Overview tab's
# waveform-diff plot and the report's waveform-comparison figure. Playback
# itself doesn't need this: the frontend already holds the raw uploaded
# file bytes client-side and can pass them straight to st.audio() without
# a round trip. This is for the plot, which needs decoded, comparable
# sample arrays -- deliberately downsampled (not the full-resolution
# signal) since a multi-minute recording at 22050 Hz would be a multi-MB
# JSON response for what's rendered as a few thousand plotted points.
# ---------------------------------------------------------------------------
@app.get("/recordings/{recording_id}/waveform")
def get_waveform(recording_id: str, n_points: int = 2000):
    R = get_recording(recording_id)
    y = np.frombuffer(R["_audio_bytes"], dtype=np.float32)
    step = max(1, len(y) // n_points)
    ds = y[::step]
    return {"samples": ds.tolist(), "sample_rate_effective": compute.SR / step,
            "duration": R.get("dur", len(y) / compute.SR)}


@app.get("/health")
def health():
    return {"status": "ok", "recordings_in_memory": len(RECORDINGS)}
