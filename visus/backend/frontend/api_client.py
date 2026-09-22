"""
ViSuS -- thin HTTP client for the FastAPI backend.

Every function here does exactly one thing: call an endpoint, convert its
JSON response back into the numpy/dict shapes app_full_client.py expects
(matching what the equivalent local compute.py call used to return), and
raise a clear error if the backend is unreachable or returns a failure.

No analysis logic lives here -- that's the backend's job now. This module
exists so app_full_client.py never has to know about requests, JSON, or
HTTP status codes directly.
"""
import numpy as np
import requests
import streamlit as st

def _resolve_base_url() -> str:
    """st.secrets.get() looks safe but isn't: with no secrets.toml at all,
    Streamlit raises StreamlitSecretNotFoundError from inside .get() itself
    (found via AppTest -- a real bug, not a hypothetical), so a bare
    hasattr(st, "secrets") guard doesn't help. Try/except is the only
    reliable way to fall back to localhost when no secrets file exists."""
    try:
        return st.secrets.get("BACKEND_URL", "http://localhost:8000")
    except Exception:
        return "http://localhost:8000"


BASE_URL = _resolve_base_url()


class BackendError(Exception):
    """Raised for any non-2xx response or connection failure, with the
    backend's own error detail (if any) surfaced in the message so a
    failure is diagnosable from the Streamlit error banner alone."""
    pass


def _url(path: str) -> str:
    return f"{BASE_URL.rstrip('/')}{path}"


def _get(path: str, **params):
    try:
        r = requests.get(_url(path), params=params, timeout=120)
    except requests.exceptions.ConnectionError as e:
        raise BackendError(
            f"Could not reach the backend at {BASE_URL}. "
            f"Is it running? (uvicorn main:app --port 8000)"
        ) from e
    if not r.ok:
        raise BackendError(_error_detail(r))
    return r.json()


def _post(path: str, json=None, files=None, data=None):
    try:
        r = requests.post(_url(path), json=json, files=files, data=data, timeout=300)
    except requests.exceptions.ConnectionError as e:
        raise BackendError(
            f"Could not reach the backend at {BASE_URL}. "
            f"Is it running? (uvicorn main:app --port 8000)"
        ) from e
    if not r.ok:
        raise BackendError(_error_detail(r))
    return r.json()


def _post_raw(path: str, json=None):
    """Like _post, but returns raw response bytes (for file downloads:
    MIDI, WAV) instead of parsing JSON."""
    try:
        r = requests.post(_url(path), json=json, timeout=300)
    except requests.exceptions.ConnectionError as e:
        raise BackendError(f"Could not reach the backend at {BASE_URL}.") from e
    if not r.ok:
        raise BackendError(_error_detail(r))
    return r.content


def _get_raw(path: str, **params):
    try:
        r = requests.get(_url(path), params=params, timeout=300)
    except requests.exceptions.ConnectionError as e:
        raise BackendError(f"Could not reach the backend at {BASE_URL}.") from e
    if not r.ok:
        raise BackendError(_error_detail(r))
    return r.content


def _error_detail(r) -> str:
    try:
        return r.json().get("detail", r.text)
    except Exception:
        return r.text or f"HTTP {r.status_code}"


def _arrayify(d: dict) -> dict:
    """Convert every list value that looks like a numeric array back to
    a numpy array. Mirrors serialize_R()'s inverse on the backend --
    scalars (tempo, key, mode strings) pass through unchanged."""
    out = {}
    for k, v in d.items():
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float, list)):
            out[k] = np.array(v, dtype=np.float32)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# 1. POST /recordings
# ---------------------------------------------------------------------------
def upload_recording(file_bytes: bytes, filename: str, feature_flags: dict = None) -> dict:
    """Returns {"recording_id": str, "name": str, "R": dict} -- R has
    numpy arrays for every feature, same shape as the old local
    run_analysis() output. feature_flags (optional) mirrors the sidebar
    checkboxes -- passed as a JSON-encoded form field so the backend can
    override its DEFAULT_FEATURE_FLAGS per upload (see main.py's
    POST /recordings; this is what makes the sidebar checkboxes actually
    do something once compute moved server-side)."""
    import json as _json
    data = {"feature_flags": _json.dumps(feature_flags)} if feature_flags is not None else None
    resp = _post("/recordings", files={"file": (filename, file_bytes)}, data=data)
    return {
        "recording_id": resp["recording_id"],
        "name": resp["name"],
        "R": _arrayify(resp["features"]),
    }


# ---------------------------------------------------------------------------
# Aligned pair -- full raw feature arrays, b already warped onto a. This is
# what most per-feature tabs (Harmony, Rhythm, Timbre, Dynamics,
# Spectrograms) actually need -- similarity/divergence/song-map are
# derived views of the same underlying alignment, not a substitute for it.
# ---------------------------------------------------------------------------
def get_aligned_pair(a: str, b: str) -> tuple:
    """Returns (Ra, Rb_aligned) -- both dicts with numpy arrays, same
    shape as a single recording's R dict from upload_recording()."""
    resp = _get(f"/pairs/{a}/{b}/aligned")
    return _arrayify(resp["recording_a"]), _arrayify(resp["recording_b_aligned"])


# ---------------------------------------------------------------------------
# 2. GET /pairs/{a}/{b}/alignment
# ---------------------------------------------------------------------------
def get_alignment(a: str, b: str) -> dict:
    resp = _get(f"/pairs/{a}/{b}/alignment")
    wp = resp["warping_paths"]
    return {
        "rms_offset_frames": resp["rms_offset_frames"],
        "rms_offset_seconds": resp["rms_offset_seconds"],
        "wp_chroma": np.array(wp["chroma"]) if wp["chroma"] else None,
        "wp_onset": np.array(wp["onset"]) if wp["onset"] else None,
        "wp_combo": np.array(wp["combo"]) if wp["combo"] else None,
    }


# ---------------------------------------------------------------------------
# 3. GET /pairs/{a}/{b}/similarity
# ---------------------------------------------------------------------------
def get_similarity(a: str, b: str, name_a: str = "A", name_b: str = "B") -> dict:
    return _get(f"/pairs/{a}/{b}/similarity", name_a=name_a, name_b=name_b)


# ---------------------------------------------------------------------------
# 4. GET /pairs/{a}/{b}/divergence
# ---------------------------------------------------------------------------
def get_divergence(a: str, b: str):
    """Returns (time, divergence) as numpy arrays, or (None, None) if the
    backend reports there isn't enough overlapping data (422)."""
    try:
        resp = _get(f"/pairs/{a}/{b}/divergence")
    except BackendError:
        return None, None
    return np.array(resp["time"]), np.array(resp["divergence"])


# ---------------------------------------------------------------------------
# 5. GET /pairs/{a}/{b}/sections
# ---------------------------------------------------------------------------
def get_sections(a: str, b: str) -> list:
    """Returns a list of (start_bar, end_bar, start_time, end_time, label,
    comparison) tuples -- comparison is {dim: (verdict, detail)}, same
    shape section_comparison() used to return, for drop-in reuse in the
    Song Map View 1 rendering code."""
    resp = _get(f"/pairs/{a}/{b}/sections")
    out = []
    for s in resp["sections"]:
        cmp = {dim: (v["verdict"], v["detail"]) for dim, v in s["comparison"].items()}
        out.append((s["start_bar"], s["end_bar"], s["start_time"], s["end_time"], s["label"], cmp))
    return out


# ---------------------------------------------------------------------------
# Song map -- stream bands, top divergent moments, section bars
# ---------------------------------------------------------------------------
def get_song_map(a: str, b: str) -> dict:
    resp = _get(f"/pairs/{a}/{b}/song-map")
    sg = resp["stream_graph"]
    return {
        "time": np.array(sg["time"]),
        "bands": {label: np.array(vals) for label, vals in sg["bands"].items()},
        "colors": sg["colors"],
        "top_divergent_moments": resp["top_divergent_moments"],
        "section_bars": resp["section_bars"],
    }


# ---------------------------------------------------------------------------
# 6. POST /matrix/similarity
# ---------------------------------------------------------------------------
def matrix_similarity(recording_ids: list) -> np.ndarray:
    resp = _post("/matrix/similarity", json={"recording_ids": recording_ids})
    return np.array(resp["matrix"])


# ---------------------------------------------------------------------------
# 7. POST /matrix/mds
# ---------------------------------------------------------------------------
def matrix_mds(recording_ids: list, selected_groups=None, weights=None) -> dict:
    body = {"recording_ids": recording_ids}
    if selected_groups is not None:
        body["selected_groups"] = selected_groups
    if weights is not None:
        body["weights"] = weights
    resp = _post("/matrix/mds", json=body)
    return {
        "coords": np.array(resp["coords"]),
        "display_coords": np.array(resp["display_coords"]),
        "overlap_groups": resp["overlap_groups"],
        "distance_matrix": np.array(resp["distance_matrix"]),
        "selected_groups": resp["selected_groups"],
        "available_groups": resp["available_groups"],
        "n_dims": resp["n_dims"],
        "green_thresh_pct": resp["green_thresh_pct"],
        "red_thresh_pct": resp["red_thresh_pct"],
    }


# ---------------------------------------------------------------------------
# 8. POST /matrix/group-view
# ---------------------------------------------------------------------------
def matrix_group_view(recording_ids: list, t0=None, t1=None) -> dict:
    body = {"recording_ids": recording_ids}
    if t0 is not None:
        body["t0"] = t0
    if t1 is not None:
        body["t1"] = t1
    resp = _post("/matrix/group-view", json=body)
    out = {}
    for key, s in resp["features"].items():
        out[key] = {
            "label": s["label"],
            "time": np.array(s["time"]),
            "mean": np.array(s["mean"]),
            "std": np.array(s["std"]),
            "per_recording": {nm: np.array(v) for nm, v in s["per_recording"].items()},
            "outlier_frames": s["outlier_frames"],
            "outlier_summary": s["outlier_summary"],
        }
    return out


# ---------------------------------------------------------------------------
# 9. POST /score-alignment  +  score upload
# ---------------------------------------------------------------------------
def upload_score(recording_id: str, midi_bytes: bytes):
    import base64
    b64 = base64.b64encode(midi_bytes).decode()
    return _post(f"/recordings/{recording_id}/score", json={"midi_base64": b64})


def score_alignment(recording_ids: list) -> dict:
    resp = _post("/score-alignment", json={"recording_ids": recording_ids})
    return {
        "voiced_pcts": resp["voiced_pcts"],
        "any_uploaded_score": resp["any_uploaded_score"],
        "consensus_chroma": np.array(resp["consensus_chroma"]),
        "midi_chromas": [np.array(mc) for mc in resp["midi_chromas"]],
        "alignments": [
            None if a is None else {**a, "offsets": np.array(a["offsets"])}
            for a in resp["alignments"]
        ],
    }


# ---------------------------------------------------------------------------
# 10/11. MIDI + audio downloads
# ---------------------------------------------------------------------------
def download_transcription_midi(recording_id: str) -> bytes:
    return _get_raw(f"/score-alignment/{recording_id}/midi")


def download_transcription_audio(recording_id: str) -> bytes:
    return _get_raw(f"/score-alignment/{recording_id}/audio")


def download_chroma_as_midi(chroma: np.ndarray, tempo: float = 120.0) -> bytes:
    return _post_raw("/midi/from-chroma", json={"chroma": chroma.tolist(), "tempo": tempo})


# ---------------------------------------------------------------------------
# Waveform preview (for the Overview waveform-diff plot / report figure --
# NOT for playback, which uses the raw uploaded bytes directly, client-side)
# ---------------------------------------------------------------------------
def get_waveform(recording_id: str, n_points: int = 2000) -> dict:
    resp = _get(f"/recordings/{recording_id}/waveform", n_points=n_points)
    return {
        "samples": np.array(resp["samples"], dtype=np.float32),
        "sample_rate_effective": resp["sample_rate_effective"],
        "duration": resp["duration"],
    }


# ---------------------------------------------------------------------------
# Presets + health
# ---------------------------------------------------------------------------
def list_presets():
    return _get("/presets")


def health() -> bool:
    try:
        r = _get("/health")
        return r.get("status") == "ok"
    except BackendError:
        return False
