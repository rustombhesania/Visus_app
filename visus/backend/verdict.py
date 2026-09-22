"""
ViSuS -- ESM (Everybody Speaks Music) verdict logic.

Produces a single plain-language line describing how two takes compare,
with zero numbers exposed. Built entirely from signals already computed
in run_analysis() / align_to() -- no new feature extraction needed.

Inputs are R dicts (as produced by app53_patched.py's run_analysis(),
after align_to()) plus pair_similarity() from the same module.

Neutral framing throughout, per Simeon's no-better/worse rule.
"""

TEMPO_SAME_BPM  = 3.0   # below this delta, tempo counts as "same"
TEMPO_LARGE_BPM = 8.0   # above this delta, tempo counts as "large drift"
SIM_CLOSE       = 0.75  # matches the MDS green threshold default
SIM_DIFFERENT   = 0.50  # matches the MDS red threshold default


def _tempo_delta(Ra, Rb):
    return abs(Rb.get("tempo", 120) - Ra.get("tempo", 120))


def _key_match(Ra, Rb):
    # Exact mode string match ("C major" == "C major"), already computed
    # in run_analysis() via detect_key()/detect_mode().
    return Ra.get("mode") is not None and Ra.get("mode") == Rb.get("mode")


def generate_esm_verdict(Ra, Rb, pair_similarity_fn):
    """
    Returns (tier, text):
      tier: "close" | "drift" | "different"
      text: plain-language verdict, no numbers, ready to render as-is.

    pair_similarity_fn: pass pair_similarity from app53_patched.py
    (kept as an argument rather than imported, so this file has no
    dependency on the monolith and can move into the FastAPI backend
    unchanged once step 2 happens).
    """
    td = _tempo_delta(Ra, Rb)
    key_ok = _key_match(Ra, Rb)
    sim = pair_similarity_fn(Ra, Rb)

    tempo_ok  = td < TEMPO_SAME_BPM
    tempo_bad = td > TEMPO_LARGE_BPM

    # ---- Tier 1: close match ----
    if key_ok and tempo_ok and sim >= SIM_CLOSE:
        return "close", "mostly matched"

    # ---- Tier 3: different feel ----
    # Multiple dimensions off, or similarity has fallen through the floor.
    dims_off = sum([not key_ok, tempo_bad, sim < SIM_DIFFERENT])
    if dims_off >= 2:
        return "different", "different feel"

    # ---- Tier 2: some drift -- pick the dominant cause ----
    if not key_ok:
        return "drift", "different key"
    if tempo_bad or not tempo_ok:
        return "drift", "rhythm drifted"
    # key matched, tempo close, but similarity alone dragged it out of
    # "close" -- no single named dimension to blame, so stay generic.
    return "drift", "some drift"
