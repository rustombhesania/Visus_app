# Everybody Speaks Music - the ESM persona

ESM (Everybody Speaks Music) is a real university band, and this persona
exists because of them specifically - not as a generic "simple mode" toggle.
This document explains who they are, why they get their own frontend
instead of a checkbox in the Full client, and what they actually see.

---

## Who ESM is and what they actually need

ESM is a small university band that rehearses the same songs repeatedly and
wants a fast, honest read on how a given take compares to previous ones -
without needing to interpret a spectrogram, a DTW alignment plot, or a
percentage "framed as science" to get there. Most members aren't audio
engineers or signal-processing students; they're musicians who want to know,
in plain terms, whether tonight's run-through matched last week's, or where
it drifted.

That's a genuinely different requirement from the Full client's audience
(Rustom and Simeon doing detailed analysis work), not a stripped-down
version of the same requirement. A researcher wants every plot available in
case it's needed. A band member wants an answer they can act on in the ten
seconds between songs at rehearsal.

## What ESM sees - and, just as deliberately, what it doesn't

- A single plain-language verdict per pair ("Mostly matched," "Some drift,"
  "Different feel"), with audio playback and key/tempo up top
- "What stands out" - summary cards for Tempo, Timing, Key/Mode, Loudness,
  Tone, Texture, and Sound Character, in words
- A simplified stream graph showing which features drove the difference
  over time (Volume, Brightness, Rhythm, Harmony)
- A section-by-section breakdown (Intro, A, B, A', C, B', Outro), each
  flagged with neutral same/slight/notable-difference language
- Four dimension cards - Harmony, Rhythm, Tone/Timbre, Dynamics - each
  pairing a percentage badge with one simplified chart (a pitch-class bar
  chart, an onset overlap graph, a spectral fingerprint, a volume-envelope
  plot)
- Up to 8 takes at once, same batch support as the Full client

So ESM isn't three bare numbers - it does show charts, just simplified,
single-purpose ones scoped to one dimension at a time. What's deliberately
absent is the research-grade view: spectrograms, DTW alignment plots, MDS
maps, chromagrams, and the Score/MIDI alignment tab. None of it renders in
ESM, not because the backend can't produce it (it's the exact same backend,
same compute), but because none of it answers the question ESM actually
has - it reads as data a musician has to learn to interpret, not an answer.

## The language rule matters more here, not less

ViSuS's project-wide rule against evaluative language ("better/worse" is
never used, only neutral same/slight/notable-difference framing) exists for
every persona, but it's especially load-bearing for ESM. A researcher
reading a percentage can contextualize it. A band member reading "this take
was 73% similar" with no other framing might reasonably read that as a
grade. ESM's verdict cards are written specifically to avoid that: a
rehearsal comparison is information for the band to use, not a score to
feel judged by.

## Why a separate app, not a mode inside the Full client

Kept as a fully separate Streamlit entrypoint (`app_esm_client.py`) rather
than a toggle inside the Full client, for a reason specific to this
audience: a toggle can be misconfigured or accidentally switched, showing a
non-technical user a spectrogram they didn't ask for and can't interpret. A
separate deploy makes that impossible - ESM's link only ever shows ESM's
view, nothing else. See `README.md`'s [architecture](README.md#architecture)
section for how both personas share one backend despite being separate
frontends.

## What ESM has actually asked for

Real feedback from a pilot session with the group, September 2026: they
found the analysis accuracy meaningfully useful (with an open item that
chord detection is sometimes off) and asked for two things - a
scrollable, FL-Studio-style timeline view (chords, key, and notes per
section, with detail on click) and somewhere to keep past rehearsals for
reference, which is mostly what they'd use it for day to day. Both are
live discussion items with Simeon on direction, not yet built.

---

**See it in action:** [`WALKTHROUGH_ESM.md`](WALKTHROUGH_ESM.md) is the
narrated, screenshot-by-screenshot tour of exactly what an ESM band member
sees, start to finish.

**Curious about the Full client instead?** See the main
[`README.md`](README.md) and [`WALKTHROUGH.md`](WALKTHROUGH.md).
