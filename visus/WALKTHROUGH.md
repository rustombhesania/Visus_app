# ViSuS Walkthrough - How the Full Client Works, Step by Step

This is the narrated tour: start to finish, in the order you'd actually use
the app, with a real screenshot at each step. Read this top to bottom and
you'll have a working mental model of ViSuS even without opening it.

All screenshots below live in `screenshots/` at the repo root - save each
one under the exact filename shown so they render here automatically on
GitHub.

---

## 1. Getting recordings in

Before any analysis happens, you choose *how many* recordings you're
comparing and *what kind of comparison* you're doing. ViSuS supports two
upload modes, chosen right at the top of the sidebar:

- **2 Rehearsals** - a focused, detailed pairwise comparison between exactly
  two takes. This is the mode for "did we play the bridge the same way in
  both run-throughs."
- **N Rehearsals (batch)** - upload anywhere from 2 to 8 takes at once.
  ViSuS computes every pairwise comparison among them automatically (that's
  N×(N−1)/2 pairs), plus the group-level views described later. This is the
  mode for "which of our five takes today was the outlier."

There's a second, independent choice alongside the audio uploads: an
**optional MIDI reference score**. You're not required to provide one - most
of ViSuS's analysis works purely from the audio itself - but if you have a
score, you can attach it in one of two ways:

- **One score for all recordings** - a single MIDI file uploaded once,
  treated as the reference for every take. Use this when it's the same
  piece performed multiple times.
- **Different score per recording** - a separate MIDI file per take. Use
  this when recordings are different arrangements or different songs
  entirely being compared structurally.

The uploaded MIDI isn't compared directly waveform-to-waveform - instead,
ViSuS separately transcribes each *audio* recording into pitch content using
pYIN pitch estimation, and later checks how well that auto-transcription
matches your uploaded score. That comparison shows up much later in the
walkthrough, in the Score Alignment section.

Audio itself accepts a broad range of formats (WAV, MP3, FLAC, OGG, M4A,
AAC, and more) - the sidebar's file picker filters to these client-side, but
the real validation happens when the backend actually tries to decode the
file.

**`full-sidebar-upload.png`**
*The sidebar: comparison mode, score mode, per-recording audio uploaders,
and optional MIDI score uploaders, all before a single byte of analysis has
run.*

Below the upload sliders, a row of Core and Advanced feature checkboxes lets
you turn on or off specific librosa transforms (mel spectrogram, chroma,
MFCC, CQT, onset detection, and others) - this directly controls what the
backend computes per upload, which matters both for how much detail you get
and how long analysis takes.

Once files are attached, clicking **Run Analysis** sends everything to the
backend and every view described below becomes available.

---

## 2. The first thing you see: Pairwise Similarity Matrix

For N ≥ 3 recordings, the very first thing rendered is a color-coded grid -
one cell per pair, green meaning similar and red meaning different. Each
cell also carries a small stream-graph thumbnail, so you can spot *which*
pairs are interesting before reading a single number.

**`full-pairwise-matrix.png`**
*The full matrix: percentage similarity plus a thumbnail per cell, diagonal
cells marked `-` since a recording is always 100% similar to itself.*

Clicking a pair (via the buttons below the grid) expands that pair's
thumbnail into the full-size chart:

**`full-stream-graph.png`**
*The clamshell-style stream graph - four colored bands (Volume, Brightness,
Rhythm, Harmony) stacked symmetrically above and below a center line. Wider
band at a given moment = that feature is driving more of the divergence
right then. This is drawn from data already fetched for the matrix, so
expanding a pair costs no extra backend call.*

---

## 3. Where recordings sit relative to each other: the MDS map

Once you have 3 or more recordings, ViSuS also computes a 2D projection -
multidimensional scaling (MDS) - placing every take as a point in space,
positioned so that overall distance between points reflects overall feature
distance. Recordings that cluster together played similarly across the
board; an outlier sits apart.

**`full-mds-map.png`**
*Every recording as a labeled point, distance lines between points showing
pairwise similarity.*

The distance calculation isn't fixed - you can reweight how much each
feature group (timbre, harmony, rhythm, dynamics) contributes:

**`full-mds-weights.png`**
*Sliders for adjusting each feature group's weight in the MDS distance
calculation - moving a slider re-runs the projection live.*

---

## 4. Feature-by-feature bar comparison

Below the map, a set of simple bar charts compares scalar metrics directly
across every uploaded recording - tempo (BPM), beat regularity, and
loudness (RMS) - useful for a quick "which take was fastest / tightest /
loudest" read without opening any detailed tab.

**`full-feature-bar-charts.png`**
*Bar charts across all takes for tempo, beat regularity, and RMS loudness.*

---

## 5. Group View - raw values, not divergence

Everything up to this point measures *difference between* recordings. Group
View is different: it plots the actual, raw feature values for every
recording over time, on shared axes - so instead of "these two differ here,"
you see "recording 3 is consistently louder than the others throughout,"
which the difference-only views can't show directly.

**`full-group-view.png`**
*All recordings overlaid on shared time-series axes, with a second view
showing group mean ± spread for the same features.*

---

## 6. The other persona: ESM (Everybody Speaks Music)

Everything above is the **Full client** - built for detailed, scientific
analysis. ViSuS also ships a second, separate frontend for a non-technical
audience: musicians who want a plain-language verdict, not a spectrogram.

**`esm-overview.png`**
*ESM's main view: percentage similarity per pair as simple cards, plus a
feature breakdown table - no raw plots.*

**`esm-detailed-comparison.png`**
*Pair-selection tabs with headline metrics (key, tempo) for the selected
pair - same underlying data as the Full client, presented much more
sparsely.*

Tapping into a specific comparison expands a plain-language explanation
instead of a chart:

**`esm-tap-to-expand.png`**
*A verdict card explaining what's different between two takes in words -
"same/slight/notable difference" language, never "better/worse."*

ESM and the Full client call the exact same backend and the exact same
analysis - the only difference is what each frontend chooses to show and
how much of the underlying compute each requests.

---

## 7. Going deeper on one pair: Alignment

Back in the Full client, selecting a specific pair opens a set of detailed
tabs. The first is Alignment - this is where Dynamic Time Warping (DTW)
lines up the two recordings in time, correcting for the fact that two takes
are never played at exactly the same tempo throughout.

**`full-alignment-warping-paths.png`**
*The DTW warping paths themselves for three alignment strategies (Chroma,
Onset, Combined) - how much each moment in one recording had to shift to
match the other.*

Each strategy can be inspected directly by its effect on the loudness curve
before vs. after alignment:

**`full-alignment-chroma-dtw.png`** - RMS energy before/after Chroma-based alignment
**`full-alignment-onset-dtw.png`** - RMS energy before/after Onset-based alignment
**`full-alignment-combo-dtw.png`** - RMS energy before/after Combined alignment

And the same idea applied to the chromagram itself, side by side with a
difference heatmap:

**`full-alignment-chroma-spectrograms.png`**
*Chromagrams for both recordings before and after alignment, with a
difference heatmap making mismatches visually obvious.*

---

## 8. Harmony

**`full-harmony-overview.png`**
*Detected key/mode per recording, dominant pitch classes, and an overall
harmonic similarity percentage for the pair.*

**`full-harmony-pitch-energy.png`**
*Energy per chromatic pitch class as bar charts, plus a differential energy
plot showing exactly which notes diverge most.*

---

## 9. Rhythm

**`full-rhythm-metrics.png`**
*Tempo (BPM), beat regularity, and a tightness score for the pair.*

**`full-rhythm-onset-comparison.png`**
*Per-frame onset envelope strength for both recordings, plus their
difference over time - where one recording's attacks landed vs. the
other's.*

---

## 10. Dynamics

**`full-dynamics-volume-over-time.png`**
*Volume envelopes over time for both recordings, with a frame-by-frame
volume delta plot underneath.*

---

## 11. Spectral characteristics

A cluster of related tabs, each isolating one spectral property and
plotting it over time for both recordings plus their difference:

**`full-spectral-brightness.png`** - spectral centroid (brightness) over time
**`full-spectral-bandwidth.png`** - spectral bandwidth (harmonic richness)
**`full-spectral-rolloff.png`** - high-frequency energy cutoff
**`full-spectral-flatness.png`** - tonal vs. noise-like character
**`full-spectral-rms.png`** - loudness curves again, in the spectral-features context

---

## 12. Zooming back out: feature overlay across all recordings

Stepping back from any one pair, this view overlays Volume, Brightness, and
Onset Strength for *every* uploaded recording on shared axes at once - a
wide-angle view after all the pairwise depth above.

**`full-feature-overlay.png`**
*All rehearsals' Volume, Brightness, and Onset Strength curves overlaid on
shared time axes.*

---

## 13. Spectrograms

The most literal view of the raw audio: 128-band mel spectrograms for the
reference recording, the DTW-aligned comparison recording, and a
differential spectrogram showing exactly where energy diverges across time
and frequency.

**`full-spectrograms.png`**
*Reference, aligned comparison, and differential mel spectrogram, side by
side.*

---

## 14. Score Alignment - where the uploaded MIDI comes back in

This is the tab that closes the loop on the optional MIDI upload from step
1. Each recording is independently transcribed to a chroma representation
via pYIN pitch estimation (monophonic pitch detection, frame by frame). If
you uploaded a reference score, ViSuS checks how well that auto-transcription
lines up with it; if you didn't, this tab still works by building a
**consensus score** - averaging the pYIN transcriptions across all your
recordings into a single reference chroma, on the theory that most takes
agree on the actual notes even if any one take has transcription noise.

**`full-score-alignment-confidence.png`**
*Voiced-frame percentage and a confidence badge per recording - how much of
each recording pYIN was actually able to pitch-track cleanly.*

**`full-score-consensus-chroma.png`**
*The consensus MIDI chromagram - the average pitch content across every
pYIN transcription, standing in as the reference score when none was
uploaded.*

**`full-score-temporal-offset.png`**
*Temporal offset vs. the consensus score, in seconds, over time - where a
recording's timing drifted ahead of or behind the group.*

Finally, each recording's own pYIN transcription is compared directly
against its real audio chromagram, giving a per-recording transcription
accuracy score:

**`full-score-pyin-vs-chroma-r1.png`** - Rehearsal 1 (92%)
**`full-score-pyin-vs-chroma-r2.png`** - Rehearsal 2 (98%)
**`full-score-pyin-vs-chroma-r3.png`** - Rehearsal 3 (83%)
**`full-score-pyin-vs-chroma-r4.png`** - Rehearsal 4 (98%)

A lower score here (like Rehearsal 3's 83%) doesn't necessarily mean the
performance was worse - it usually means that particular recording was
harder for pYIN to pitch-track cleanly (more polyphony, noise, or fast
passages), which is exactly what the confidence badges in the first
screenshot of this section are there to flag.

---

## That's the full loop

Upload (audio, optionally MIDI) → sidebar feature selection → matrix and
stream graphs → MDS map → bar charts → Group View → (or, on the ESM side:
percentages and plain-language verdicts) → per-pair deep dives (alignment,
harmony, rhythm, dynamics, spectral) → feature overlay → spectrograms →
score alignment closing the loop on the MIDI reference from step 1.

Every view here is backend-computed and frontend-drawn from raw arrays - see
`README.md` for the architecture behind how that split works, and
`ARCHITECTURE.md` for the Full-vs-ESM persona comparison in more technical
detail.
