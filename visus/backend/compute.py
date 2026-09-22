"""
ViSuS backend -- pure-compute functions extracted from app53_patched.py.

Ported per VISUS_BACKEND_API_DESIGN.md section 1: every function here had
zero st.* calls in the original monolith (confirmed by an automated count
during extraction, not just by the design doc audit). No logic changed --
this is a straight move, not a rewrite. Line spans pulled via ast, not
regex, so nothing got truncated or merged across function boundaries.
streamlit and matplotlib imports dropped: nothing here renders or reads
session_state.
"""
import io, os, tempfile

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt, find_peaks, sosfilt
from librosa.sequence import dtw as librosa_dtw

try:
    import pywt; HAS_PYWT = True
except ImportError:
    HAS_PYWT = False

SR     = 22050
HOP    = 512
N_MELS = 128
N_MFCC = 13
N_CQT  = 84
CQT_FMIN  = librosa.note_to_hz('C1')
MIDI_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

MODE_INTERVALS = {
    "Ionian (Major)":[0,2,4,5,7,9,11],  "Dorian":[0,2,3,5,7,9,10],
    "Phrygian":[0,1,3,5,7,8,10],        "Lydian":[0,2,4,6,7,9,11],
    "Mixolydian":[0,2,4,5,7,9,10],      "Aeolian (Minor)":[0,2,3,5,7,8,10],
    "Locrian":[0,1,3,5,6,8,10],
}

MODE_CHARACTER = {
    "Ionian (Major)":   "Bright, happy, resolved.",
    "Dorian":           "Minor with a raised 6th — jazzy, hopeful.",
    "Phrygian":         "Dark, tense, Spanish flavour.",
    "Lydian":           "Dreamy, floating, raised 4th.",
    "Mixolydian":       "Major but flattened 7th — bluesy, dominant.",
    "Aeolian (Minor)":  "Natural minor — sad, introspective.",
    "Locrian":          "Unstable, dissonant, rarely used.",
}

def midi_name(midi): return f"{MIDI_NAMES[midi%12]}{midi//12-1}"

def cqt_bin_to_note(b):
    freqs = librosa.cqt_frequencies(N_CQT, fmin=CQT_FMIN, bins_per_octave=12)
    midi  = int(round(12*np.log2(freqs[b]/440)+69))
    return midi_name(midi)

def load_audio(data, fname):
    suf = os.path.splitext(fname)[-1] or ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suf) as f:
        f.write(data); tmp = f.name
    y, _ = librosa.load(tmp, sr=SR, mono=True)
    os.unlink(tmp); return y

def detect_key(chroma):
    major=[6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88]
    minor=[6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17]
    cm=chroma.mean(1); cm/=(cm.sum()+1e-10)
    best,score,is_min=0,-np.inf,False
    for k in range(12):
        sm=np.corrcoef(cm,np.roll(major,k))[0,1]
        sn=np.corrcoef(cm,np.roll(minor,k))[0,1]
        if sm>score: score,best,is_min=sm,k,False
        if sn>score: score,best,is_min=sn,k,True
    return MIDI_NAMES[best],is_min

def detect_mode(chroma):
    cm=chroma.mean(1); cm/=(cm.sum()+1e-10)
    best_root,best_mode,best_score=0,"Ionian (Major)",-np.inf
    for mode_name,intervals in MODE_INTERVALS.items():
        for root in range(12):
            pcs=set((root+i)%12 for i in intervals)
            score=sum(cm[p] for p in pcs)-0.5*sum(cm[p] for p in range(12) if p not in pcs)
            if score>best_score: best_score,best_root,best_mode=score,root,mode_name
    return MIDI_NAMES[best_root],best_mode,round(float(best_score),4)

def estimate_rt60(y):
    D=np.abs(librosa.stft(y,n_fft=2048,hop_length=HOP))**2
    edb=10*np.log10(D.sum(0)+1e-10); edb-=edb.max()
    fd=HOP/SR; peaks,_=find_peaks(edb,prominence=6,distance=int(0.2/fd))
    rates=[]
    for p in peaks:
        win=int(3/fd); seg=edb[p:min(p+win,len(edb))]
        cut=len(seg)
        for k in range(1,len(seg)):
            if seg[k]>seg[k-1]+2: cut=k; break
        seg=seg[:cut]
        if len(seg)<int(0.1/fd) or seg[-1]-seg[0]>-3: continue
        sl,_=np.polyfit(np.arange(len(seg))*fd,seg,1)
        if sl<-2: rates.append(sl)
    if len(rates)>=2:
        rt60=float(np.clip(abs(-60/np.median(rates)),0.05,10))
    else:
        rt60=float(np.clip(abs(np.percentile(edb,5))/40,0.1,3))
    return round(rt60,3),edb

def gammatone_filterbank(y, n_filters=32, f_min=50, f_max=8000):
    def hz2erb(f): return 21.4*np.log10(1+f/229)
    def erb2hz(e): return 229*(10**(e/21.4)-1)
    cfs=erb2hz(np.linspace(hz2erb(f_min),hz2erb(f_max),n_filters))
    nf=1+(len(y)-1)//HOP; E=np.zeros((n_filters,nf))
    for i,fc in enumerate(cfs):
        bw=1.019*(24.7*(4.37*fc/1000+1)); lo,hi=max(fc-bw/2,1),min(fc+bw/2,SR/2-1)
        if lo>=hi or hi>=SR/2: continue
        try:
            b,a=butter(2,[lo/(SR/2),hi/(SR/2)],btype='band')
            flt=filtfilt(b,a,y)
            for j in range(nf):
                s,e=j*HOP,min((j+1)*HOP,len(flt))
                E[i,j]=np.sqrt(np.mean(flt[s:e]**2)+1e-10)
        except: pass
    E=np.log1p(E); mn,mx=E.min(),E.max()
    if mx>mn: E=(E-mn)/(mx-mn)
    return E

def compute_cwt(y, n_scales=32):
    if not HAS_PYWT: return None
    step=max(1,len(y)//3000); yd=y[::step]
    scales=np.geomspace(2,64,n_scales)
    coef,_=pywt.cwt(yd,scales,'cmor1.5-1.0')
    P=np.abs(coef)**2; mn,mx=P.min(),P.max()
    if mx>mn: P=(P-mn)/(mx-mn)
    return P

def run_analysis(y_bytes,
                 do_mel,do_cqt,do_mfcc,do_chroma,do_spectral,do_onset,do_sms,
                 do_stft,do_cwt,do_gammatone,do_tonnetz,do_zcr,do_reverb):
    y=np.frombuffer(y_bytes,dtype=np.float32)
    R={"dur":len(y)/SR}

    needs_stft = do_stft or do_mel or do_sms or do_reverb or do_onset
    needs_cqt  = do_cqt or do_chroma or do_tonnetz

    if needs_stft:
        D=librosa.stft(y,n_fft=2048,hop_length=HOP)
    if do_stft:
        R["stft"]=librosa.amplitude_to_db(np.abs(D),ref=np.max)
    if do_mel:
        R["mel"]=librosa.power_to_db(librosa.feature.melspectrogram(y=y,sr=SR,n_mels=N_MELS,hop_length=HOP),ref=np.max)
    if needs_cqt:
        C=np.abs(librosa.cqt(y,sr=SR,hop_length=HOP,n_bins=N_CQT,bins_per_octave=12))
    if do_cqt:
        R["cqt"]=librosa.amplitude_to_db(C,ref=np.max)
    if do_mfcc:
        m=librosa.feature.mfcc(y=y,sr=SR,n_mfcc=N_MFCC,hop_length=HOP)
        R["mfcc_means"]=m.mean(1); R["mfcc_stds"]=m.std(1)
        # Normalize per coefficient for display — each coeff scaled to [-1, +1]
        # using its own frame-level range so all 12 are visually comparable.
        # Without this, coefficients 1-4 dominate and 5-12 appear flat near zero.
        coeff_min  = m.min(1)   # min per coefficient across all frames
        coeff_max  = m.max(1)   # max per coefficient across all frames
        coeff_range = coeff_max - coeff_min
        coeff_range[coeff_range < 1e-8] = 1.0   # avoid divide-by-zero
        # Normalize mean to [-1, +1] relative to that coefficient's own range
        R["mfcc_means_z"] = 2 * (m.mean(1) - coeff_min) / coeff_range - 1.0
    if do_chroma:
        R["chroma"]=librosa.feature.chroma_cqt(y=y,sr=SR,hop_length=HOP)
        ka,mi=detect_key(R["chroma"]); R["key"]=f"{ka} {'minor' if mi else 'major'}"
        root,mode,_=detect_mode(R["chroma"])
        R["mode"]=f"{root} {mode}"; R["root"]=root; R["mode_name"]=mode
    if do_spectral:
        R["centroid"] =librosa.feature.spectral_centroid(y=y,sr=SR,hop_length=HOP)[0]
        R["bandwidth"]=librosa.feature.spectral_bandwidth(y=y,sr=SR,hop_length=HOP)[0]
        R["rolloff"]  =librosa.feature.spectral_rolloff(y=y,sr=SR,hop_length=HOP)[0]
        R["flatness"] =librosa.feature.spectral_flatness(y=y,hop_length=HOP)[0]
        R["rms"]      =librosa.feature.rms(y=y,hop_length=HOP)[0]
        if do_zcr:
            R["zcr"]=librosa.feature.zero_crossing_rate(y,hop_length=HOP)[0]
    if do_onset:
        oa=librosa.onset.onset_strength(y=y,sr=SR,hop_length=HOP)
        R["onset"]=oa
        _,beats=librosa.beat.beat_track(onset_envelope=oa,sr=SR,hop_length=HOP)
        R["beats"]=beats
        R["beat_reg"]=round(float(np.std(np.diff(beats))) if len(beats)>2 else 0,3)

        # Tempo octave-error correction: librosa.beat.tempo() can lock
        # onto 2x or 0.5x the true tempo, especially on short/repetitive
        # audio. beat_track() uses a different (dynamic-programming +
        # internal prior) approach and is generally more robust to this
        # failure mode, so cross-check the two and prefer the
        # beat-derived value when they disagree by roughly an octave.
        naive_tempo = float(np.atleast_1d(librosa.beat.tempo(onset_envelope=oa,sr=SR,hop_length=HOP))[0])
        beat_times = librosa.frames_to_time(beats, sr=SR, hop_length=HOP)
        if len(beat_times) > 2:
            from_beats_tempo = 60.0/float(np.median(np.diff(beat_times)))
        else:
            from_beats_tempo = naive_tempo
        _ratio = naive_tempo/from_beats_tempo if from_beats_tempo > 0 else 1.0
        if 1.8 <= _ratio <= 2.2 or 0.45 <= _ratio <= 0.55:
            R["tempo"] = round(from_beats_tempo, 1)
            R["tempo_raw"] = round(naive_tempo, 1)  # kept for debugging/logging only
        else:
            R["tempo"] = round(naive_tempo, 1)
    else:
        R["tempo"]=120.0; R["beat_reg"]=0.0
    if do_sms and needs_stft:
        H,P=librosa.decompose.hpss(D,margin=3.0)
        R["sms_h"]=librosa.amplitude_to_db(np.abs(H),ref=np.max)
        R["sms_p"]=librosa.amplitude_to_db(np.abs(P),ref=np.max)
        he=np.sum(np.abs(H)**2,0)
        R["sms_ratio"]=he/(he+np.sum(np.abs(P)**2,0)+1e-10)
    if do_tonnetz:
        R["tonnetz"]=librosa.feature.tonnetz(y=librosa.effects.harmonic(y),sr=SR)
    if do_reverb:
        R["rt60"],R["decay"]=estimate_rt60(y)
    if do_cwt:
        R["cwt"]=compute_cwt(y)
    if do_gammatone:
        R["gam"]=gammatone_filterbank(y)
    # Global feature ranges (for shared y-axis in plots)
    for k in ["centroid","bandwidth","rolloff","flatness","rms"]:
        if k in R: R[f"{k}_range"]=(float(R[k].min()),float(R[k].max()))
    return R

def safe_normalize_chroma(c):
    """
    Normalize chroma so each frame has unit L2 norm.
    Frames that are all-zero (silence) get replaced with a uniform
    distribution (1/12 per pitch class) to avoid NaN in cosine distance.
    """
    norms = np.linalg.norm(c, axis=0, keepdims=True)          # shape (1, T)
    silent = (norms < 1e-6).flatten()                          # frames with no energy
    c = c / (norms + 1e-10)                                    # normalize all frames
    c[:, silent] = 1.0 / c.shape[0]                           # replace silent frames
    return c.astype(np.float32)

def dtw_on_feature(feat_a_chroma, feat_b_chroma, strategy='combo',
                   onset_a=None, onset_b=None):
    """
    DTW alignment using one of three strategies:
      'chroma'  — chroma features only
      'onset'   — onset strength envelope only
      'combo'   — chroma + onset blended 50/50 (default)
    feat_a_chroma / feat_b_chroma: (12, T) chroma arrays — always required.
    onset_a / onset_b: (T,) onset arrays — required for onset and combo.
    Returns warping path wp of shape (N, 2).
    """
    def safe_norm1d(x):
        x = np.array(x, dtype=np.float32)
        mx = x.max()
        if mx < 1e-8: return np.full_like(x, 1.0/max(len(x),1))
        normed = x/mx; normed[normed<1e-8] = 1e-8
        return normed

    ca = safe_normalize_chroma(feat_a_chroma.astype(np.float32))
    cb = safe_normalize_chroma(feat_b_chroma.astype(np.float32))

    if strategy == 'chroma':
        fa, fb = ca, cb

    elif strategy == 'onset':
        if onset_a is None or onset_b is None:
            fa, fb = ca, cb  # fallback to chroma
        else:
            fa = safe_norm1d(onset_a).reshape(1, -1)
            fb = safe_norm1d(onset_b).reshape(1, -1)

    elif strategy == 'combo':
        if onset_a is None or onset_b is None:
            fa, fb = ca, cb  # fallback to chroma
        else:
            oa_n = safe_norm1d(onset_a)
            ob_n = safe_norm1d(onset_b)
            nc_a = min(ca.shape[1], len(oa_n))
            nc_b = min(cb.shape[1], len(ob_n))
            # Expand onset to 12×T and blend 50/50 with chroma
            oa_exp = np.tile(oa_n[:nc_a], (ca.shape[0], 1))
            ob_exp = np.tile(ob_n[:nc_b], (cb.shape[0], 1))
            fa = 0.5 * ca[:, :nc_a] + 0.5 * oa_exp
            fb = 0.5 * cb[:, :nc_b] + 0.5 * ob_exp
    else:
        fa, fb = ca, cb  # safe fallback

    fa = np.nan_to_num(fa, nan=1e-8, posinf=1.0, neginf=0.0)
    fb = np.nan_to_num(fb, nan=1e-8, posinf=1.0, neginf=0.0)
    _, wp = librosa_dtw(X=fa, Y=fb, metric='cosine')
    return wp

def apply_warp(arr, wp, n_a):
    pa,pb=wp[:,0],wp[:,1]
    if arr.ndim==2:
        out=np.zeros((arr.shape[0],n_a),dtype=np.float32)
        for i in range(n_a):
            idx=pb[pa==i]
            if len(idx): out[:,i]=arr[:,np.clip(idx,0,arr.shape[1]-1)].mean(1)
    else:
        out=np.zeros(n_a,dtype=np.float32)
        for i in range(n_a):
            idx=pb[pa==i]
            if len(idx): out[i]=arr[np.clip(idx,0,len(arr)-1)].mean()
    return out

def coarse_rms_offset(rms_ref, rms_other, sr=SR, hop=HOP):
    """Coarse alignment via RMS cross-correlation. Returns frame offset."""
    from scipy.signal import correlate
    n = min(len(rms_ref), len(rms_other), 2000)
    a = (rms_ref[:n] - rms_ref[:n].mean()) / (rms_ref[:n].std() + 1e-8)
    b = (rms_other[:n] - rms_other[:n].mean()) / (rms_other[:n].std() + 1e-8)
    corr = correlate(a.astype(np.float64), b.astype(np.float64), mode="full")
    lag = int(np.argmax(corr)) - (n - 1)
    return int(np.clip(lag, -int(30*sr/hop), int(30*sr/hop)))

def align_to(R_ref, R_other, do_dtw_flag, dtw_strategy='combo'):
    """
    Two-stage alignment:
      Stage 1: Coarse RMS cross-correlation offset (shifts all arrays).
      Stage 2: Fine DTW using selected strategy (chroma / onset / combo).
    Stores warping paths: _wp_chroma, _wp_onset, _wp_combo.
    Default warp applied to feature arrays uses the selected strategy.
    """
    if not do_dtw_flag:
        return R_other

    SKIP_KEYS = {
        "mfcc_means","mfcc_stds","mfcc_means_z",
        "tempo","beat_reg","beats","key","mode","root",
        "mode_name","mode_char_a","mode_char_b",
        "rt60","decay","dur",
        "centroid_range","bandwidth_range","rolloff_range",
        "flatness_range","rms_range",
    }
    MIN_FRAMES = 50
    aligned = dict(R_other)

    # Stage 1 — Coarse RMS offset
    rms_offset = 0
    if "rms" in R_ref and "rms" in R_other:
        rms_offset = coarse_rms_offset(R_ref["rms"], R_other["rms"])
        aligned["_rms_offset_frames"]  = rms_offset
        aligned["_rms_offset_seconds"] = round(rms_offset * HOP / SR, 3)
        if rms_offset != 0:
            for k in list(aligned.keys()):
                if k.startswith("_") or k in SKIP_KEYS: continue
                v = aligned[k]
                if not isinstance(v, np.ndarray): continue
                if v.ndim == 1 and len(v) >= MIN_FRAMES:
                    if rms_offset > 0: aligned[k] = v[min(rms_offset,len(v)-1):]
                    else:
                        aligned[k] = np.concatenate([np.zeros(-rms_offset,dtype=v.dtype), v])
                elif v.ndim == 2 and v.shape[1] >= MIN_FRAMES:
                    if rms_offset > 0: aligned[k] = v[:,min(rms_offset,v.shape[1]-1):]
                    else:
                        aligned[k] = np.concatenate(
                            [np.zeros((v.shape[0],-rms_offset),dtype=v.dtype), v], axis=1)

    # Stage 2 — DTW strategies (chroma, onset, combo — no RMS)
    n_a = R_ref["chroma"].shape[1] if "chroma" in R_ref else None
    onset_ref = R_ref.get("onset")
    onset_b   = aligned.get("onset")

    if "chroma" in R_ref and "chroma" in aligned and n_a:
        try:
            aligned["_wp_chroma"] = dtw_on_feature(
                R_ref["chroma"], aligned["chroma"], strategy='chroma')
        except Exception: pass
        try:
            aligned["_wp_onset"] = dtw_on_feature(
                R_ref["chroma"], aligned["chroma"], strategy='onset',
                onset_a=onset_ref, onset_b=onset_b)
        except Exception: pass
        try:
            aligned["_wp_combo"] = dtw_on_feature(
                R_ref["chroma"], aligned["chroma"], strategy='combo',
                onset_a=onset_ref, onset_b=onset_b)
        except Exception: pass

    # Pick default warp based on selected strategy
    wp_key_map = {'chroma':'_wp_chroma','onset':'_wp_onset','combo':'_wp_combo'}
    _wp_pref = aligned.get(wp_key_map.get(dtw_strategy, '_wp_combo'))
    _wp_fall = aligned.get("_wp_chroma")
    wp_default = _wp_pref if _wp_pref is not None else _wp_fall

    if wp_default is not None and n_a:
        keys_2d = [k for k in ["stft","mel","cqt","cwt","gam","chroma",
                                "tonnetz","sms_h","sms_p"] if k in aligned]
        keys_1d = [k for k in ["centroid","bandwidth","rolloff","flatness",
                                "rms","zcr","onset","sms_ratio"] if k in aligned]
        for k in keys_2d:
            if aligned[k].shape[1] != n_a:
                aligned[k] = apply_warp(aligned[k].astype(np.float32), wp_default, n_a)
        for k in keys_1d:
            if len(aligned[k]) != n_a:
                aligned[k] = apply_warp(aligned[k].astype(np.float32), wp_default, n_a)

    aligned["_wp"] = wp_default  # backward compat
    return aligned

def detect_structure(y_bytes, tempo):
    y=np.frombuffer(y_bytes,dtype=np.float32)
    mel=librosa.power_to_db(
        librosa.feature.melspectrogram(y=y,sr=SR,n_mels=64,hop_length=HOP),ref=np.max)
    try:
        bounds=librosa.segment.agglomerative(mel,8)
        bt=librosa.frames_to_time(bounds,sr=SR,hop_length=HOP)
    except:
        bt=np.linspace(0,len(y)/SR,9)
    spb=60/max(tempo,1); secs=[]
    labels=["Intro","A","B","A'","C","B'","Outro","Coda"]
    for i in range(len(bt)-1):
        t0,t1=float(bt[i]),float(bt[i+1])
        b0=int(t0/(spb*4))+1; b1=max(b0+1,int(t1/(spb*4))+1)
        secs.append((b0,b1,t0,t1,labels[i%len(labels)]))
    return secs

def divergence_1d(Ra, Rb):
    from scipy.signal import savgol_filter
    arrays=[]
    for k in ["centroid","rms","onset","sms_ratio","flatness"]:
        if k in Ra and k in Rb:
            n=min(len(Ra[k]),len(Rb[k]))
            a=(Ra[k][:n]-Ra[k][:n].mean())/(Ra[k][:n].std()+1e-8)
            b=(Rb[k][:n]-Rb[k][:n].mean())/(Rb[k][:n].std()+1e-8)
            arrays.append(np.abs(a-b))
    if not arrays: return None,None
    nm=min(len(d) for d in arrays)
    combined=np.mean([d[:nm] for d in arrays],axis=0)
    if len(combined)>21:
        combined=savgol_filter(combined,min(51,len(combined)//4*2+1),3)
    combined=np.clip(combined,0,None)
    mx=combined.max()
    if mx>0: combined/=mx
    return np.arange(len(combined))*HOP/SR, combined

STREAM_FEATS = [
    ("rms",       "Volume",     "#4CAF50"),
    ("centroid",  "Brightness", "#2196F3"),
    ("onset",     "Rhythm",     "#FF9800"),
    ("sms_ratio", "Harmony",    "#9C27B0"),
]

def stream_divergence_bands(Ra, Rb):
    """
    Per-feature-band divergence stream. Math half of
    make_stream_graph_png() / render_song_map() View 2 in
    app53_patched.py -- ported verbatim (savgol smoothing, clip, clamshell
    scale). The matplotlib fill_between/legend/label drawing stays out of
    this function entirely; it belongs to whichever frontend renders it.

    The -0.5/+0.5 clamshell scale (each band's share of the peak combined
    divergence) is kept here rather than left to the frontend: it's a
    locked, meaningful transform (Simeon's clamshell-envelope decision),
    not a display preference, so every frontend should render the same
    scaled numbers rather than re-deriving the scale factor itself.

    Returns:
      time: (T,) seconds, or None if no usable feature pair exists
      bands: dict label -> (T,) array, already clamshell-scaled
      colors: dict label -> hex color string
    """
    from scipy.signal import savgol_filter as _sgf
    streams = {}
    for key, label, col in STREAM_FEATS:
        arr_a = Ra.get(key); arr_b = Rb.get(key)
        if arr_a is None or arr_b is None:
            continue
        n_s = min(len(arr_a), len(arr_b))
        a_n = (arr_a[:n_s]-arr_a[:n_s].mean())/(arr_a[:n_s].std()+1e-8)
        b_n = (arr_b[:n_s]-arr_b[:n_s].mean())/(arr_b[:n_s].std()+1e-8)
        div_s = np.abs(a_n - b_n)
        if len(div_s) > 21:
            div_s = _sgf(div_s, min(51, len(div_s)//4*2+1), 3)
        streams[label] = (np.clip(div_s, 0, None), col)

    if not streams:
        return None, None, None

    min_len = min(len(v) for v, _ in streams.values())
    labels  = list(streams.keys())
    vals    = np.array([v[:min_len] for v, _ in streams.values()])  # (n_bands, T)
    colors  = {label: col for label, (_, col) in streams.items()}

    total_s    = vals.sum(axis=0)
    peak_total = total_s.max() + 1e-8
    scale      = 0.5 / peak_total
    vals_scaled = vals * scale  # peak stacked sum -> 0.5, per clamshell decision

    time  = np.arange(min_len) * HOP / SR
    bands = {label: vals_scaled[i] for i, label in enumerate(labels)}
    return time, bands, colors


def top_divergent_moments(Ra, Rb, n=3):
    """
    Top-N moments (by the combined 5-feature divergence from
    divergence_1d) plus which single feature band dominates the
    difference at each moment. Powers the "Top 3 most different moments"
    callout under the stream graph in render_song_map() View 2.

    Returns a list of dicts: time (s), divergence (0-1), dominant_band
    (str, matches a key from stream_divergence_bands()' bands dict).
    Empty list if there isn't enough data to compute either half.
    """
    t_full, div_full = divergence_1d(Ra, Rb)
    if t_full is None:
        return []
    time, bands, colors = stream_divergence_bands(Ra, Rb)
    if time is None:
        return []

    labels = list(bands.keys())
    vals_stack = np.array([bands[l] for l in labels])  # (n_bands, T)

    top_idx = np.argsort(div_full)[::-1][:n]
    moments = []
    for idx in top_idx:
        pt = float(t_full[idx])
        fi = min(int(pt * SR / HOP), vals_stack.shape[1] - 1)
        dom_i = int(np.argmax(vals_stack[:, fi]))
        moments.append({
            "time": pt,
            "divergence": float(div_full[idx]),
            "dominant_band": labels[dom_i],
        })
    return moments


def section_divergence_breakdown(Ra, Rb, secs):
    """
    Per-section, per-dimension mean difference, normalized by each
    dimension's full-recording range. Math half of render_song_map()
    View 3 (stacked bar per section) -- the matplotlib bar/annotate
    drawing and the section table's icon/color mapping stay frontend-only.

    secs: list of (start_bar, end_bar, start_time, end_time, label)
    tuples, exactly as returned by detect_structure().

    Returns a list of dicts, one per section, each with the section's
    span plus a per-dimension divergence dict (rms/centroid/onset/
    sms_ratio -> float). A section's total divergence is the sum of its
    four dimension values -- left to the caller, since "total" is a
    display aggregate, not a distinct computed quantity.
    """
    dim_keys = ["rms", "centroid", "onset", "sms_ratio"]
    rows = []
    for b0, b1, t0, t1, lbl in secs:
        fs_s = max(0, int(t0 * SR / HOP))
        fe_s = max(fs_s + 1, int(t1 * SR / HOP))
        row = {}
        for dk in dim_keys:
            arr_a = Ra.get(dk); arr_b = Rb.get(dk)
            if arr_a is None or arr_b is None:
                row[dk] = 0.0
                continue
            n = min(len(arr_a), len(arr_b))
            sa = arr_a[fs_s:min(fe_s, n)]
            sb = arr_b[fs_s:min(fe_s, n)]
            if len(sa) == 0 or len(sb) == 0:
                row[dk] = 0.0
                continue
            full_range = max(arr_a.max()-arr_a.min(), arr_b.max()-arr_b.min(), 1e-8)
            row[dk] = float(np.abs(sb.mean()-sa.mean()) / full_range)
        rows.append({
            "start_bar": b0, "end_bar": b1,
            "start_time": t0, "end_time": t1,
            "label": lbl, "divergence": row,
        })
    return rows


def generate_summary(Ra, Rb, name_a, name_b):
    from scipy.signal import savgol_filter
    lines=[]
    ta,tb=Ra.get("tempo",120),Rb.get("tempo",120); td=tb-ta
    if abs(td)<1:
        lines.append(("🎯","Tempo: identical",f"Both at {ta:.1f} BPM."))
    elif abs(td)<3:
        lines.append(("🎯",f"Tempo: nearly same (Δ{td:+.1f} BPM)",f"{name_a}: {ta:.1f} · {name_b}: {tb:.1f} BPM."))
    else:
        faster=name_b if td>0 else name_a
        lines.append(("⚡",f"Tempo: {faster} is {abs(td):.1f} BPM faster",
            f"{name_a}: {ta:.1f} · {name_b}: {tb:.1f} BPM. Clearly audible difference."))
    ra,rb=Ra.get("beat_reg",0),Rb.get("beat_reg",0)
    if ra>0 and rb>0:
        pct=(ra-rb)/(ra+1e-10)*100
        tighter=name_b if rb<ra else name_a
        feel="locked in" if min(ra,rb)<1.5 else "fairly consistent" if min(ra,rb)<3 else "loose"
        lines.append(("🥁",f"Timing: {tighter} is {abs(pct):.0f}% tighter",
            f"{name_a}={ra:.2f} · {name_b}={rb:.2f} beat regularity. Tighter feel: {feel}."))
    if "mode" in Ra and "mode" in Rb:
        if Ra["mode"]==Rb["mode"]:
            lines.append(("🎹",f"Key & mode: both in {Ra['mode']}",
                MODE_CHARACTER.get(Ra.get("mode_name",""),"")))
        else:
            lines.append(("🎹",f"Key/mode differs: {name_a}={Ra['mode']} · {name_b}={Rb['mode']}",
                f"{name_a}: {MODE_CHARACTER.get(Ra.get('mode_name',''),'')} "
                f"{name_b}: {MODE_CHARACTER.get(Rb.get('mode_name',''),'')}" ))
    if "rms" in Ra and "rms" in Rb:
        rva,rvb=float(Ra["rms"].mean()),float(Rb["rms"].mean())
        pct=(rvb-rva)/(rva+1e-10)*100
        if abs(pct)<5:
            lines.append(("🔊","Loudness: same","Similar average energy."))
        else:
            louder=name_b if pct>0 else name_a
            lines.append(("🔊",f"Loudness: {louder} is {abs(pct):.0f}% louder",
                "Could be gain difference or performance intensity."))
    if "centroid" in Ra and "centroid" in Rb:
        cva,cvb=float(Ra["centroid"].mean()),float(Rb["centroid"].mean())
        dh=cvb-cva
        if abs(dh)>150:
            brighter=name_b if dh>0 else name_a
            lines.append(("🎨",f"Tone: {brighter} sounds brighter",
                f"Brighter recording has more high-frequency energy."))
        else:
            lines.append(("🎨","Tone: same brightness","Similar tonal colour."))
    if "sms_ratio" in Ra and "sms_ratio" in Rb:
        ha,hb=float(Ra["sms_ratio"].mean()),float(Rb["sms_ratio"].mean())
        dh=hb-ha
        if abs(dh)>0.03:
            more=name_b if dh>0 else name_a
            lines.append(("🎼",f"Texture: {more} is more melodic",
                f"{name_a}={ha:.2f} · {name_b}={hb:.2f} harmonic ratio."))
        else:
            lines.append(("🎼","Texture: same balance","Similar melodic/rhythmic content."))
    # Tone & texture summary (folded in from removed tab)
    tone_parts=[]
    if "centroid" in Ra and "centroid" in Rb:
        cva2=float(Ra["centroid"].mean()); cvb2=float(Rb["centroid"].mean())
        dh2=cvb2-cva2
        if abs(dh2)>150:
            tone_parts.append(f"{name_b if dh2>0 else name_a} sounds brighter")
        else:
            tone_parts.append("similar brightness")
    if "sms_ratio" in Ra and "sms_ratio" in Rb:
        def tex_w(h): return "very melodic" if h>0.75 else "melodic" if h>0.55 else "balanced" if h>0.4 else "rhythmic"
        wa2=tex_w(float(Ra["sms_ratio"].mean())); wb2=tex_w(float(Rb["sms_ratio"].mean()))
        if wa2!=wb2: tone_parts.append(f"{name_a} is {wa2}, {name_b} is {wb2}")
        else: tone_parts.append(f"both are {wa2} in texture")
    if "flatness" in Ra and "flatness" in Rb:
        def flat_w(f): return "clean" if f<0.05 else "slightly gritty" if f<0.15 else "noisy"
        wa3=flat_w(float(Ra["flatness"].mean())); wb3=flat_w(float(Rb["flatness"].mean()))
        if wa3!=wb3: tone_parts.append(f"{name_a} is {wa3}, {name_b} is {wb3}")
        else: tone_parts.append(f"both sound {wa3}")
    if tone_parts:
        lines.append(("🎵","Sound character","; ".join(tone_parts)+"."))
    # Most different moment
    times,div=divergence_1d(Ra,Rb)
    if times is not None:
        peak=int(np.argmax(div)); pt=float(times[peak])
        pm,ps=int(pt//60),int(pt%60)
        avg_t2=(Ra.get("tempo",120)+Rb.get("tempo",120))/2
        bar=int(pt/(60/max(avg_t2,1)*4))+1
        sev="large" if div[peak]>0.7 else "moderate" if div[peak]>0.4 else "small"
        lines.append(("📍",f"Most different: {pm}:{ps:02d} (bar {bar})",
            f"Divergence is {sev} at this point. Zoom in to investigate."))
    return lines

def audio_to_midi_chroma(y, sr=SR, hop=HOP):
    """
    Convert a monophonic audio recording to a chroma representation
    via pYIN pitch estimation.

    Pipeline:
      1. pYIN → F0 estimate per frame (fundamental frequency in Hz)
      2. F0 → MIDI note number (round to nearest semitone)
      3. MIDI note → one-hot chroma (12 pitch classes)

    Returns:
      chroma_midi : (12, T) array — pitch class energy per frame
      voiced_flag : (T,) bool array — True where pYIN was confident
      voiced_pct  : float — % of frames where a clear pitch was detected

    Works best for single-instrument monophonic recordings.
    Confidence is lower with noise, vibrato, or fast passages.
    """
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr, hop_length=hop,
        fill_na=None)

    n_frames = len(f0)
    chroma_midi = np.zeros((12, n_frames), dtype=np.float32)

    for t, (freq, voiced) in enumerate(zip(f0, voiced_flag)):
        if voiced and freq is not None and freq > 0:
            midi_note = int(round(12 * np.log2(freq / 440.0) + 69))
            pitch_class = midi_note % 12
            chroma_midi[pitch_class, t] = 1.0

    voiced_pct = float(voiced_flag.sum() / max(len(voiced_flag), 1) * 100)
    return chroma_midi, voiced_flag, voiced_pct

def midi_chroma_to_pretty_midi(chroma_midi, tempo=120.0,
                                hop=HOP, sr=SR):
    """
    Convert a chroma-from-pYIN array to a pretty_midi.PrettyMIDI object.
    Each voiced frame becomes a short MIDI note event.
    Consecutive frames of the same pitch class are merged into one note.
    Requires: pip install pretty_midi
    """
    try:
        import pretty_midi
    except ImportError:
        raise ImportError(
            "pretty_midi is not installed. "
            "Run: pip install pretty_midi")
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst = pretty_midi.Instrument(program=0, name="pYIN transcription")

    frame_dur = hop / sr
    active = {}   # pitch_class → start_time

    n_frames = chroma_midi.shape[1]
    for t in range(n_frames):
        time = t * frame_dur
        active_pcs = set(np.where(chroma_midi[:, t] > 0)[0])

        # End notes that dropped out
        ended = [pc for pc in list(active) if pc not in active_pcs]
        for pc in ended:
            start = active.pop(pc)
            # Use octave 4 as default (MIDI note = pc + 60)
            note = pretty_midi.Note(
                velocity=80,
                pitch=pc + 60,
                start=start,
                end=max(time, start + frame_dur))
            inst.notes.append(note)

        # Start new notes
        for pc in active_pcs:
            if pc not in active:
                active[pc] = time

    # Close any remaining open notes
    end_time = n_frames * frame_dur
    for pc, start in active.items():
        note = pretty_midi.Note(
            velocity=80,
            pitch=pc + 60,
            start=start,
            end=max(end_time, start + frame_dur))
        inst.notes.append(note)

    inst.notes.sort(key=lambda n: n.start)
    pm.instruments.append(inst)
    return pm

def synthesize_midi_audio(pm, sr=SR):
    """
    Render a pretty_midi.PrettyMIDI object to audio using simple sine-wave
    synthesis (no soundfont required). Returns float32 waveform.
    Lets the user actually listen to a generated/uploaded MIDI transcription
    to sanity-check it against the source recording.
    """
    y = pm.synthesize(fs=sr)
    return y.astype(np.float32)

def midi_file_to_chroma(midi_bytes, n_frames_target, hop=HOP, sr=SR):
    """
    Parse an uploaded MIDI file (.mid) and build a (12, T) chroma array
    from its note events, matching the frame rate used elsewhere
    (HOP/SR). Used when the user uploads a real score instead of relying
    on the auto-generated consensus -- feeds compute_midi_alignment()'s
    uploaded_score_chromas argument.

    Ported verbatim from app53_patched.py (lines 1086-1142). This was
    genuinely missing from the initial extraction, not intentionally
    deferred -- it's the one piece that makes /score-alignment's
    "uploaded score" path (vs. consensus-only) actually work.

    Some exported MIDI files contain out-of-range data bytes (e.g. from
    certain DAWs/notation tools) which makes strict parsers raise
    "data byte must be in range 0..127". Sanitized via mido's clip=True
    first (which clamps invalid bytes instead of raising), then handed
    to pretty_midi for note extraction.
    """
    try:
        import pretty_midi
    except ImportError:
        raise ImportError("pretty_midi not installed. Run: pip install pretty_midi")
    import io as _io

    if not midi_bytes or len(midi_bytes) < 4:
        raise ValueError("Uploaded file is empty or too small to be a valid MIDI file.")
    if midi_bytes[:4] != b'MThd':
        raise ValueError(
            "This doesn't look like a standard MIDI file "
            "(missing 'MThd' header). Make sure you uploaded a .mid/.midi file.")

    try:
        pm = pretty_midi.PrettyMIDI(_io.BytesIO(midi_bytes))
    except Exception as e1:
        try:
            import mido
            mid = mido.MidiFile(file=_io.BytesIO(midi_bytes), clip=True)
            buf = _io.BytesIO()
            mid.save(file=buf)
            buf.seek(0)
            pm = pretty_midi.PrettyMIDI(buf)
        except Exception as e2:
            raise ValueError(
                f"Could not parse this MIDI file even after attempting repair "
                f"(original error: {e1}; repair error: {e2}). "
                f"Try re-exporting the MIDI from its source application.")

    frame_dur = hop / sr
    total_dur = pm.get_end_time()
    n_frames  = max(int(np.ceil(total_dur / frame_dur)), n_frames_target)
    chroma = np.zeros((12, n_frames), dtype=np.float32)
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            f_start = int(note.start / frame_dur)
            f_end   = max(f_start+1, int(note.end / frame_dur))
            pc = note.pitch % 12
            chroma[pc, f_start:min(f_end, n_frames)] = 1.0
    return chroma[:, :n_frames_target] if n_frames >= n_frames_target else chroma


def compute_midi_alignment(Rs_aligned, ys, names,
                            dtw_strategy='combo', hop=HOP, sr=SR,
                            uploaded_score_chromas=None):
    """
    For each recording:
      1. Generate a MIDI chroma via pYIN (the "auto-transcription").
      2. Determine that recording's reference:
         - its OWN uploaded score, if provided (uploaded_score_chromas[i])
         - otherwise a consensus MIDI chroma (mean of all recordings'
           auto-transcriptions that don't have their own uploaded score)
      3. DTW-align each recording's AUDIO chroma to its reference.
      4. If a recording has its own uploaded score, also compute
         "transcription accuracy" — chroma cosine similarity between the
         auto-transcription (pYIN) and the uploaded score — as a sanity
         check on how good the automatic transcription is.

    uploaded_score_chromas: list same length as `names`, entries are
    (12,T) chroma arrays or None. Pass None (the whole argument) if no
    recording has an uploaded score.

    Returns:
      midi_chromas: list of (12, T) — pYIN auto-transcriptions, one per recording
      consensus:    (12, T) fallback consensus chroma (used by recordings
                    without their own uploaded score)
      alignments:   list of dicts per recording:
                    wp, mean_offset_s, max_offset_s, score,
                    used_own_score (bool),
                    transcription_accuracy (float or None)
      voiced_pcts:  list of float — pYIN confidence per recording
      any_uploaded_score: bool — whether at least one recording has an uploaded score
    """
    from librosa.sequence import dtw as librosa_dtw
    n = len(names)
    if uploaded_score_chromas is None:
        uploaded_score_chromas = [None] * n

    # Step 1: generate MIDI chroma for each recording (auto-transcription)
    midi_chromas  = []
    voiced_pcts   = []
    for y in ys:
        mc, vf, vp = audio_to_midi_chroma(y.astype(np.float32), sr=sr, hop=hop)
        midi_chromas.append(mc)
        voiced_pcts.append(vp)

    # Step 2: fallback consensus — mean of transcriptions from recordings
    # that do NOT have their own uploaded score (so an uploaded score
    # doesn't bias the group average of everyone else)
    no_own_score_idx = [i for i in range(n) if uploaded_score_chromas[i] is None]
    if no_own_score_idx:
        min_t = min(midi_chromas[i].shape[1] for i in no_own_score_idx)
        stacked = np.array([midi_chromas[i][:, :min_t] for i in no_own_score_idx])
        consensus = stacked.mean(axis=0)
    else:
        # every recording has its own score — build a plain consensus anyway
        # (used only as a display fallback, not for alignment in this case)
        min_t = min(mc.shape[1] for mc in midi_chromas)
        stacked = np.array([mc[:, :min_t] for mc in midi_chromas])
        consensus = stacked.mean(axis=0)
    norms = np.linalg.norm(consensus, axis=0, keepdims=True)
    silent = (norms < 1e-6).flatten()
    consensus = consensus / (norms + 1e-10)
    consensus[:, silent] = 1.0 / 12

    any_uploaded_score = any(c is not None for c in uploaded_score_chromas)

    # Step 3: per-recording alignment against its own reference
    alignments = []
    for i, R in enumerate(Rs_aligned):
        audio_chroma = R.get("chroma")
        if audio_chroma is None:
            alignments.append(None)
            continue

        own_score = uploaded_score_chromas[i]
        used_own  = own_score is not None
        reference = (safe_normalize_chroma(own_score.astype(np.float32))
                     if used_own else consensus)

        # Transcription accuracy: pYIN auto-transcription vs this
        # recording's own uploaded score (only computable if uploaded)
        transcription_accuracy = None
        if used_own:
            mc_i = midi_chromas[i]
            t_common = min(mc_i.shape[1], reference.shape[1])
            if t_common > 0:
                a = mc_i[:, :t_common].mean(1)
                b = reference[:, :t_common].mean(1)
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if na > 1e-8 and nb > 1e-8:
                    cos = float(np.dot(a, b) / (na * nb))
                    transcription_accuracy = float(np.clip((cos + 1) / 2, 0, 1))

        ac = safe_normalize_chroma(audio_chroma.astype(np.float32))
        t_ref = reference.shape[1]
        t_audio = ac.shape[1]
        ac_trim  = ac[:, :min(t_audio, t_ref)]
        ref_trim = reference[:, :min(t_audio, t_ref)]
        try:
            _, wp = librosa_dtw(X=ac_trim, Y=ref_trim, metric='cosine')
            offsets = np.abs(wp[:, 0].astype(float) -
                             wp[:, 1].astype(float)) * hop / sr
            mean_off  = float(offsets.mean())
            max_off   = float(offsets.max())
            max_frames = max(float(wp[:, 0].max()),
                             float(wp[:, 1].max()), 1.0)
            score = float(np.clip(
                1.0 - offsets.mean() / (max_frames * hop / sr), 0, 1))
            alignments.append({
                "wp": wp,
                "mean_offset_s": round(mean_off, 3),
                "max_offset_s":  round(max_off,  3),
                "score":         round(score, 3),
                "offsets":       offsets,
                "used_own_score": used_own,
                "transcription_accuracy": transcription_accuracy,
            })
        except Exception:
            alignments.append(None)

    return midi_chromas, consensus, alignments, voiced_pcts, any_uploaded_score

def pair_similarity_breakdown(Ra, Rb):
    """
    Named sub-scores behind pair_similarity() -- VISUS_BACKEND_API_DESIGN.md
    section 4 describes GET /pairs/{a}/{b}/similarity as "overall score +
    5 sub-scores", but the initial port only returned the blended overall.
    This is the same five terms pair_similarity() averages, just kept
    named instead of collapsed -- powers the "Why these scores? — feature
    breakdown" table (app53_patched.py lines 3400-3429), which computed
    these same formulas a second time, inline, in the frontend.

    Returns dict: label -> score in [0,1], using only the labels whose
    inputs are present (matches pair_similarity()'s own availability
    checks). Same five labels as the N-recording breakdown table: Timbre,
    Harmony, Tempo, Rhythm, Brightness.
    """
    out = {}
    if "mfcc_means" in Ra and "mfcc_means" in Rb:
        ma, mb = Ra["mfcc_means"], Rb["mfcc_means"]
        cos = float(np.dot(ma,mb)/(np.linalg.norm(ma)*np.linalg.norm(mb)+1e-10))
        out["Timbre"] = (cos+1)/2
    if "chroma" in Ra and "chroma" in Rb:
        ca, cb = Ra["chroma"].mean(1), Rb["chroma"].mean(1)
        cos = float(np.dot(ca,cb)/(np.linalg.norm(ca)*np.linalg.norm(cb)+1e-10))
        out["Harmony"] = (cos+1)/2
    ta, tb = Ra.get("tempo",120), Rb.get("tempo",120)
    out["Tempo"] = max(0, 1 - abs(ta-tb)/50)
    ra, rb = Ra.get("beat_reg",0), Rb.get("beat_reg",0)
    if ra>0 and rb>0:
        out["Rhythm"] = max(0, 1 - abs(ra-rb)/max(ra,rb,1e-8))
    if "centroid" in Ra and "centroid" in Rb:
        ca_v = float(Ra["centroid"].mean())
        cb_v = float(Rb["centroid"].mean())
        out["Brightness"] = max(0, 1 - abs(ca_v-cb_v)/max(ca_v,cb_v,1e-8))
    return out


def pair_similarity(Ra, Rb):
    """
    Fast similarity score for pair selection.
    Uses MFCC cosine + chroma cosine + tempo proximity + beat regularity.
    Returns score in [0, 1] where 1 = identical. Equal to the mean of
    pair_similarity_breakdown()'s values -- kept as a separate function
    (rather than deriving it from the breakdown every call) since this is
    the hot path used by similarity_matrix() and compute_mds() for every
    pair in a batch.
    """
    scores = []
    if "mfcc_means" in Ra and "mfcc_means" in Rb:
        ma, mb = Ra["mfcc_means"], Rb["mfcc_means"]
        cos = float(np.dot(ma,mb)/(np.linalg.norm(ma)*np.linalg.norm(mb)+1e-10))
        scores.append((cos+1)/2)
    if "chroma" in Ra and "chroma" in Rb:
        ca, cb = Ra["chroma"].mean(1), Rb["chroma"].mean(1)
        cos = float(np.dot(ca,cb)/(np.linalg.norm(ca)*np.linalg.norm(cb)+1e-10))
        scores.append((cos+1)/2)
    ta, tb = Ra.get("tempo",120), Rb.get("tempo",120)
    scores.append(max(0, 1 - abs(ta-tb)/50))
    ra, rb = Ra.get("beat_reg",0), Rb.get("beat_reg",0)
    if ra>0 and rb>0:
        scores.append(max(0, 1 - abs(ra-rb)/max(ra,rb,1e-8)))
    if "centroid" in Ra and "centroid" in Rb:
        ca_v = float(Ra["centroid"].mean())
        cb_v = float(Rb["centroid"].mean())
        scores.append(max(0, 1 - abs(ca_v-cb_v)/max(ca_v,cb_v,1e-8)))
    return float(np.mean(scores)) if scores else 0.5

def similarity_matrix(Rs):
    """
    Full (n,n) pairwise similarity matrix, diagonal = 1.0. Ported from
    render_matrix()'s sim_matrix build -- symmetric, so only the upper
    triangle is actually computed. Powers POST /matrix/similarity.
    """
    n = len(Rs)
    mat = np.zeros((n, n))
    for i in range(n):
        mat[i, i] = 1.0
        for j in range(i+1, n):
            s = pair_similarity(Rs[i], Rs[j])
            mat[i, j] = s
            mat[j, i] = s
    return mat


# ---------------------------------------------------------------------------
# MDS map -- feature registry, weighted distance matrix, ring-spread layout,
# data-driven color thresholds. Ported from render_matrix()'s MDS section
# (VISUS_BACKEND_API_DESIGN.md: "MDS coincident-point ring-spread logic:
# pure geometry on computed coordinates -- backend"). All matplotlib
# (plot_mds()'s ax.* drawing) is left out; this returns only the numbers
# a frontend needs to draw dots, connecting lines and a colorbar itself.
# ---------------------------------------------------------------------------
FEATURE_GROUPS = [
    ("MFCC (timbre)",            lambda R: R["mfcc_means"][1:].tolist() if "mfcc_means" in R else None),
    ("Chroma (harmony)",         lambda R: R["chroma"].mean(1).tolist() if "chroma" in R else None),
    ("Tempo",                    lambda R: [float(R["tempo"])] if "tempo" in R else None),
    ("Beat regularity",          lambda R: [float(R["beat_reg"])] if "beat_reg" in R else None),
    ("RMS (loudness)",           lambda R: [float(R["rms"].mean())] if "rms" in R else None),
    ("Centroid (brightness)",    lambda R: [float(R["centroid"].mean())] if "centroid" in R else None),
    ("Bandwidth",                lambda R: [float(R["bandwidth"].mean())] if "bandwidth" in R else None),
    ("Rolloff",                  lambda R: [float(R["rolloff"].mean())] if "rolloff" in R else None),
    ("Flatness",                 lambda R: [float(R["flatness"].mean())] if "flatness" in R else None),
    ("Harmonic ratio (SMS)",     lambda R: [float(R["sms_ratio"].mean())] if "sms_ratio" in R else None),
    ("Tonnetz (tonal centroid)", lambda R: R["tonnetz"].mean(1).tolist() if "tonnetz" in R else None),
]

FEATURE_GROUP_DIMS = {
    "MFCC (timbre)": 12, "Chroma (harmony)": 12,
    "Tonnetz (tonal centroid)": 6,
}


def available_feature_groups(R0):
    return [name for name, fn in FEATURE_GROUPS if fn(R0) is not None]


def build_feature_vector(R, selected_groups):
    vec = []
    group_slices = {}
    for name, fn in FEATURE_GROUPS:
        if name not in selected_groups:
            continue
        vals = fn(R)
        if vals is None:
            continue
        start = len(vec)
        vec.extend(vals)
        group_slices[name] = (start, len(vec))
    return np.array(vec, dtype=np.float32), group_slices


def build_feature_distance_matrix(Rs_list, selected_groups, weights):
    """
    Full (n,n) feature-distance matrix: z-score normalise across all
    recordings (not per-pair), apply per-group weight multiplier, scale
    by the matrix's own max so the largest distance is 1.0. Ported
    verbatim -- this is the exact function VISUS_BACKEND_API_DESIGN.md
    calls out as real computation, not a UI concern.
    """
    vecs_and_slices = [build_feature_vector(R, selected_groups) for R in Rs_list]
    vecs = [v for v, _ in vecs_and_slices]
    group_slices = vecs_and_slices[0][1] if vecs_and_slices else {}
    min_len = min(len(v) for v in vecs) if vecs else 0
    if min_len == 0:
        n_ = len(Rs_list)
        return np.zeros((n_, n_))
    vecs = np.array([v[:min_len] for v in vecs])
    mean = vecs.mean(axis=0)
    std = vecs.std(axis=0)
    std[std < 1e-8] = 1e-8
    normed = (vecs - mean) / std

    for gname, (s, e) in group_slices.items():
        w = weights.get(gname, 1.0)
        if e <= normed.shape[1]:
            normed[:, s:e] *= w

    n_ = len(Rs_list)
    raw = np.zeros((n_, n_))
    for i in range(n_):
        for j in range(n_):
            if i != j:
                raw[i, j] = float(np.linalg.norm(normed[i] - normed[j]))
    mx = raw.max()
    if mx > 1e-8:
        raw = raw / mx
    return raw


def auto_color_thresholds(sim_mat):
    """
    Data-driven default green/red similarity thresholds from this
    dataset's own off-diagonal similarity spread -- falls back to the
    fixed 75/50 split when there isn't enough spread to adapt. Ported
    verbatim from render_matrix(). Returns (green_pct, red_pct) as ints.
    """
    n = sim_mat.shape[0]
    off_diag = sim_mat[~np.eye(n, dtype=bool)]
    if len(off_diag) >= 2 and off_diag.std() > 1e-4:
        mean_, std_ = float(off_diag.mean()), float(off_diag.std())
        green_def = int(round((mean_ + 0.5*std_) * 100 / 5) * 5)
        red_def   = int(round((mean_ - 0.5*std_) * 100 / 5) * 5)
        green_def = min(max(green_def, 10), 95)
        red_def   = min(max(red_def, 5), green_def - 10)
    else:
        green_def, red_def = 75, 50
    return green_def, red_def


def mds_ring_layout(coords, names):
    """
    Nudge coincident/near-coincident MDS points apart into a small ring
    so no recording's dot silently hides behind another's. Pure geometry
    on already-computed coordinates -- ported from plot_mds()'s
    display_coords logic, with all matplotlib drawing removed.

    Returns (display_coords, overlap_groups): display_coords is (n,2),
    same as coords except points within an overlap group are spread onto
    a ring around their shared true position; overlap_groups is a list
    of index-lists that were coincident and got spread.
    """
    n = coords.shape[0]
    x_range = coords[:, 0].max() - coords[:, 0].min() + 1e-8
    y_range = coords[:, 1].max() - coords[:, 1].min() + 1e-8
    plot_scale = max(x_range, y_range)
    coincide_eps = 0.015 * plot_scale
    display_coords = coords.copy()
    visited = set()
    overlap_groups = []
    for i in range(n):
        if i in visited:
            continue
        group = [i]
        for j in range(i+1, n):
            if j in visited:
                continue
            if np.linalg.norm(coords[i] - coords[j]) < coincide_eps:
                group.append(j)
        if len(group) > 1:
            overlap_groups.append(group)
            visited.update(group)
            ring_r = 0.045 * plot_scale
            cx, cy = coords[group].mean(axis=0)
            for k, idx in enumerate(group):
                ang = 2*np.pi*k/len(group)
                display_coords[idx, 0] = cx + ring_r*np.cos(ang)
                display_coords[idx, 1] = cy + ring_r*np.sin(ang)
    return display_coords, overlap_groups


def compute_mds(Rs, names, selected_groups=None, weights=None):
    """
    Full MDS pipeline for the "Recording Map" view: feature registry ->
    weighted z-scored distance matrix -> 2D MDS projection -> ring-spread
    layout for coincident points -> data-driven default color thresholds.
    Requires n >= 3 recordings, matching the guard in render_matrix().

    selected_groups: list of group names to include, or None for all
      groups available given Rs[0]'s computed features.
    weights: dict group_name -> float multiplier, or None for all 1.0.

    Returns a dict: coords, display_coords, overlap_groups,
    distance_matrix, selected_groups, available_groups, n_dims,
    green_thresh_pct, red_thresh_pct.
    """
    from sklearn.manifold import MDS as _MDS
    n = len(Rs)
    if n < 3:
        raise ValueError("MDS requires at least 3 recordings")

    available = available_feature_groups(Rs[0])
    if selected_groups is None:
        selected_groups = available
    if weights is None:
        weights = {g: 1.0 for g in available}

    feat_mat = build_feature_distance_matrix(Rs, selected_groups, weights)
    mds = _MDS(n_components=2, dissimilarity='precomputed',
               random_state=42, n_init=4, normalized_stress='auto')
    coords = mds.fit_transform(feat_mat)

    display_coords, overlap_groups = mds_ring_layout(coords, names)
    green_def, red_def = auto_color_thresholds(similarity_matrix(Rs))
    n_dims = sum(FEATURE_GROUP_DIMS.get(g, 1) for g in selected_groups)

    return {
        "coords": coords,
        "display_coords": display_coords,
        "overlap_groups": overlap_groups,
        "distance_matrix": feat_mat,
        "selected_groups": selected_groups,
        "available_groups": available,
        "n_dims": n_dims,
        "green_thresh_pct": green_def,
        "red_thresh_pct": red_def,
    }


# ---------------------------------------------------------------------------
# Group view -- mean/spread/outlier stats across N recordings, per feature.
# Ported from render_matrix()'s "Group mean ± spread" tab and its outlier
# summary. Powers POST /matrix/group-view.
# ---------------------------------------------------------------------------
OVERLAY_FEATURE_KEYS = [
    ("rms",       "Volume (RMS)"),
    ("centroid",  "Brightness (Centroid Hz)"),
    ("onset",     "Onset Strength"),
    ("sms_ratio", "Harmonic Ratio"),
    ("bandwidth", "Bandwidth (Hz)"),
    ("flatness",  "Spectral Flatness"),
]


def available_overlay_features(Rs):
    return [(k, lbl) for k, lbl in OVERLAY_FEATURE_KEYS if all(k in R for R in Rs)]


def group_view_stats(Rs, names, t0=None, t1=None):
    """
    Group mean/spread/outlier stats across N recordings, per feature.
    Matplotlib fill_between/plot/scatter drawing removed -- only the
    numbers remain.

    t0, t1: optional time window in seconds (matches the "zoom window"
    UI control in app53_patched.py). When given, each feature array is
    cropped to that range by frame index before computing stats -- this
    follows the frontend-input-feeds-real-computation rule (state-
    classification rule in VISUS_BACKEND_API_DESIGN.md section 2): the
    zoom window's *value* is UI state, but once it changes what gets
    computed (which frames go into mean/std/outliers), that computation
    belongs here, parameterized by the value, not silently redone in
    every frontend.

    Returns dict: feature_key -> {label, time, mean, std, per_recording
    (name -> array), outlier_frames (name -> list of frame indices where
    |deviation| > 1.5*std), outlier_summary (name/mean_deviation/
    others_avg, or None if no recording stood out by >1.3x the rest)}.
    """
    feats = available_overlay_features(Rs)
    out = {}
    for key, label in feats:
        arrays = []
        used_names = []
        for R, nm in zip(Rs, names):
            arr = R.get(key)
            if arr is None:
                continue
            if t0 is not None and t1 is not None:
                fs = max(0, int(t0 * SR / HOP))
                fe = max(fs + 1, int(t1 * SR / HOP))
                arr = arr[fs:fe]
            if len(arr) > 0:
                arrays.append(arr)
                used_names.append(nm)
        if len(arrays) < 2:
            continue

        min_len = min(len(a) for a in arrays)
        arrays = [a[:min_len] for a in arrays]
        time = (np.linspace(t0, t1, min_len) if t0 is not None
                else np.arange(min_len) * HOP / SR)
        stacked = np.array(arrays)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)

        outlier_thresh = 1.5 * std
        outlier_frames = {}
        for nm, arr in zip(used_names, arrays):
            dev = np.abs(arr - mean)
            idx = np.where(dev > outlier_thresh)[0]
            outlier_frames[nm] = idx.tolist()

        devs = [float(np.mean(np.abs(a - mean))) for a in arrays]
        outlier_summary = None
        if len(devs) > 1:
            max_dev = max(devs)
            worst_i = int(np.argmax(devs))
            others = float(np.mean(sorted(devs)[:-1]))
            if max_dev > others * 1.3:
                outlier_summary = {
                    "name": used_names[worst_i],
                    "mean_deviation": max_dev,
                    "others_avg": others,
                }

        out[key] = {
            "label": label,
            "time": time,
            "mean": mean,
            "std": std,
            "per_recording": {nm: arr for nm, arr in zip(used_names, arrays)},
            "outlier_frames": outlier_frames,
            "outlier_summary": outlier_summary,
        }
    return out


def section_comparison(Ra, Rb, t0, t1, name_a, name_b):
    """
    Compare Ra and Rb in time window t0–t1 on musical dimensions.
    Returns dict of {dimension: (verdict, detail)}.
    verdict: "same" | "slight" | "noticeable"
    All dimensions are neutral — no "better/worse" judgement.
    """
    def safe_mean(arr, fs, fe):
        """Slice array by frame indices and return mean. Handles empty slices."""
        sl = arr[fs:fe] if arr.ndim==1 else arr[:,fs:fe]
        if sl.ndim==1:
            return float(sl.mean()) if len(sl)>0 else float(arr.mean())
        return float(sl.mean()) if sl.size>0 else float(arr.mean())

    fs_s = max(0, int(t0 * SR / HOP))
    fe_s = max(fs_s+1, int(t1 * SR / HOP))
    result = {}

    # Rhythm — beat regularity comparison (scalar, not per-frame)
    if "beat_reg" in Ra and "beat_reg" in Rb:
        ra = float(Ra["beat_reg"]); rb = float(Rb["beat_reg"])
        mx = max(ra, rb, 1e-8)
        diff_pct = abs(ra-rb)/mx*100
        if diff_pct < 8:
            result["🥁 Rhythm"] = ("same",
                f"Both consistently timed ({name_a}={ra:.2f} · {name_b}={rb:.2f})")
        else:
            tighter = name_a if ra < rb else name_b
            sev = "noticeable" if diff_pct>20 else "slight"
            result["🥁 Rhythm"] = (sev,
                f"{tighter} is more consistent ({name_a}={ra:.2f} · {name_b}={rb:.2f})")

    # Harmony — chroma cosine in this section
    if "chroma" in Ra and "chroma" in Rb:
        ca = Ra["chroma"][:,fs_s:fe_s]; cb = Rb["chroma"][:,fs_s:fe_s]
        nc = min(ca.shape[1], cb.shape[1])
        if nc > 2:
            ca_m = ca[:,:nc].mean(1); cb_m = cb[:,:nc].mean(1)
            nca = np.linalg.norm(ca_m); ncb = np.linalg.norm(cb_m)
            if nca > 1e-8 and ncb > 1e-8:
                cos = float(np.dot(ca_m,cb_m)/(nca*ncb))
                sim = int((cos+1)/2*100)
                if sim >= 90:
                    result["🎵 Harmony"] = ("same",
                        f"Nearly identical harmonic content ({sim}% match)")
                elif sim >= 70:
                    result["🎵 Harmony"] = ("slight",
                        f"Similar harmony with some differences ({sim}% match)")
                else:
                    result["🎵 Harmony"] = ("noticeable",
                        f"Diverging harmonic emphasis ({sim}% match)")

    # Harmonic ratio (SMS) — how melodic vs percussive each recording is
    if "sms_ratio" in Ra and "sms_ratio" in Rb:
        ha = safe_mean(Ra["sms_ratio"], fs_s, fe_s)
        hb = safe_mean(Rb["sms_ratio"], fs_s, fe_s)
        diff_h = abs(ha-hb)
        if diff_h < 0.04:
            result["🎼 Melodic balance"] = ("same",
                f"Both equally melodic/rhythmic ({name_a}={ha:.2f} · {name_b}={hb:.2f})")
        else:
            more_mel = name_a if ha > hb else name_b
            sev = "noticeable" if diff_h>0.10 else "slight"
            result["🎼 Melodic balance"] = (sev,
                f"{more_mel} has more sustained/melodic content "
                f"({name_a}={ha:.2f} · {name_b}={hb:.2f})")

    # Volume — RMS difference
    if "rms" in Ra and "rms" in Rb:
        va = safe_mean(Ra["rms"], fs_s, fe_s)
        vb = safe_mean(Rb["rms"], fs_s, fe_s)
        diff_v = abs(va-vb)/(max(va,vb)+1e-10)*100
        if diff_v < 8:
            result["🔊 Volume"] = ("same", "Similar energy in this section")
        else:
            louder = name_a if va > vb else name_b
            sev = "noticeable" if diff_v>25 else "slight"
            result["🔊 Volume"] = (sev,
                f"{louder} is {diff_v:.0f}% louder here")

    # Brightness — spectral centroid
    if "centroid" in Ra and "centroid" in Rb:
        ca_v = safe_mean(Ra["centroid"], fs_s, fe_s)
        cb_v = safe_mean(Rb["centroid"], fs_s, fe_s)
        diff_c = abs(ca_v-cb_v)/(max(ca_v,cb_v)+1e-10)*100
        if diff_c < 5:
            result["🎨 Brightness"] = ("same", f"Same tonal colour ({ca_v:.0f} Hz avg)")
        else:
            brighter = name_a if ca_v > cb_v else name_b
            sev = "noticeable" if diff_c>15 else "slight"
            result["🎨 Brightness"] = (sev,
                f"{brighter} sounds brighter ({name_a}={ca_v:.0f} · {name_b}={cb_v:.0f} Hz)")

    return result
