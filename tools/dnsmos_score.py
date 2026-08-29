#!/usr/bin/env python3
"""Score every wav in a sweep folder with DNSMOS (speechmos) and update results.json
+ index.html. Runs INSIDE ursula's clip-builder image (it has speechmos + torch):

    tools/dnsmos_score.sh sweeps/<name>

OVRL/SIG/BAK are perceptual-quality estimates (1-5): SIG = speech signal quality
(buzz, metallic, robotic artifacts pull it down), BAK = background noise, OVRL =
overall. They rate cleanliness/naturalness, NOT likeness to the reference voice —
a clip can score 4+ and still not sound like her. Use them to rank the parameter
variants; use your ears for the voice.
"""
import json, sys
from pathlib import Path
import numpy as np, soundfile as sf
from speechmos import dnsmos

root = Path(sys.argv[1])
res = json.loads((root / "results.json").read_text())
for r in res["rows"]:
    if "file" not in r:
        continue
    x, sr = sf.read(str(root / r["file"]), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != 16000:
        import torch, torchaudio
        x = torchaudio.functional.resample(torch.from_numpy(x), sr, 16000).numpy()
    x = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)   # resampling can overshoot; speechmos refuses >1
    d = dnsmos.run(x, sr=16000)
    g = lambda *ks: next((float(d[k]) for k in ks if k in d), None)
    r["dnsmos"] = {"ovrl": g("ovrl_mos", "OVRL"), "sig": g("sig_mos", "SIG"), "bak": g("bak_mos", "BAK"), "p808": g("p808_mos")}
    print(f"{r['n']:02d} {r['name']:<42} OVRL {r['dnsmos']['ovrl']:.2f}  SIG {r['dnsmos']['sig']:.2f}  BAK {r['dnsmos']['bak']:.2f}")
(root / "results.json").write_text(json.dumps(res, indent=1))
# inject a DNSMOS column into the gallery
html = (root / "index.html").read_text()
if "<th>DNSMOS" not in html:
    html = html.replace("<th>WER</th>", "<th>WER</th><th>DNSMOS<br><small>ovrl / sig / bak</small></th>", 1)
    for r in res["rows"]:
        if "dnsmos" not in r:
            continue
        w = "" if r.get("wer") is None else f"{r['wer']*100:.0f}%"
        dn = r["dnsmos"]
        cell = f"<td>{dn['ovrl']:.2f} / {dn['sig']:.2f} / {dn['bak']:.2f}</td>"
        needle = f"<td>{w}</td><td><small>"
        i = html.find(f"src=\"{r['file']}\"")
        j = html.find(needle, i)
        if i > 0 and j > 0:
            html = html[: j + len(f"<td>{w}</td>")] + cell + html[j + len(f"<td>{w}</td>") :]
    (root / "index.html").write_text(html)
