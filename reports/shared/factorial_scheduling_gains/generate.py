"""Regenerate this figure entirely from the bundled evaluation summaries."""
from pathlib import Path
import hashlib
import json
import sys
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
out = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE / 'outputs'
out.mkdir(parents=True, exist_ok=True)
for item in json.loads((HERE / 'data/manifest.json').read_text()):
    actual = hashlib.sha256((HERE / item['file']).read_bytes()).hexdigest()
    assert actual == item['sha256'], f"Input changed: {item['file']}"
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'pdf.fonttype': 42})
from analysis import analyze, factorial_figure
stats = analyze()
payload = stats
factorial_figure(stats, out)
(out / 'analysis.json').write_text(json.dumps(payload, indent=2) + '\n')
print(f"Verified 16 input files; generated figure and analysis in {out}")
