from pathlib import Path
import nbformat
from nbconvert import HTMLExporter
import zipfile

out = Path(__file__).resolve().parents[1] / 'outputs'
nb = nbformat.read(out / 'Qismet.ipynb', as_version=4)
for cell in nb.cells:
    if cell.cell_type == 'code':
        cell.source = cell.source.replace("fig.suptitle('Qismet · 8 hidden neurons → 4 · ideal Qiskit simulation', fontsize=16)",
                                          "fig.suptitle(f'Qismet · {HIDDEN} hidden neurons → {keep} · ideal Qiskit simulation', fontsize=16)")
interpretation = '''### What this executed run supports

The QAOA-selected four-neuron networks retained the dense model's accuracy on this split,
but **random pruning achieved the same mean test accuracy (97.67%)**. Magnitude pruning
averaged 98.00%, and classical swap search averaged 97.67%. These small differences are
descriptive; no statistical superiority is established.

The quadratic approximation has limited predictive quality at the requested pruning budget:
held-out-mask R² values are **−0.262, 0.300, and −0.122**. Two are negative. The mean
rank correlation is 0.427. QAOA's conditional top-decile probability is **10.06%**, against
10.00% for uniform feasible masks, so it learns cardinality much more strongly than good-mask ranking.

All nine COBYLA restarts reached the declared evaluation limit; the result is the best observed
within that budget, not a claim of optimizer convergence. The next research steps are to improve
the surrogate and evaluate a cardinality-preserving mixer under a new, predeclared protocol.
'''
nb.cells.insert(-1, nbformat.v4.new_markdown_cell(interpretation))
nbformat.write(nb, out / 'Qismet.ipynb')
body, _ = HTMLExporter().from_notebook_node(nb)
(out / 'Qismet.html').write_text(body, encoding='utf-8')
with (out / 'results/RESULTS.md').open('a', encoding='utf-8') as f:
    f.write('\n' + interpretation + '\n')
bundle = out / 'Qismet_complete.zip'
with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as z:
    for path in sorted(out.rglob('*')):
        if path.is_file() and path != bundle and '__pycache__' not in path.parts and path.name != 'ibm_credentials.local.json':
            z.write(path, path.relative_to(out))
with zipfile.ZipFile(bundle) as z:
    assert z.testzip() is None
    print('Bundle:', len(z.namelist()), 'files,', bundle.stat().st_size, 'bytes')
print('Finalized executed notebook, HTML, report and portable bundle.')
