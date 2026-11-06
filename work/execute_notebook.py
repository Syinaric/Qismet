from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

root = Path(__file__).resolve().parents[1]
path = root / 'outputs' / 'Qismet.ipynb'
nb = nbformat.read(path, as_version=4)
def started(cell, cell_index, **kwargs):
    print(f'Executing cell {cell_index}', flush=True)
def finished(cell, cell_index, **kwargs):
    for output in cell.get('outputs', []):
        if output.output_type == 'stream':
            print(output.text, flush=True)
client = NotebookClient(nb, timeout=1200, kernel_name='python3',
                        resources={'metadata': {'path': str(path.parent)}},
                        on_cell_start=started, on_cell_executed=finished)
try:
    client.execute()
finally:
    nbformat.write(nb, path)
exporter = HTMLExporter()
body, resources = exporter.from_notebook_node(nb)
(path.parent / 'Qismet.html').write_text(body, encoding='utf-8')
print('Notebook execution and HTML export completed.', flush=True)
