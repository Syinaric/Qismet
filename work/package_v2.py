from pathlib import Path
import json, hashlib, zipfile
import nbformat

root=Path(__file__).resolve().parents[1]/'outputs'
for relative in ['diagnostics_v2/QUBO_Diagnostic.ipynb','fashion_v2/Fashion_Qismet.ipynb']:
    nb=nbformat.read(root/relative,as_version=4)
    nbformat.validate(nb)
    cells=[c for c in nb.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in cells)
    assert not [o for c in cells for o in c.outputs if o.output_type=='error']
    print(relative,':',len(cells),'executed cells, no errors')
files=[p for p in root.rglob('*') if p.is_file() and p.suffix!='.zip' and p.name!='ibm_credentials.local.json'
       and '__pycache__' not in p.parts and 'data_cache' not in p.parts]
manifest={str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
          for p in files if p.name!='v2_manifest_sha256.json'}
(root/'v2_manifest_sha256.json').write_text(json.dumps(manifest,indent=2))
files=[p for p in root.rglob('*') if p.is_file() and p.suffix!='.zip' and p.name!='ibm_credentials.local.json'
       and '__pycache__' not in p.parts and 'data_cache' not in p.parts]
bundle=root/'Qismet_v2_complete.zip'
with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
    for p in files: z.write(p,p.relative_to(root))
with zipfile.ZipFile(bundle) as z: assert z.testzip() is None
print('Packaged',len(files),'files:',bundle.stat().st_size,'bytes')
