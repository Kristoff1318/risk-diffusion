"""Record hashes for the current source and saved experimental artifacts."""
import hashlib,json,platform
from pathlib import Path
import torch,numpy,scipy,matplotlib

def main():
    root=Path('risk_diffusion');dest=root/'revision/outputs/manifest.json'
    files=list((root/'revision').glob('*.py'))+list((root/'revision').glob('*.sh'))+list(root.glob('*.md'))+[root/'ORIGINAL_PLAN.txt']
    files += [p for p in (root/'revision/outputs').rglob('*') if p.is_file() and p!=dest and p.suffix in {'.json','.npz','.pt','.png'}]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}
    record={'python':platform.python_version(),'torch':torch.__version__,'numpy':numpy.__version__,'scipy':scipy.__version__,
            'matplotlib':matplotlib.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'sha256':hashes,'note':'Content manifest, not a promise of bitwise reproduction across software/hardware.'}
    dest.write_text(json.dumps(record,indent=2));print('Recorded',len(hashes),'source and artifact hashes')
if __name__=='__main__':main()
