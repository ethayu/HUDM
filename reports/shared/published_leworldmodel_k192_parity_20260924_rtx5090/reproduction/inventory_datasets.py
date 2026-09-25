"""Inventory expected Lance fragments; hash only completely received, stable files."""
import datetime
import hashlib
import json
from pathlib import Path
import lance

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'source_snapshot/data/upstream'
OUTPUT = ROOT / 'local_dataset_inventory.json'
NAMES = ('ogb_cube_single_expert.lance', 'reacher.lance', 'pusht_expert_train.lance', 'tworoom.lance')
previous = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {}
cached = {record['relative_path']: record for ds in previous.get('datasets', []) for record in ds.get('files', [])}
results = []
for name in NAMES:
    path = DATA / name
    if not path.is_dir():
        results.append({'name': name, 'status': 'not_received', 'files': []})
        continue
    try:
        dataset = lance.dataset(str(path))
        files = []
        for fragment in dataset.get_fragments():
            for datafile in fragment.metadata.files:
                target = path / 'data' / datafile.path
                relative = str(target.relative_to(ROOT / 'source_snapshot'))
                record = {'relative_path': relative, 'expected_bytes': datafile.file_size_bytes, 'present_bytes': 0}
                if target.is_file():
                    before = target.stat()
                    record.update(present_bytes=before.st_size, mtime_ns=before.st_mtime_ns, ctime_ns=before.st_ctime_ns, inode=before.st_ino)
                    old = cached.get(relative, {})
                    if before.st_size == datafile.file_size_bytes:
                        if old.get('present_bytes') == before.st_size and old.get('mtime_ns') == before.st_mtime_ns and old.get('ctime_ns') == before.st_ctime_ns and old.get('inode') == before.st_ino and old.get('sha256'):
                            record['sha256'] = old['sha256']
                        else:
                            with target.open('rb') as stream:
                                checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
                            after = target.stat()
                            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino):
                                record['sha256'] = checksum
                            else:
                                record['status'] = 'changed_during_hash'
                files.append(record)
        complete = bool(files) and all(f.get('sha256') and f['present_bytes'] == f['expected_bytes'] for f in files)
        results.append({'name': name, 'version': dataset.version, 'rows': dataset.count_rows(),
                        'status': 'complete_bytes_pending_source_hash_comparison' if complete else 'receiving',
                        'expected_data_bytes': sum(f['expected_bytes'] or 0 for f in files),
                        'present_data_bytes': sum(f['present_bytes'] for f in files), 'files': files})
    except Exception as exc:
        results.append({'name': name, 'status': 'metadata_not_ready', 'error': str(exc), 'files': []})
payload = {'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'source_hashes_compared': False, 'datasets': results}
temp = OUTPUT.with_suffix('.tmp')
temp.write_text(json.dumps(payload, indent=2) + '\n')
temp.replace(OUTPUT)
print(json.dumps([{k: ds[k] for k in ('name', 'status', 'expected_data_bytes', 'present_data_bytes') if k in ds} for ds in results], indent=2))
