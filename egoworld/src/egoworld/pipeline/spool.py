"""Bounded, lossless phase handoffs: one contiguous blob + metadata per block.

Only descriptors reach the driver. No pickle, escaping memmaps or compression
of hot tensors. Scratch is never a stage commit; a failed step restarts at input.
"""
from pathlib import Path
import json
import shutil
import uuid
import numpy as np
from egoworld.utils.hashing import sha256_file
from egoworld.utils.timing import phase


class Spool:
    def __init__(self, root, limit, block_limit):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.peak = 0
        self.limit, self.block_limit = int(limit), int(block_limit)

    @property
    def used(self):
        return sum(p.stat().st_size for p in self.root.glob('*/*') if p.is_file())

    def put(self, value):
        folder = self.root / uuid.uuid4().hex
        folder.mkdir()
        available = min(self.block_limit, self.limit - self.used)
        written = 0
        def encode(item, handle):
            nonlocal written
            if hasattr(item, 'detach'):
                from egoworld.operators.common import array
                item = array(item)
            if isinstance(item, np.ndarray):
                if item.dtype.hasobject or item.dtype.kind not in 'buifc':
                    raise ValueError('Only numeric/bool arrays are allowed in phase spools')
                if written + item.nbytes > available:
                    raise ValueError('Phase spool byte budget exceeded; increase explicit split budgets')
                spec = {'__array__': True, 'dtype': item.dtype.str, 'shape': list(item.shape),
                        'offset': written, 'bytes': item.nbytes}
                # ascontiguousarray can turn a scalar into shape (1,); preserve
                # its original shape in metadata and copy only its raw bytes.
                if item.nbytes:
                    handle.write(memoryview(np.ascontiguousarray(item)).cast('B'))
                written += item.nbytes
                return spec
            if isinstance(item, np.generic):
                return item.item()
            if isinstance(item, dict):
                if '__array__' in item:
                    raise ValueError('Reserved spool key')
                return {str(k): encode(v, handle) for k, v in item.items()}
            if isinstance(item, (tuple, list)):
                return [encode(v, handle) for v in item]
            return item
        try:
            with phase('spool_write'):
                blob = folder / 'arrays.dat'
                with blob.open('wb') as handle:
                    tree = encode(value, handle)
                metadata = json.dumps({'tree': tree, 'data_bytes': written,
                    'data_sha256': sha256_file(str(blob))}, allow_nan=False).encode()
                if written + len(metadata) > available:
                    raise ValueError('Phase spool byte budget exceeded; increase explicit split budgets')
                (folder / 'tree.json').write_bytes(metadata)
                checksum = sha256_file(str(folder / 'tree.json'))
            self.peak = max(self.peak, self.used)
            return {'block': folder.name, 'bytes': written + len(metadata), 'sha256': checksum}
        except BaseException:
            shutil.rmtree(folder)
            raise

    def _folder(self, descriptor):
        name = descriptor['block']
        if Path(name).name != name or not name or name in ('.', '..'):
            raise ValueError('Invalid phase block')
        return self.root / name

    def get(self, descriptor):
        folder = self._folder(descriptor)
        with phase('spool_read'):
            metadata = folder / 'tree.json'
            if sha256_file(str(metadata)) != descriptor['sha256']:
                raise ValueError('Corrupt phase metadata')
            content = json.loads(metadata.read_text())
            blob = folder / 'arrays.dat'
            size = blob.stat().st_size
            if size != content['data_bytes'] or sha256_file(str(blob)) != content['data_sha256']:
                raise ValueError('Corrupt phase array')
            mapping = np.memmap(blob, dtype=np.uint8, mode='r') if size else None
            def decode(item):
                if isinstance(item, dict) and '__array__' in item:
                    dtype, shape, offset = np.dtype(item['dtype']), tuple(item['shape']), item['offset']
                    if (dtype.hasobject or dtype.kind not in 'buifc' or any(not isinstance(n, int) or n < 0 for n in shape)
                            or not isinstance(offset, int) or offset < 0):
                        raise ValueError('Invalid phase array layout')
                    import math
                    nbytes = math.prod(shape) * dtype.itemsize
                    if nbytes != item['bytes'] or offset + nbytes > size:
                        raise ValueError('Invalid phase array bounds')
                    if not nbytes:
                        return np.empty(shape, dtype=dtype)
                    return np.array(np.ndarray(shape, dtype=dtype, buffer=mapping, offset=offset), copy=True)
                if isinstance(item, dict):
                    return {k: decode(v) for k, v in item.items()}
                if isinstance(item, list):
                    return [decode(v) for v in item]
                return item
            try:
                return decode(content['tree'])
            finally:
                if mapping is not None:
                    mapping._mmap.close()

    def discard(self, descriptor):
        shutil.rmtree(self._folder(descriptor), ignore_errors=True)

    def clear(self):
        shutil.rmtree(self.root, ignore_errors=True)


def spool_path(execution, work_dir, stage):
    if not stage or Path(stage).name != stage or stage in (".", ".."):
        raise ValueError("Invalid spool stage")
    if execution.get("split_directory"):
        import hashlib
        namespace = hashlib.sha256(str(Path(work_dir).resolve()).encode()).hexdigest()
        return Path(execution["split_directory"]) / namespace / stage
    return Path(work_dir) / ".phases" / stage


def release_spool(execution, work_dir, stage):
    root = spool_path(execution, work_dir, stage)
    shutil.rmtree(root, ignore_errors=True)
    try:
        root.parent.rmdir()  # only our empty namespace, never the user's base directory
    except OSError:
        pass
