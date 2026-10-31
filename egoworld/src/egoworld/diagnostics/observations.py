"""Opt-in diagnostic observations inside the actual model environment."""
from functools import lru_cache
import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import sys
import threading
import time


def source_identity():
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(package.rglob('*.py')):
        digest.update(str(path.relative_to(package)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def model_source_identities():
    identities = {}
    for name in ('sam2', 'groundingdino', 'hamer', 'fast3r', 'estimater', 'dex_retargeting'):
        module = sys.modules.get(name)
        location = getattr(module, '__file__', None)
        if not location:
            continue
        source = Path(location).resolve()
        files = sorted(source.parent.rglob('*.py')) if source.name == '__init__.py' else [source]
        digest = hashlib.sha256()
        for path in files:
            digest.update(str(path.relative_to(source.parent)).encode())
            digest.update(path.read_bytes())
        identities[name] = {'path': str(source), 'python_source_sha256': digest.hexdigest()}
    return identities


@lru_cache(maxsize=1)
def runtime_identity():
    result = {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
            'source_tree_hash': source_identity(), 'model_sources': model_source_identities(),
            'packages': sorted((dist.metadata.get('Name', 'unknown'), dist.version)
                               for dist in importlib.metadata.distributions()),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():
        try:
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            result['gpu'] = {'name': props.name, 'total_memory_bytes': props.total_memory,
                             'compute_capability': [props.major, props.minor],
                             'uuid': str(getattr(props, 'uuid', 'unavailable')),
                             'torch_cuda_version': torch.version.cuda}
        except Exception as exc:
            result['gpu_identity_error'] = str(exc)
    return result


def process_tree_rss():
    """Linux live RSS sum (shared pages may be counted more than once)."""
    if not Path('/proc/self/statm').exists():
        return None
    parents, sizes = {}, {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            pid = int(path.parent.name)
            fields = path.read_text().rsplit(')', 1)[1].split()
            parents[pid] = int(fields[1])
            sizes[pid] = int((path.parent / 'statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
        except (OSError, ValueError, IndexError):
            continue
    family = {os.getpid()}
    while children := {pid for pid, parent in parents.items() if parent in family} - family:
        family.update(children)
    return sum(sizes.get(pid, 0) for pid in family)


class TrialObserver:
    """Sampling is advisory; unavailable NVML never masquerades as zero memory."""
    def __init__(self, gpu, interval=.1):
        self.gpu, self.interval = gpu, interval
        self.stop = threading.Event()
        self.thread = None
        self.nvml = self.handle = None
        self.data = {'sample_interval_seconds': interval, 'samples': 0,
                     'process_tree_peak_rss_bytes': None, 'device_used_peak_bytes': None,
                     'device_free_min_bytes': None, 'device_total_bytes': None,
                     'device_observation': 'not_requested' if not gpu else 'unavailable',
                     'scope': 'Sampled process-tree RSS sum and whole-device memory; shared pages/other processes may be included and short peaks may be missed.'}

    def _synchronize(self):
        torch = sys.modules.get('torch')
        if self.gpu and torch is not None and torch.cuda.is_initialized():
            torch.cuda.synchronize()

    def __enter__(self):
        self._synchronize()
        if self.gpu:
            try:
                import pynvml
                self.nvml = pynvml
                pynvml.nvmlInit()
                visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0].strip()
                # CUDA numeric ordinals do not necessarily match NVML ordinals.
                if visible.startswith(('GPU-', 'MIG-')):
                    self.handle = pynvml.nvmlDeviceGetHandleByUUID(visible)
                self.data.update(driver_version=pynvml.nvmlSystemGetDriverVersion())
                for key in ('device_uuid', 'driver_version'):
                    if isinstance(self.data.get(key), bytes):
                        self.data[key] = self.data[key].decode()
            except Exception as exc:
                self.data['device_observation_error'] = str(exc)
                self.handle = None
        self._sample()
        self.data['device_baseline_used_bytes'] = self.data['device_used_peak_bytes']
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        self.started = time.perf_counter()
        return self

    def _sample(self):
        if self.nvml is not None and self.handle is None:
            try:
                torch = sys.modules.get('torch')
                if torch is not None and torch.cuda.is_initialized():
                    props = torch.cuda.get_device_properties(torch.cuda.current_device())
                    uuid = str(getattr(props, 'uuid', ''))
                    if uuid and not uuid.startswith(('GPU-', 'MIG-')):
                        from uuid import UUID
                        uuid = 'GPU-' + str(UUID(uuid))
                    if uuid.startswith(('GPU-', 'MIG-')):
                        self.handle = self.nvml.nvmlDeviceGetHandleByUUID(uuid)
            except Exception as exc:
                self.data['device_observation_error'] = str(exc)
        try:
            rss = process_tree_rss()
            if rss is not None:
                self.data['process_tree_peak_rss_bytes'] = max(self.data['process_tree_peak_rss_bytes'] or 0, rss)
        except Exception as exc:
            self.data['rss_observation_error'] = str(exc)
        if self.handle is not None:
            try:
                info = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
                if self.data['device_observation'] != 'partial':
                    self.data['device_observation'] = 'nvml'
                uuid = self.nvml.nvmlDeviceGetUUID(self.handle)
                self.data['device_uuid'] = uuid.decode() if isinstance(uuid, bytes) else uuid
                self.data['device_total_bytes'] = int(info.total)
                self.data['device_used_peak_bytes'] = max(self.data['device_used_peak_bytes'] or 0, int(info.used))
                previous = self.data['device_free_min_bytes']
                self.data['device_free_min_bytes'] = min(previous if previous is not None else int(info.free), int(info.free))
            except Exception as exc:
                self.data['device_observation'] = 'partial'
                self.data['device_observation_error'] = str(exc)
        self.data['samples'] += 1

    def _loop(self):
        while not self.stop.wait(self.interval):
            self._sample()

    def finish(self):
        try:
            self._synchronize()
        except Exception as exc:
            self.data['synchronization_error'] = str(exc)
        self.data['synchronized_stage_wall_seconds'] = time.perf_counter() - self.started
        self.stop.set()
        self.thread.join()
        self._sample()
        if self.nvml is not None:
            try:
                self.nvml.nvmlShutdown()
            except Exception:
                pass
        return self.data
