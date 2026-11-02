"""Optional matched-model OP for Data-Juicer custom_operator_paths.

Import only in a Data-Juicer environment. The native model runner, prepared
clip, upstream receipts and Writer are the same ones used by EgoWorld. This
does not substitute HaWoR/MegaSaM for HaMeR/Fast3R or claim measured speedups.
"""
from dataclasses import asdict
from pathlib import Path

from data_juicer.ops.base_op import OPERATORS, Mapper


@OPERATORS.register_module("egoworld_stage_mapper")
class EgoWorldStageMapper(Mapper):
    _batched_op = False
    _NON_FINGERPRINT_ATTRS = getattr(Mapper, "_NON_FINGERPRINT_ATTRS", frozenset()) | {"_actor"}

    def __getstate__(self):
        return {**self.__dict__, "_actor": None}

    def __init__(self, config_path, stage, output_root, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from egoworld.config import load_config
        from egoworld.pipeline.stages import OPERATORS as NATIVE
        if stage not in NATIVE:
            raise ValueError(f"Unknown EgoWorld stage {stage}")
        self.config = asdict(load_config(config_path).resolved())
        self.stage, self.output_root = stage, str(Path(output_root).resolve())
        self._actor = None

    def process_single(self, sample, context=False):
        from egoworld.pipeline.stages import ModelActor
        from egoworld.pipeline.artifacts import WriterActor
        row = dict(sample)
        clip = row["egoworld_clip"]
        for key in ("clip_id", "video_id"):
            value = clip[key]
            if not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value):
                raise ValueError(f"Invalid {key}")
        folder = Path(self.output_root) / f"video_id={clip['video_id']}" / f"clip_id={clip['clip_id']}"
        job = {"stage": self.stage, "clip": clip, "directory": str(folder),
               "work_dir": str(folder / ".work" / self.stage), "fingerprint": row["egoworld_fingerprint"]}
        Path(job["work_dir"]).mkdir(parents=True, exist_ok=True)
        if self._actor is None:
            self._actor = ModelActor(self.config)
        packet = self._actor.process(job, row["egoworld_prepared"], row.get("egoworld_inputs", {}))
        receipt = WriterActor(self.config).write_stage(job, packet)
        row["egoworld_inputs"] = {**row.get("egoworld_inputs", {}), self.stage: receipt}
        return row

    def close(self):
        if self._actor is not None:
            self._actor.close()
            self._actor = None

    def __del__(self):
        if hasattr(self, "_actor"):
            self.close()
