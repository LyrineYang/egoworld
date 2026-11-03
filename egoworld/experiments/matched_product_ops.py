"""Explicit shared-kernel Data-Juicer recipe operators, not the native baseline.

DefaultExecutor owns the OP chain. Production prepare, model and Writer APIs
are shared to isolate execution cost without substituting predicted outputs.
"""
from pathlib import Path
import json

from data_juicer.ops.base_op import OPERATORS, Mapper
from egoworld.product.datajuicer_adapter import EgoWorldStageMapper


@OPERATORS.register_module("egoworld_matched_prepare_mapper")
class MatchedPrepareMapper(Mapper):
    def __init__(self, config: dict = None, work_root: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config, self.work_root = config, work_root

    def process_single(self, sample):
        from egoworld.pipeline.stages import PrepareActor
        row = dict(sample)
        state = json.loads(row["egoworld_state"])
        clip = state["egoworld_clip"]
        job = {"clip": clip, "work_dir": str(Path(self.work_root) / clip["clip_id"])}
        state["egoworld_prepared"] = PrepareActor(self.config).prepare(job)
        row["egoworld_state"] = json.dumps(state, allow_nan=False)
        return row


@OPERATORS.register_module("egoworld_matched_stage_mapper")
class MatchedStageMapper(EgoWorldStageMapper):
    def process_single(self, sample, context=False):
        row = dict(sample)
        state = super().process_single(json.loads(row["egoworld_state"]), context)
        # Receipt stats acquire CUDA observation keys after the first request.
        # Keep their complete JSON rather than forcing heterogeneous metadata
        # into Arrow's first-row-inferred struct schema. Model arrays stay on disk.
        row["egoworld_state"] = json.dumps(state, allow_nan=False)
        return row

    def run(self, dataset, **kwargs):
        # HF can run a one-process map in this process. Release the previous
        # stage's CUDA runner before loading the next model, also on failure.
        try:
            return super().run(dataset, **kwargs)
        finally:
            self.close()


@OPERATORS.register_module("egoworld_matched_finalize_mapper")
class MatchedFinalizeMapper(Mapper):
    def __init__(self, config: dict = None, output_root: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config, self.output_root = config, output_root

    def process_single(self, sample):
        from egoworld.pipeline.artifacts import WriterActor
        row = dict(sample)
        state = json.loads(row["egoworld_state"])
        clip = state["egoworld_clip"]
        directory = Path(self.output_root) / f"video_id={clip['video_id']}" / f"clip_id={clip['clip_id']}"
        meta = WriterActor(self.config).finalize({"clip": clip, "directory": str(directory),
            "fingerprint": state["egoworld_fingerprint"]}, state["egoworld_inputs"], {})
        state["egoworld_final_status"] = meta["status"]
        row["egoworld_state"] = json.dumps(state, allow_nan=False)
        return row
