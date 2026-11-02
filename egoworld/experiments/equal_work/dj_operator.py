"""Data-Juicer custom operator; requires the optional isolated benchmark env."""
from data_juicer.ops.base_op import OPERATORS, Mapper
from equal_work.contract import canonical, evaluate


@OPERATORS.register_module("egoworld_hand_contract_mapper")
class EgoWorldHandContractMapper(Mapper):
    """Annotate every row so rejected rows remain auditable."""

    def process_single(self, sample):
        sample["decision"] = canonical(evaluate(sample))
        return sample
