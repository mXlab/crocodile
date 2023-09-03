from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from simple_parsing import parse, ArgumentParser, ConflictResolution
from crocodile.utils.conditional_fields import WithConditionalFields, conditional_field


@dataclass
class CNNStack:
    name: str = "stack"
    num_layers: int = 3
    kernel_sizes: Tuple[int, int, int] = (7, 5, 5)
    num_filters: List[Tuple[int, int]] = field(
        default_factory=[(32, 16), (64, 64)].copy
    )


parser = ArgumentParser(conflict_resolution=ConflictResolution.ALWAYS_MERGE)
parser.add_arguments(CNNStack, dest="config", default=CNNStack())
args = parser.parse_args()
print(args)
