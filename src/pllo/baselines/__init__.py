"""Stage 7.5c - direct prior-work primitive implementations.

Each baseline self-declares which primitive it implements from a known
paper formula, which features it does NOT implement, and whether it can
be runtime-compared to our scheme. The module never substitutes a
generic family-level proxy for a missing primitive.
"""

from pllo.baselines.baseline_protocol import (
    BaselineProtocol,
    BaselineSelfDeclaration,
    UnsupportedResult,
)
from pllo.baselines.amulet import (
    AmuletConfig,
    AmuletStaticPHQ,
    ours_right_mask_kv_append,
)
from pllo.baselines.arrow import ArrowConfig, ArrowDirectPrimitive
from pllo.baselines.cryptonets import (
    CryptoNetsArithmeticSkeleton,
    CryptoNetsConfig,
)
from pllo.baselines.darknight import (
    DarKnightBlindingPrimitive,
    DarKnightConfig,
)
from pllo.baselines.gazelle_costed import (
    CostModelBaseline,
    CostModelEntry,
    DelphiCostModel,
    GazelleCostModel,
    MiniONNCostModel,
    SecureMLCostModel,
)
from pllo.baselines.slalom import SlalomConfig, SlalomDelegatedLinear

__all__ = [
    "BaselineProtocol",
    "BaselineSelfDeclaration",
    "UnsupportedResult",
    "AmuletConfig",
    "AmuletStaticPHQ",
    "ours_right_mask_kv_append",
    "ArrowConfig",
    "ArrowDirectPrimitive",
    "CryptoNetsArithmeticSkeleton",
    "CryptoNetsConfig",
    "DarKnightBlindingPrimitive",
    "DarKnightConfig",
    "CostModelBaseline",
    "CostModelEntry",
    "DelphiCostModel",
    "GazelleCostModel",
    "MiniONNCostModel",
    "SecureMLCostModel",
    "SlalomConfig",
    "SlalomDelegatedLinear",
]
