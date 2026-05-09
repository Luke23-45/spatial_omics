"""AnnData compatibility layer.

The canonical container for this package is ``AnnData``.  When the
optional ``anndata`` dependency is unavailable we provide a minimal local
stand-in that supports the subset of behavior used by the feasibility
pipeline and tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:  # pragma: no cover - exercised implicitly when dependency exists
    from anndata import AnnData as _AnnData

    AnnData = _AnnData
    HAS_ANNDATA = True
except ImportError:  # pragma: no cover - fallback is exercised in tests
    HAS_ANNDATA = False

    @dataclass
    class AnnData:
        """Small fallback implementation.

        Only the pieces used in this repository are supported.
        """

        X: np.ndarray
        obs: pd.DataFrame
        var: pd.DataFrame
        obsm: dict[str, Any]
        uns: dict[str, Any]

        def __init__(
            self,
            X: np.ndarray | None = None,
            obs: pd.DataFrame | None = None,
            var: pd.DataFrame | None = None,
            obsm: dict[str, Any] | None = None,
            uns: dict[str, Any] | None = None,
        ) -> None:
            self.X = np.asarray(X) if X is not None else np.zeros((0, 0), dtype=np.float32)
            self.obs = obs.copy() if obs is not None else pd.DataFrame()
            self.var = var.copy() if var is not None else pd.DataFrame()
            self.obsm = dict(obsm or {})
            self.uns = dict(uns or {})

        @property
        def n_obs(self) -> int:
            return int(self.obs.shape[0])

        @property
        def n_vars(self) -> int:
            return int(self.X.shape[1]) if self.X.ndim == 2 else 0

        def copy(self) -> "AnnData":
            return AnnData(
                X=np.array(self.X, copy=True),
                obs=self.obs.copy(),
                var=self.var.copy(),
                obsm={k: np.array(v, copy=True) if isinstance(v, np.ndarray) else v for k, v in self.obsm.items()},
                uns={k: v for k, v in self.uns.items()},
            )


__all__ = ["AnnData", "HAS_ANNDATA"]
