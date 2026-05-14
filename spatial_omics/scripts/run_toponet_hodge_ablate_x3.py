from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from spatial_omics.models.toponet_hodge import run_toponet_hodge_study, save_toponet_hodge_results
from spatial_omics.config import load_config, TopoNetHodgeConfig


ABLATIONS = (
    ("x21_geom_only", True, False, False),
    ("x21_ortho_only", False, True, False),
    ("x21_morse_only", False, False, True),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the x21 surgical TopoNet-Hodge ablations.")
    parser.add_argument("--config", required=True, help="Path to the TopoNetHodgeConfig yaml file.")
    args = parser.parse_args()

    base_cfg = load_config(args.config, TopoNetHodgeConfig)

    root_output_dir = Path(base_cfg.output_dir)
    for variant_name, use_geometric_weights, use_orthogonal_restrictions, use_morse_gating in ABLATIONS:
        cfg = replace(
            base_cfg,
            output_dir=str(root_output_dir / variant_name),
            use_geometric_weights=use_geometric_weights,
            use_orthogonal_restrictions=use_orthogonal_restrictions,
            use_morse_gating=use_morse_gating,
        )
        results = run_toponet_hodge_study(cfg)
        results["ablation"] = {
            "name": variant_name,
            "use_geometric_weights": use_geometric_weights,
            "use_orthogonal_restrictions": use_orthogonal_restrictions,
            "use_morse_gating": use_morse_gating,
        }
        save_toponet_hodge_results(results, cfg.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
