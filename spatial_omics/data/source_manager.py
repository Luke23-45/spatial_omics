from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import urlopen, urlretrieve

from spatial_omics.data.catalog import PROJECT_ROOT, DatasetResolution, get_catalog_entry, known_dataset_names, resolve_catalog_entry
from spatial_omics.utils.io import ensure_dir


SOURCE_MANIFEST_FILENAME = "source_state.json"


@dataclass(frozen=True)
class DatasetSourceState:
    name: str
    dataset_name: str
    adapter: str
    status: str
    modality: str
    disease: str
    resolution: str
    task_scope: str
    study_dir: str | None
    input_dir: str | None
    source_uris: tuple[str, ...]
    notes: tuple[str, ...]


def dataset_source_state(name: str) -> DatasetSourceState:
    entry = get_catalog_entry(name)
    resolution = resolve_catalog_entry(name)
    return DatasetSourceState(
        name=entry.name,
        dataset_name=resolution.dataset_name,
        adapter=resolution.adapter,
        status=resolution.status,
        modality=entry.modality,
        disease=entry.disease,
        resolution=entry.resolution,
        task_scope=entry.task_scope,
        study_dir=resolution.study_dir,
        input_dir=resolution.input_dir,
        source_uris=tuple(resolution.source_uris),
        notes=tuple(resolution.notes),
    )


def source_inventory(dataset_names: list[str] | None = None) -> list[DatasetSourceState]:
    names = dataset_names or known_dataset_names()
    return [dataset_source_state(name) for name in names]


def write_source_inventory(output_path: str | Path, dataset_names: list[str] | None = None) -> Path:
    output_path = Path(output_path)
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": [asdict(item) for item in source_inventory(dataset_names)],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def write_dataset_source_state(output_dir: str | Path, resolution: DatasetResolution, *, source_id: str | None = None) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_id": source_id,
        "resolution": {
            "name": resolution.name,
            "dataset_name": resolution.dataset_name,
            "adapter": resolution.adapter,
            "status": resolution.status,
            "study_dir": resolution.study_dir,
            "input_dir": resolution.input_dir,
            "source_uris": list(resolution.source_uris),
            "notes": list(resolution.notes),
        },
    }
    path = output_dir / SOURCE_MANIFEST_FILENAME
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def ensure_catalog_source_available(name: str) -> DatasetResolution:
    resolution = resolve_catalog_entry(name)
    if resolution.status != "missing":
        return resolution
    if name == "keren_tnbc":
        _download_keren_tnbc()
        return resolve_catalog_entry(name)
    return resolution


def _download_keren_tnbc() -> Path:
    article_url = "https://api.figshare.com/v2/articles/26068006"
    article_payload = json.loads(urlopen(article_url).read().decode("utf-8"))
    files = article_payload.get("files", [])
    if not files:
        raise RuntimeError("Figshare article for Keren TNBC did not return any files.")
    download_url = str(files[0]["download_url"])
    filename = str(files[0]["name"])
    target_dir = ensure_dir(PROJECT_ROOT / "data" / "spatial_omics" / "raw" / "keren_tnbc")
    target_path = target_dir / filename
    if not target_path.is_file():
        urlretrieve(download_url, target_path)
    return target_path


__all__ = [
    "DatasetSourceState",
    "SOURCE_MANIFEST_FILENAME",
    "ensure_catalog_source_available",
    "dataset_source_state",
    "source_inventory",
    "write_dataset_source_state",
    "write_source_inventory",
]
