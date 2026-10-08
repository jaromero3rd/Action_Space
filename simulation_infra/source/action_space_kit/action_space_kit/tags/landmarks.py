"""AprilTag landmarks (from the real-world map) for use in simulation training.

`tag_landmarks.yaml` is generated from drone_deployment_infra/config/map.yaml: each tag's
size and position in the tag-10 origin frame (metres; x right, y up, z out of the tag
face). Use `positions()` / `load_landmarks()` to place or reference the tags in a sim
task, and `marker_cfg()` to draw them (Isaac is imported lazily, so this module loads
fine without Isaac for tooling/tests).
"""

from __future__ import annotations

from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
LANDMARKS_YAML = _HERE / "tag_landmarks.yaml"


def load_landmarks(path: Path | None = None) -> dict:
    data = yaml.safe_load((Path(path) if path else LANDMARKS_YAML).read_text())
    tags = {int(k): {"size_m": float(v["size_m"]), "pos": tuple(float(x) for x in v["pos"])}
            for k, v in data["tags"].items()}
    return {"origin_tag": int(data["origin_tag"]), "tags": tags}


def positions(path: Path | None = None) -> dict[int, tuple[float, float, float]]:
    """{tag_id: (x, y, z)} in the tag-10 origin frame, metres."""
    return {tid: t["pos"] for tid, t in load_landmarks(path)["tags"].items()}


def marker_cfg(prim_path: str = "/Visuals/tag_landmarks"):
    """Isaac VisualizationMarkersCfg: one flat cuboid prototype per tag (sized to the tag).

    Isaac is imported here, not at module import, so the rest of this module works without
    it. In a task: VisualizationMarkers(marker_cfg()).visualize(translations, marker_indices).
    """
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkersCfg

    markers = {}
    for tid, t in load_landmarks()["tags"].items():
        s = t["size_m"] or 0.13
        markers[f"tag_{tid}"] = sim_utils.CuboidCfg(
            size=(s, s, 0.01),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95)),
        )
    return VisualizationMarkersCfg(prim_path=prim_path, markers=markers)
