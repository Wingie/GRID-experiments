"""Build a hierarchical category graph over the parsed RimWorld entities.

The graph mirrors the conceptual hierarchy in the project spec §3a
(Items / Buildings / Research / Jobs / Events -> sub-categories -> specific defs). It is
used (a) as a human-readable prior when inspecting SID clusters, and (b) to provide a
coarse category label per entity for the ``eval_semantic_ids`` hierarchical-consistency
metric. It does *not* force SID prefixes — semantic IDs remain per-entity and independent
(TIGER-style), with the graph used only for measurement.

Offline-verifiable: pure dict/string logic over already-parsed entities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from rimworld_agent.utils import cfg_get, get_logger, write_json

log = get_logger("build_entity_graph")

# Top-level category inference from def_type + fields.
_RESEARCH = "Research"
_JOBS = "Jobs"
_EVENTS = "Events"
_ITEMS = "Items"
_BUILDINGS = "Buildings"
_MISC = "Misc"


@dataclass
class GraphNode:
    category: str  # top-level (Items, Buildings, ...)
    subcategory: str  # finer bucket (Resources, Weapons, Power, ...)
    def_name: str
    def_type: str

    def path(self) -> tuple[str, str, str]:
        return (self.category, self.subcategory, self.def_name)


def _thing_subcategory(fields: dict[str, Any]) -> tuple[str, str]:
    """Map a ThingDef's fields to (top_category, subcategory)."""
    cats = fields.get("thingCategories") or []
    cats = [c.lower() for c in cats] if isinstance(cats, list) else [str(cats).lower()]
    text = " ".join(cats)
    # Buildings vs items: ThingDef with a building component or category.
    if "building" in fields or "Building" in str(fields.get("thingClass", "")):
        if any(k in text for k in ("power", "battery", "generator", "solar")):
            return _BUILDINGS, "Power"
        if any(k in text for k in ("production", "workbench", "smelter")):
            return _BUILDINGS, "Production"
        if any(k in text for k in ("security", "turret", "sandbag")):
            return _BUILDINGS, "Security"
        if any(k in text for k in ("furniture", "bed", "table", "chair")):
            return _BUILDINGS, "Furniture"
        return _BUILDINGS, "Structure"
    if any(k in text for k in ("weapon", "gun", "melee")):
        return _ITEMS, "Weapons"
    if any(k in text for k in ("apparel", "clothing", "armor")):
        return _ITEMS, "Apparel"
    if any(k in text for k in ("food", "meal", "meat", "raw")):
        return _ITEMS, "Meals"
    if any(k in text for k in ("resource", "manufactured", "stuff", "metal", "wood")):
        return _ITEMS, "Resources"
    return _ITEMS, "Other"


def _categorize(def_type: str, fields: dict[str, Any]) -> tuple[str, str]:
    if def_type == "ResearchProjectDef":
        tech = str(fields.get("techLevel", "Neolithic"))
        return _RESEARCH, tech
    if def_type == "WorkTypeDef":
        return _JOBS, "WorkType"
    if def_type == "JobDef":
        return _JOBS, "Job"
    if def_type in ("IncidentDef", "QuestScriptDef"):
        cat = str(fields.get("category", ""))
        sub = "Threats" if "Threat" in cat or "Raid" in str(fields.get("defName", "")) else "Opportunities"
        return _EVENTS, sub
    if def_type == "ThingDef":
        return _thing_subcategory(fields)
    return _MISC, def_type


@dataclass
class EntityGraph:
    nodes: dict[str, GraphNode] = field(default_factory=dict)  # def_name -> node

    def category_of(self, def_name: str) -> str:
        node = self.nodes.get(def_name)
        return node.category if node else _MISC

    def subcategory_of(self, def_name: str) -> str:
        node = self.nodes.get(def_name)
        return node.subcategory if node else _MISC

    def as_tree(self) -> dict[str, dict[str, list[str]]]:
        tree: dict[str, dict[str, list[str]]] = {}
        for node in self.nodes.values():
            tree.setdefault(node.category, {}).setdefault(node.subcategory, []).append(node.def_name)
        for cat in tree.values():
            for sub in cat.values():
                sub.sort()
        return tree


def build_entity_graph(entities: Iterable) -> EntityGraph:
    """Build the :class:`EntityGraph` from parsed :class:`RimWorldEntity` objects."""
    graph = EntityGraph()
    for e in entities:
        cat, sub = _categorize(e.def_type, getattr(e, "fields", {}) or {})
        graph.nodes[e.def_name] = GraphNode(cat, sub, e.def_name, e.def_type)
    log.info("graph: %d entities across %d top categories", len(graph.nodes), len(graph.as_tree()))
    return graph


def main() -> None:
    import hydra
    from omegaconf import DictConfig

    from rimworld_agent.knowledge.extract_defs import DEFAULT_DEF_TYPES, extract_defs

    @hydra.main(version_base="1.3", config_path="../../configs", config_name="base.yaml")
    def _run(cfg: DictConfig) -> None:
        entities = extract_defs(
            cfg_get(cfg, "paths.xml_defs_dir", "data/rimworld_xml_defs"),
            cfg_get(cfg, "knowledge.def_types", DEFAULT_DEF_TYPES),
        )
        graph = build_entity_graph(entities)
        out = cfg_get(cfg, "paths.entity_graph_file", "results/entity_graph.json")
        write_json(graph.as_tree(), out)
        log.info("wrote entity graph -> %s", out)

    _run()


if __name__ == "__main__":
    main()
