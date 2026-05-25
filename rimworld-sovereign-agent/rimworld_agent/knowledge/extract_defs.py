"""Parse RimWorld XML ``Def`` files into structured :class:`RimWorldEntity` records.

RimWorld's entire content lives in XML Defs (``Mods/Core/Defs/``). Each Def maps to a C#
class. Defs use single inheritance via the ``ParentName`` attribute referencing a parent's
``Name`` attribute; parents marked ``Abstract="True"`` are templates that are not emitted
as real entities. We resolve the full inheritance chain so every concrete entity carries
the merged field set of all its ancestors (gotcha #7 in the project spec).

Offline-verifiable: this module needs only ``lxml`` and a directory of XML files; it does
not require a GPU, network, or a RimWorld install. See ``tests`` for a fixture-based run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lxml import etree

from rimworld_agent.utils import cfg_get, ensure_dir, get_logger, write_json

log = get_logger("extract_defs")

# The Def categories we care about (project spec §2a). ``None`` def_types == all.
DEFAULT_DEF_TYPES = [
    "ThingDef",
    "ResearchProjectDef",
    "RecipeDef",
    "WorkTypeDef",
    "JobDef",
    "BiomeDef",
    "ThoughtDef",
    "HediffDef",
    "IncidentDef",
    "QuestScriptDef",
    "TaleDef",
    "TraitDef",
    "SkillDef",
]

# Best-effort engine-namespace attribution for common Def types. Most gameplay Defs are
# defined by the RimWorld assembly; a handful of structural ones live in Verse.
_VERSE_DEF_TYPES = {"ThingDef", "JobDef", "TerrainDef", "BiomeDef"}


@dataclass
class RimWorldEntity:
    def_name: str
    def_type: str
    label: str = ""
    description: str = ""
    parent_def: str | None = None  # the ParentName attribute value (a parent's Name)
    fields: dict[str, Any] = field(default_factory=dict)
    file_path: str = ""
    namespace: str = "RimWorld"
    children: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "def_name": self.def_name,
            "def_type": self.def_type,
            "label": self.label,
            "description": self.description,
            "parent_def": self.parent_def,
            "fields": self.fields,
            "file_path": self.file_path,
            "namespace": self.namespace,
            "children": self.children,
            "references": self.references,
        }

    def text_blob(self) -> str:
        """A flat text rendering used for token mining and entity embedding."""
        parts = [f"{self.def_type} {self.def_name}", self.label, self.description]
        for k, v in self.fields.items():
            parts.append(f"{k}: {v}")
        return "\n".join(p for p in parts if p)


# --------------------------------------------------------------------------- #
# XML -> python
# --------------------------------------------------------------------------- #
def _elem_to_value(elem: etree._Element) -> Any:
    """Recursively convert an XML element to a python value.

    Leaf -> its text; an element with ``<li>`` children -> a list; otherwise a dict keyed
    by child tag. ``Class`` / ``Inherit`` style attributes are folded in under ``@attr``.
    """
    children = [c for c in elem if isinstance(c.tag, str)]
    if not children:
        text = (elem.text or "").strip()
        return text
    li_children = [c for c in children if c.tag == "li"]
    if li_children and len(li_children) == len(children):
        return [_elem_to_value(c) for c in li_children]
    out: dict[str, Any] = {}
    for c in children:
        out[c.tag] = _elem_to_value(c)
    return out


@dataclass
class _RawDef:
    def_type: str
    name: str | None  # the ``Name`` attribute (parent handle)
    parent: str | None  # the ``ParentName`` attribute
    abstract: bool
    fields: dict[str, Any]
    file_path: str


def _load_raw_defs(xml_dir: str | Path, def_types: list[str] | None) -> list[_RawDef]:
    xml_dir = Path(xml_dir)
    allowed = set(def_types) if def_types else None
    raws: list[_RawDef] = []
    parser = etree.XMLParser(remove_comments=True, recover=True)
    for path in sorted(xml_dir.rglob("*.xml")):
        try:
            tree = etree.parse(str(path), parser)
        except etree.XMLSyntaxError as exc:
            log.warning("skipping unparsable %s: %s", path, exc)
            continue
        root = tree.getroot()
        if root is None:
            continue
        for elem in root:
            if not isinstance(elem.tag, str):
                continue
            def_type = elem.tag
            if allowed is not None and def_type not in allowed:
                continue
            fields = _elem_to_value(elem)
            if not isinstance(fields, dict):
                fields = {}
            raws.append(
                _RawDef(
                    def_type=def_type,
                    name=elem.get("Name"),
                    parent=elem.get("ParentName"),
                    abstract=str(elem.get("Abstract", "")).lower() == "true",
                    fields=fields,
                    file_path=str(path),
                )
            )
    return raws


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge ``override`` onto a copy of ``base`` (child wins; lists are replaced)."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_fields(raw: _RawDef, by_name: dict[str, _RawDef], _seen: set[str] | None = None) -> dict:
    """Merge ``raw``'s fields onto its fully-resolved parent chain."""
    if raw.parent is None or raw.parent not in by_name:
        return dict(raw.fields)
    _seen = _seen or set()
    if raw.parent in _seen:
        log.warning("cyclic ParentName at %s -> %s", raw.name, raw.parent)
        return dict(raw.fields)
    _seen.add(raw.parent)
    parent_fields = _resolve_fields(by_name[raw.parent], by_name, _seen)
    return _deep_merge(parent_fields, raw.fields)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def extract_defs(
    xml_dir: str | Path,
    def_types: list[str] | None = None,
    include_abstract: bool = False,
) -> list[RimWorldEntity]:
    """Parse all Defs under ``xml_dir`` and return resolved :class:`RimWorldEntity` list."""
    raws = _load_raw_defs(xml_dir, def_types)
    by_name = {r.name: r for r in raws if r.name}

    # children index: parent Name -> list of child defNames
    children_of: dict[str, list[str]] = {}
    for r in raws:
        if r.parent:
            dn = str(r.fields.get("defName", "")).strip()
            if dn:
                children_of.setdefault(r.parent, []).append(dn)

    entities: list[RimWorldEntity] = []
    for r in raws:
        if r.abstract and not include_abstract:
            continue
        resolved = _resolve_fields(r, by_name)
        def_name = str(resolved.get("defName", r.name or "")).strip()
        if not def_name:
            continue
        ns = "Verse" if r.def_type in _VERSE_DEF_TYPES else "RimWorld"
        entities.append(
            RimWorldEntity(
                def_name=def_name,
                def_type=r.def_type,
                label=str(resolved.get("label", "")).strip(),
                description=str(resolved.get("description", "")).strip(),
                parent_def=r.parent,
                fields=resolved,
                file_path=r.file_path,
                namespace=ns,
                children=children_of.get(r.name or "__none__", []),
            )
        )

    _attach_references(entities)
    log.info("extracted %d entities from %s", len(entities), xml_dir)
    return entities


def _attach_references(entities: list[RimWorldEntity]) -> None:
    """Populate ``references``: other defNames mentioned anywhere in an entity's fields."""
    known = {e.def_name for e in entities}

    def _collect(value: Any, out: set[str]) -> None:
        if isinstance(value, str):
            if value in known:
                out.add(value)
        elif isinstance(value, dict):
            for v in value.values():
                _collect(v, out)
        elif isinstance(value, list):
            for v in value:
                _collect(v, out)

    for e in entities:
        refs: set[str] = set()
        _collect(e.fields, refs)
        refs.discard(e.def_name)
        e.references = sorted(refs)


def main() -> None:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base="1.3", config_path="../../configs", config_name="base.yaml")
    def _run(cfg: DictConfig) -> None:
        xml_dir = cfg_get(cfg, "paths.xml_defs_dir", "data/rimworld_xml_defs")
        out = cfg_get(cfg, "paths.entities_file", "results/entities.json")
        def_types = cfg_get(cfg, "knowledge.def_types", DEFAULT_DEF_TYPES)
        entities = extract_defs(xml_dir, def_types)
        write_json([e.to_dict() for e in entities], out)
        ensure_dir(Path(out).parent)
        log.info("wrote %d entities -> %s", len(entities), out)

    _run()


if __name__ == "__main__":
    main()
