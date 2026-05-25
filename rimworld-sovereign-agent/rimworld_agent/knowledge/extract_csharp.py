"""Parse decompiled RimWorld C# (``Assembly-CSharp.dll`` -> ILSpy output) into
:class:`CSharpEntity` records using ``tree-sitter-c-sharp``.

We focus on the engine (``Verse.*``) and game-logic (``RimWorld.*``) namespaces and, for
each type, capture its kind, base type, public method names, and field names (spec §2b).
Decompiled field names are preserved even when local variables are obfuscated (gotcha #8).

This module requires the ``csharp`` extra (``tree-sitter`` + ``tree-sitter-c-sharp``) and a
directory of ``.cs`` files; it is not exercised in the offline test suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rimworld_agent.utils import cfg_get, get_logger, write_json

log = get_logger("extract_csharp")

_TYPE_NODES = {
    "class_declaration": "class",
    "interface_declaration": "interface",
    "enum_declaration": "enum",
    "struct_declaration": "struct",
    "record_declaration": "record",
}


@dataclass
class CSharpEntity:
    name: str
    namespace: str
    kind: str  # class | interface | enum | struct | record
    parent_class: str | None = None
    methods: list[str] = field(default_factory=list)
    fields: list[str] = field(default_factory=list)
    file_path: str = ""
    linked_defs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    def text_blob(self) -> str:
        base = f" : {self.parent_class}" if self.parent_class else ""
        sig = f"{self.kind} {self.namespace}.{self.name}{base}"
        return "\n".join(
            [sig, "methods: " + ", ".join(self.methods), "fields: " + ", ".join(self.fields)]
        )


def _make_parser():
    """Build a tree-sitter C# parser, tolerating both old and new tree-sitter APIs."""
    try:
        import tree_sitter_c_sharp as tscs
        from tree_sitter import Language, Parser
    except ImportError as exc:  # pragma: no cover - requires the `csharp` extra
        raise ImportError(
            "extract_csharp needs `pip install tree-sitter tree-sitter-c-sharp` "
            "(the `csharp` extra)."
        ) from exc

    language = Language(tscs.language())
    try:
        return Parser(language)  # tree-sitter >= 0.22
    except TypeError:  # pragma: no cover - older API
        parser = Parser()
        parser.set_language(language)
        return parser


def _text(node, src: bytes) -> str:
    return src[node.start_byte : node.end_byte].decode("utf8", errors="replace")


def _child_name(node, src: bytes) -> str:
    n = node.child_by_field_name("name")
    return _text(n, src) if n is not None else ""


def _find_namespace(node, src: bytes) -> str:
    cur = node.parent
    while cur is not None:
        if cur.type in ("namespace_declaration", "file_scoped_namespace_declaration"):
            return _child_name(cur, src)
        cur = cur.parent
    return ""


def _base_type(node, src: bytes) -> str | None:
    bases = node.child_by_field_name("bases")
    if bases is None:
        for c in node.children:
            if c.type == "base_list":
                bases = c
                break
    if bases is None:
        return None
    for c in bases.children:
        if c.type in ("identifier", "qualified_name", "generic_name"):
            return _text(c, src)
    return None


def _collect_members(node, src: bytes) -> tuple[list[str], list[str]]:
    methods: list[str] = []
    fields: list[str] = []
    body = node.child_by_field_name("body")
    if body is None:
        return methods, fields
    for c in body.children:
        if c.type == "method_declaration":
            name = _child_name(c, src)
            if name:
                methods.append(name)
        elif c.type == "field_declaration":
            for decl in c.children:
                if decl.type == "variable_declaration":
                    for v in decl.children:
                        if v.type == "variable_declarator":
                            fields.append(_text(v.child_by_field_name("name") or v, src))
    return methods, fields


def extract_csharp(source_dir: str | Path, namespaces: list[str] | None = None) -> list[CSharpEntity]:
    """Parse all ``.cs`` files under ``source_dir`` into :class:`CSharpEntity` records."""
    parser = _make_parser()
    source_dir = Path(source_dir)
    keep_ns = tuple(namespaces) if namespaces else ("RimWorld", "Verse")
    entities: list[CSharpEntity] = []

    for path in sorted(source_dir.rglob("*.cs")):
        src = path.read_bytes()
        tree = parser.parse(src)
        stack = [tree.root_node]
        while stack:
            node = stack.pop()
            kind = _TYPE_NODES.get(node.type)
            if kind:
                ns = _find_namespace(node, src)
                if not keep_ns or ns.split(".")[0] in keep_ns:
                    methods, fields = _collect_members(node, src)
                    entities.append(
                        CSharpEntity(
                            name=_child_name(node, src),
                            namespace=ns,
                            kind=kind,
                            parent_class=_base_type(node, src),
                            methods=methods,
                            fields=fields,
                            file_path=str(path),
                        )
                    )
            stack.extend(node.children)

    log.info("extracted %d C# entities from %s", len(entities), source_dir)
    return entities


def link_defs_to_classes(csharp: list[CSharpEntity], def_entities: list) -> None:
    """Populate ``linked_defs``: which Def names reference each C# class via a ``Class``
    or ``thingClass``/``compClass`` field. Mutates ``csharp`` in place.
    """
    by_class: dict[str, CSharpEntity] = {e.name: e for e in csharp}

    def _scan(value, def_name: str) -> None:
        if isinstance(value, str):
            short = value.rsplit(".", 1)[-1]
            ent = by_class.get(short)
            if ent is not None and def_name not in ent.linked_defs:
                ent.linked_defs.append(def_name)
        elif isinstance(value, dict):
            for v in value.values():
                _scan(v, def_name)
        elif isinstance(value, list):
            for v in value:
                _scan(v, def_name)

    for d in def_entities:
        fields = getattr(d, "fields", {})
        _scan(fields, getattr(d, "def_name", ""))


def main() -> None:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base="1.3", config_path="../../configs", config_name="base.yaml")
    def _run(cfg: DictConfig) -> None:
        src_dir = cfg_get(cfg, "paths.csharp_source_dir", "data/rimworld_source")
        out = cfg_get(cfg, "paths.csharp_entities_file", "results/csharp_entities.json")
        namespaces = cfg_get(cfg, "knowledge.csharp_namespaces", ["RimWorld", "Verse"])
        entities = extract_csharp(src_dir, namespaces)
        write_json([e.to_dict() for e in entities], out)
        log.info("wrote %d C# entities -> %s", len(entities), out)

    _run()


if __name__ == "__main__":
    main()
