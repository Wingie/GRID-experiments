"""AST-based extraction of code entities (classes, functions, methods).

Primary backend is tree-sitter (multi-language). For Python we also ship a stdlib
``ast`` backend that always works offline, which keeps the pipeline + tests runnable
without compiled grammars. Each entity becomes a :class:`CodeEntity` carrying the text
views the embedder needs (signature, docstring, body) plus the parent/child graph that
gives the semantic-ID hierarchy its structure (module -> class -> method).
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from typing import Iterable

from src.utils import LANG_EXTS, RepoFile, get_logger

log = get_logger("extract_entities")


@dataclass
class CodeEntity:
    name: str
    kind: str  # "class" | "function" | "method"
    file_path: str
    body: str
    signature: str
    docstring: str | None = None
    parent: str | None = None  # parent class name for methods
    children: list[str] = field(default_factory=list)  # member function names
    language: str = ""
    start_line: int = 0
    end_line: int = 0

    @property
    def uid(self) -> str:
        """Stable identifier across the pipeline (file-scoped, parent-qualified)."""
        qual = f"{self.parent}." if self.parent else ""
        return f"{self.file_path}::{qual}{self.name}"

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Python backend (stdlib ast) - always available
# --------------------------------------------------------------------------- #
def _python_signature(node: ast.AST, source: str) -> str:
    line = getattr(node, "lineno", 1)
    return source.splitlines()[line - 1].strip() if 0 < line <= len(source.splitlines()) else ""


def _extract_python(rf: RepoFile) -> list[CodeEntity]:
    try:
        tree = ast.parse(rf.text)
    except SyntaxError as exc:
        log.debug("Skipping %s: %s", rf.rel_path, exc)
        return []

    entities: list[CodeEntity] = []

    def visit(node: ast.AST, enclosing: CodeEntity | None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                ent = CodeEntity(
                    name=child.name,
                    kind="class",
                    file_path=rf.rel_path,
                    body=ast.get_source_segment(rf.text, child) or "",
                    signature=_python_signature(child, rf.text),
                    docstring=ast.get_docstring(child),
                    parent=enclosing.name if enclosing else None,
                    language=rf.language,
                    start_line=child.lineno,
                    end_line=getattr(child, "end_lineno", child.lineno),
                )
                entities.append(ent)
                visit(child, ent)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                is_method = enclosing is not None and enclosing.kind == "class"
                ent = CodeEntity(
                    name=child.name,
                    kind="method" if is_method else "function",
                    file_path=rf.rel_path,
                    body=ast.get_source_segment(rf.text, child) or "",
                    signature=_python_signature(child, rf.text),
                    docstring=ast.get_docstring(child),
                    parent=enclosing.name if is_method else None,
                    language=rf.language,
                    start_line=child.lineno,
                    end_line=getattr(child, "end_lineno", child.lineno),
                )
                entities.append(ent)
                if is_method:
                    enclosing.children.append(child.name)
                # recurse but keep the class as enclosing only for direct methods
                visit(child, enclosing if is_method else None)
            else:
                visit(child, enclosing)

    visit(tree, None)
    return entities


# --------------------------------------------------------------------------- #
# tree-sitter backend (multi-language) - best effort
# --------------------------------------------------------------------------- #
# node types that behave as class-like vs function-like, per language.
_CLASS_TYPES = {
    "python": {"class_definition"},
    "javascript": {"class_declaration"},
    "typescript": {"class_declaration", "interface_declaration"},
    "tsx": {"class_declaration", "interface_declaration"},
    "rust": {"struct_item", "trait_item", "impl_item"},
    "go": {"type_declaration"},
    "java": {"class_declaration", "interface_declaration"},
    "ruby": {"class", "module"},
}
_FUNC_TYPES = {
    "python": {"function_definition"},
    "javascript": {"function_declaration", "method_definition"},
    "typescript": {"function_declaration", "method_definition"},
    "tsx": {"function_declaration", "method_definition"},
    "rust": {"function_item"},
    "go": {"function_declaration", "method_declaration"},
    "java": {"method_declaration", "constructor_declaration"},
    "ruby": {"method", "singleton_method"},
}

_PARSER_CACHE: dict[str, object] = {}


def _get_parser(language: str, module_name: str):
    """Build (and cache) a tree-sitter parser for a language, or None if unavailable."""
    if language in _PARSER_CACHE:
        return _PARSER_CACHE[language]
    try:
        import importlib

        from tree_sitter import Language, Parser

        grammar = importlib.import_module(module_name)
        # typescript/tsx expose language_typescript()/language_tsx()
        if language == "typescript" and hasattr(grammar, "language_typescript"):
            lang_ptr = grammar.language_typescript()
        elif language == "tsx" and hasattr(grammar, "language_tsx"):
            lang_ptr = grammar.language_tsx()
        else:
            lang_ptr = grammar.language()
        ts_lang = Language(lang_ptr)
        try:
            parser = Parser(ts_lang)  # tree-sitter >= 0.22
        except TypeError:
            parser = Parser()  # tree-sitter ~0.21
            parser.set_language(ts_lang)
        _PARSER_CACHE[language] = parser
        return parser
    except Exception as exc:
        log.debug("tree-sitter unavailable for %s (%s)", language, exc)
        _PARSER_CACHE[language] = None
        return None


def _node_name(node) -> str:
    field_name = node.child_by_field_name("name")
    if field_name is not None:
        return field_name.text.decode("utf-8", "replace")
    # fall back to first identifier child
    for child in node.children:
        if "identifier" in child.type:
            return child.text.decode("utf-8", "replace")
    return "<anonymous>"


def _extract_treesitter(rf: RepoFile) -> list[CodeEntity] | None:
    module_name = next(
        (m for _ext, (lang, m) in LANG_EXTS.items() if lang == rf.language and m),
        None,
    )
    if not module_name:
        return None
    parser = _get_parser(rf.language, module_name)
    if parser is None:
        return None

    class_types = _CLASS_TYPES.get(rf.language, set())
    func_types = _FUNC_TYPES.get(rf.language, set())
    src_bytes = rf.text.encode("utf-8")
    tree = parser.parse(src_bytes)
    entities: list[CodeEntity] = []

    def text_of(node) -> str:
        return src_bytes[node.start_byte : node.end_byte].decode("utf-8", "replace")

    def visit(node, enclosing: CodeEntity | None) -> None:
        for child in node.children:
            if child.type in class_types:
                body = text_of(child)
                ent = CodeEntity(
                    name=_node_name(child),
                    kind="class",
                    file_path=rf.rel_path,
                    body=body,
                    signature=body.splitlines()[0].strip() if body else "",
                    parent=enclosing.name if enclosing else None,
                    language=rf.language,
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                )
                entities.append(ent)
                visit(child, ent)
            elif child.type in func_types:
                is_method = enclosing is not None and enclosing.kind == "class"
                body = text_of(child)
                ent = CodeEntity(
                    name=_node_name(child),
                    kind="method" if is_method else "function",
                    file_path=rf.rel_path,
                    body=body,
                    signature=body.splitlines()[0].strip() if body else "",
                    parent=enclosing.name if is_method else None,
                    language=rf.language,
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                )
                entities.append(ent)
                if is_method:
                    enclosing.children.append(ent.name)
                visit(child, enclosing if is_method else None)
            else:
                visit(child, enclosing)

    visit(tree.root_node, None)
    return entities


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def extract_from_file(rf: RepoFile) -> list[CodeEntity]:
    """Extract entities from one file, choosing the best available backend."""
    if rf.language == "python":
        ents = _extract_treesitter(rf)
        # stdlib ast is reliable and offline; prefer it when tree-sitter is missing
        return ents if ents else _extract_python(rf)
    ents = _extract_treesitter(rf)
    if ents is None:
        log.debug("No backend for language %s (%s); skipping.", rf.language, rf.rel_path)
        return []
    return ents


def extract_entities(repo_files: Iterable[RepoFile]) -> list[CodeEntity]:
    """Extract all classes/functions/methods from the given repo files."""
    entities: list[CodeEntity] = []
    for rf in repo_files:
        entities.extend(extract_from_file(rf))
    log.info("Extracted %d code entities", len(entities))
    return entities


def main() -> None:
    import argparse
    from pathlib import Path

    from src.utils import cfg_get, iter_repo_files, load_config, write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    files = iter_repo_files(cfg_get(cfg, "paths.target_repo"), cfg_get(cfg, "languages", ["py"]))
    entities = extract_entities(files)
    out = Path(cfg_get(cfg, "paths.entities_file", "results/entities.json"))
    write_json([e.to_dict() for e in entities], out)
    log.info("Wrote %d entities -> %s", len(entities), out)


if __name__ == "__main__":
    main()
