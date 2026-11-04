"""Build the deployable single-file Gemini pipe from maintainable sources.

OpenWebUI can upload only one pipe file.  The maintainable source lives under
``src/gemini_pipe`` and this script splices the compiled sections into
``gemini_pipe.py`` at the repository root.

The two source lists below are the single source of truth for what goes into
the artifact:

- ``STREAM_SOURCES``: ``[("section.name", "relative/path.py"), ...]`` compiled
  into module-level generated sections, in the order their marker pairs appear
  in the template.
- ``METHOD_GROUP_SOURCES``: ``["relative/path.py", ...]`` spliced into one
  ``class Pipe`` body, in this order.  A duplicate method name in a later file
  would silently override the earlier one, so the build refuses to emit it.
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPE_DIR = REPO_ROOT
PIPE_FILE = PIPE_DIR / "gemini_pipe.py"
MIN_FILE = PIPE_DIR / "gemini_pipe.min.py"
SRC_DIR = PIPE_DIR / "src" / "gemini_pipe"
TEMPLATE_SOURCE = SRC_DIR / "pipe_template.py"

STREAM_SOURCES: list[tuple[str, str]] = [
    ("response.grounding", "response/grounding.py"),
    ("response.state", "response/state.py"),
    ("response.status_events", "response/status_events.py"),
]

METHOD_GROUP_SOURCES: list[str] = [
    "request/rag.py",
    "request/history.py",
    "request/tools.py",
    "request/config.py",
    "response/formatting.py",
    "shared/models.py",
    "shared/tasks.py",
]

METHOD_GROUP_INSERT_MARKER = "    # COMPILED PIPE METHOD GROUPS INSERTION POINT"
BEGIN_METHOD_GROUPS = "    # BEGIN GENERATED SECTION: gemini_pipe.pipe_method_groups"
END_METHOD_GROUPS = "    # END GENERATED SECTION: gemini_pipe.pipe_method_groups"

GENERATED_SECTION_RE = re.compile(
    r"[ \t]*# BEGIN GENERATED SECTION: gemini_pipe\.(?P<section>[\w.]+)\n"
    r".*?[ \t]*# END GENERATED SECTION: gemini_pipe\.(?P=section)\n",
    re.S,
)


class BuildError(RuntimeError):
    """Raised when a source file or the template cannot be compiled."""


def _markers(section: str) -> tuple[str, str]:
    """Begin/end marker pair delimiting one generated section in the template."""
    return (
        f"# BEGIN GENERATED SECTION: gemini_pipe.{section}",
        f"# END GENERATED SECTION: gemini_pipe.{section}",
    )


def _parse(path: str, text: str) -> ast.Module:
    try:
        return ast.parse(text)
    except SyntaxError as e:
        line = e.lineno or 1
        raise BuildError(f"{path}:{line}: syntax error: {e.msg}") from None


def guard_stream_source(path: str, text: str) -> None:
    """A stream source must not import inside a function or class body.

    Stream sections are spliced in at module level; the template provides all
    imports, so a nested import would be dead code or a NameError at runtime.
    """
    tree = _parse(path, text)
    parents = {
        child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
    }
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)) and parents.get(node) is not tree:
            raise BuildError(
                f"{path}:{node.lineno}: import inside a function or class is not "
                "allowed in a stream source (the template provides all imports)"
            )


def guard_method_group(path: str, text: str) -> None:
    """A method-group file keeps only what ``class Pipe`` already has.

    Module level may hold imports, the docstring, ``logger =`` and exactly one
    class.  Anything else (a second class, a helper function, a module-level
    constant) would be spliced into the class body or lost.
    """
    tree = _parse(path, text)
    class_nodes: list[ast.ClassDef] = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Assign) and [
            t.id for t in node.targets if isinstance(t, ast.Name)
        ] == ["logger"]:
            continue
        if isinstance(node, ast.ClassDef):
            class_nodes.append(node)
            continue
        raise BuildError(
            f"{path}:{node.lineno}: module-level {type(node).__name__} is not allowed "
            "in a method group (only imports, a docstring, `logger =` and one class)"
        )
    if len(class_nodes) != 1:
        line = class_nodes[1].lineno if len(class_nodes) > 1 else 1
        raise BuildError(
            f"{path}:{line}: a method group must contain exactly one class, "
            f"found {len(class_nodes)}"
        )


def guard_duplicate_methods(sources: dict[str, str]) -> None:
    """The same method name in two method-group files would silently override."""
    seen: dict[str, str] = {}
    for path, text in sources.items():
        tree = _parse(path, text)
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    owner = seen.get(item.name)
                    if owner is not None and owner != path:
                        raise BuildError(
                            f"{path}:{item.lineno} and {owner}: method "
                            f"{item.name!r} is defined in both method-group files"
                        )
                    seen[item.name] = path


def _compile_stream_source(source: str, begin_marker: str, end_marker: str) -> str:
    """Strip imports, the module docstring and ``logger =``; keep the rest byte for byte."""
    lines = source.splitlines()
    tree = ast.parse(source)
    drop: set[int] = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            drop.update(range(node.lineno - 1, node.end_lineno))
        elif (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            drop.update(range(node.lineno - 1, node.end_lineno))
        elif (
            isinstance(node, ast.Assign)
            and [t.id for t in node.targets if isinstance(t, ast.Name)] == ["logger"]
        ):
            drop.update(range(node.lineno - 1, node.end_lineno))
    keep = [line for idx, line in enumerate(lines) if idx not in drop]
    body = "\n".join(keep).strip() + "\n"
    return f"{begin_marker}\n{body}{end_marker}\n\n"


def _method_group_source_to_class_body(path: str, source: str) -> str:
    tree = _parse(path, source)
    class_node = next((n for n in tree.body if isinstance(n, ast.ClassDef)), None)
    if class_node is None:
        raise BuildError(f"{path}:1: method group source does not contain a class")
    lines = source.splitlines()
    body_lines = lines[class_node.lineno :]
    while body_lines and not body_lines[0].strip():
        body_lines.pop(0)
    return "\n".join(body_lines).rstrip() + "\n\n"


def _compile_method_groups(sources: dict[str, str]) -> str:
    parts = [BEGIN_METHOD_GROUPS + "\n"]
    for rel in METHOD_GROUP_SOURCES:
        parts.append(_method_group_source_to_class_body(rel, sources[rel]))
    parts.append(END_METHOD_GROUPS + "\n")
    return "".join(parts)


def _replace_section(text: str, compiled: str, begin_marker: str, end_marker: str) -> str:
    start = text.index(begin_marker)
    end = text.index(end_marker, start) + len(end_marker)
    while end < len(text) and text[end] in "\r\n":
        end += 1
    return text[:start] + compiled + text[end:]


def _insert_method_groups(text: str, compiled: str) -> str:
    if BEGIN_METHOD_GROUPS in text and END_METHOD_GROUPS in text:
        return _replace_section(text, compiled, BEGIN_METHOD_GROUPS, END_METHOD_GROUPS)
    marker_pos = text.find(METHOD_GROUP_INSERT_MARKER)
    if marker_pos == -1:
        raise BuildError(
            f"{TEMPLATE_SOURCE}: missing the method-group insertion marker "
            f"{METHOD_GROUP_INSERT_MARKER!r} inside `class Pipe`"
        )
    insert_pos = text.find("\n", marker_pos)
    if insert_pos == -1:
        insert_pos = marker_pos + len(METHOD_GROUP_INSERT_MARKER)
    else:
        insert_pos += 1
    return text[:insert_pos] + compiled + text[insert_pos:]


def _strip_generated_sections(text: str) -> str:
    """Empty every generated section but keep its marker pair.

    The surviving markers pin section order to the template: sections are
    filled in place, so a module stays above the code that annotates against it.
    """

    def _empty(match: re.Match[str]) -> str:
        begin, end = _markers(match.group("section"))
        return f"\n{begin}\n{end}\n"

    return GENERATED_SECTION_RE.sub(_empty, text)


def _check_stream_markers(template_text: str) -> None:
    listed = {section for section, _ in STREAM_SOURCES}
    found = {match.group("section") for match in GENERATED_SECTION_RE.finditer(template_text)}
    for section in sorted(listed - found):
        begin, end = _markers(section)
        raise BuildError(
            f"stream section {section!r} has no marker pair in the template "
            f"(missing {begin!r} / {end!r})"
        )
    for section in sorted(found - listed):
        raise BuildError(
            f"template has a marker pair for section {section!r} that no "
            "stream source uses (add it to STREAM_SOURCES or remove the markers)"
        )


def _check_sources() -> dict[str, str]:
    if not TEMPLATE_SOURCE.exists():
        raise BuildError(
            f"Missing pipe template: {TEMPLATE_SOURCE}. Restore {PIPE_FILE} or run "
            "--refresh-template first."
        )
    for _section, rel in STREAM_SOURCES:
        if not (SRC_DIR / rel).exists():
            raise BuildError(f"Missing stream source module: {rel}")
    groups: dict[str, str] = {}
    for rel in METHOD_GROUP_SOURCES:
        path = SRC_DIR / rel
        if not path.exists():
            raise BuildError(f"Missing method group source module: {rel}")
        groups[rel] = path.read_text(encoding="utf-8")
    return groups


def build_text() -> str:
    """Compile the artifact in memory and return its text."""
    groups = _check_sources()
    template_text = TEMPLATE_SOURCE.read_text(encoding="utf-8")
    _check_stream_markers(template_text)

    stream_texts = {
        rel: (SRC_DIR / rel).read_text(encoding="utf-8") for _section, rel in STREAM_SOURCES
    }
    for rel, text in stream_texts.items():
        guard_stream_source(rel, text)
    for rel, text in groups.items():
        guard_method_group(rel, text)
    guard_duplicate_methods(groups)

    text = template_text
    for section, rel in STREAM_SOURCES:
        begin, end = _markers(section)
        text = _replace_section(text, _compile_stream_source(stream_texts[rel], begin, end), begin, end)
    text = _insert_method_groups(text, _compile_method_groups(groups))
    return text


def _compile_check(label: str, text: str) -> None:
    import py_compile
    import tempfile

    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", encoding="utf-8", delete=False, newline="\n"
    ) as handle:
        handle.write(text)
        probe = Path(handle.name)
    try:
        py_compile.compile(str(probe), doraise=True)
    except py_compile.PyCompileError as e:
        raise BuildError(f"{label} does not compile: {e}") from None
    finally:
        probe.unlink(missing_ok=True)


def write_minified(source: str) -> None:
    """Refresh the minified artifact alongside the full one.

    Both files are committed and both are offered as an install source, so the
    minified one must never lag behind: a stale copy would hand users an older
    pipe than the repo advertises. Building it here removes the chance of
    forgetting the separate minify step.
    """
    import py_compile
    import tempfile

    from minify_pipe import minify

    minified = minify(source)
    # Compile before publishing: the minifier rewrites the token stream, and a
    # syntactically broken artifact must not reach the repo.
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", encoding="utf-8", delete=False, newline="\n"
    ) as handle:
        handle.write(minified)
        probe = Path(handle.name)
    try:
        py_compile.compile(str(probe), doraise=True)
    except py_compile.PyCompileError as e:
        raise BuildError(f"{MIN_FILE.name} does not compile: {e}") from None
    finally:
        probe.unlink(missing_ok=True)

    MIN_FILE.write_text(minified, encoding="utf-8", newline="\n")
    saved = (1 - len(minified.encode("utf-8")) / len(source.encode("utf-8"))) * 100
    print(f"built {MIN_FILE.relative_to(REPO_ROOT)} ({saved:.1f}% smaller)")


def build() -> None:
    text = build_text()
    _compile_check(PIPE_FILE.name, text)
    PIPE_FILE.write_text(text, encoding="utf-8", newline="\n")
    print(f"built {PIPE_FILE.relative_to(REPO_ROOT)}")
    write_minified(text)


def check() -> int:
    """Rebuild in memory and verify the committed artifacts are current."""
    text = build_text()
    _compile_check(PIPE_FILE.name, text)
    from minify_pipe import minify

    minified = minify(text)
    _compile_check(MIN_FILE.name, minified)

    stale = []
    for path, expected in ((PIPE_FILE, text), (MIN_FILE, minified)):
        if not path.exists():
            stale.append(path)
        elif path.read_text(encoding="utf-8") != expected:
            stale.append(path)
    if stale:
        for path in stale:
            print(
                f"stale artifact: {path.name} differs from a fresh build; "
                f"run {Path(__file__).name} to regenerate",
                file=sys.stderr,
            )
        return 1
    print("artifacts are up to date")
    return 0


def refresh_template() -> None:
    if not PIPE_FILE.exists():
        raise BuildError(f"Cannot refresh template because {PIPE_FILE} is missing")
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    template_text = _strip_generated_sections(PIPE_FILE.read_text(encoding="utf-8"))
    TEMPLATE_SOURCE.write_text(template_text, encoding="utf-8", newline="\n")
    print(f"refreshed {TEMPLATE_SOURCE.relative_to(REPO_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if the committed artifacts differ from a fresh build")
    parser.add_argument("--refresh-template", action="store_true", help="copy current gemini_pipe.py into src/gemini_pipe/pipe_template.py")
    args = parser.parse_args()
    try:
        if args.check:
            sys.exit(check())
        if args.refresh_template:
            refresh_template()
        build()
    except BuildError as e:
        print(f"build error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
