"""執行パスにLLMクライアントが混入していないことを検査する。

執行の判断は1秒未満で終わる必要がある一方、LLMは5〜15秒かかる。
方針を書くだけでは後から誰かが便利さに負けて入れてしまうので、テストで縛る。
Phase 0 では対象モジュールがまだ存在せず自明に通るが、誘惑が生まれる前に入れておく。
"""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "aitrading"

#: 実行時のホットパス。ここにLLMクライアントを入れてはいけない。
EXECUTION_PACKAGES = ("execution", "risk", "strategy")

LLM_MODULES = {"anthropic", "openai", "google", "cohere", "litellm", "langchain"}


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def test_execution_modules_do_not_import_llm_clients():
    offenders = []
    for package in EXECUTION_PACKAGES:
        for path in (SRC / package).rglob("*.py"):
            hits = _imported_roots(path) & LLM_MODULES
            if hits:
                offenders.append(f"{path.relative_to(SRC)}: {sorted(hits)}")
    assert not offenders, "執行パスにLLMクライアントが混入している:\n" + "\n".join(offenders)
