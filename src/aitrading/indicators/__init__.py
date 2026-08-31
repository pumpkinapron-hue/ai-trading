"""指標パッケージ。

配下のモジュールを機械的に全部読み込んでから `INDICATORS` を公開する。

`from .core import *` だけにすると、**新しいモジュールを足して `@indicator` を
付けても、ここに import を書き足すまでレジストリに載らない**。載らなければ
トランケーション不変性検査の対象にもならず、先読みしている指標が沈黙のまま
通る（実測: `shift(-5)` を使う指標を別モジュールに置いても 61 passed だった）。

「テストを書き忘れても先読み検査だけは必ず走る」のがこの層の設計意図なので、
「import を書き忘れたら検査対象から外れる」という抜け道を残さない。
ファイルを置けば検査対象に入る。
"""

from __future__ import annotations

import importlib
import pkgutil

from aitrading.indicators.registry import INDICATORS, indicator

__all__ = ["INDICATORS", "indicator"]

for _module in pkgutil.iter_modules(__path__):
    if _module.name != "registry":
        importlib.import_module(f"{__name__}.{_module.name}")
