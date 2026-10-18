"""Task B 的验证工具包（**不属于交付实现**）。

⚠️ 这个 `__init__.py` 的存在只有一个目的：让 notebook 里能写 `from tests.grad_check import ...`。
   没有它的话，`tests` 会变成 namespace package，可能被环境里别的同名目录抢走解析。

⛔ 本包**可以** import torch —— 它是对拍/检验工具，不是 `npnn/` 的交付实现。
   交付实现的纯度由 `test_purity.py` 守卫，与这里无关。
"""
