"""纯度守卫（铁律 2 / 第 0 阶段 0.3）—— 把「实现包里只有 numpy」变成一条能跑的断言。

用法（在 task_b/ 下）：
    python tests/test_purity.py             # 检查 npnn/
    python tests/test_purity.py --selftest  # 反向验证：故意注入违规后，本测试必须失败
    python tests/test_purity.py --package <dir> [--extra-path <dir> ...]   # 指定别的包

两层：

**静态层** —— ast 解析 `npnn/**/*.py`，连同它 import 的**项目本地模块**（如 `config.py`）一起
递归检查，收集所有 import 的顶层模块名。
    允许 = numpy ∪ 标准库 ∪ 项目本地模块（继续递归）
    **默认拒绝**：不在这三类里的一律算违规 —— 这比「固定白名单」更严，也更省心。
为什么必须用 ast：字符串/正则搜索会被注释和 docstring 里的 "we do not use torch" 误伤，
也会漏掉 `importlib.import_module('torch')` 这种写法（下面额外扫了这类调用）。

**动态层** —— 在**子进程**里：
    1. `snapshot = set(sys.modules)`
    2. `import npnn`，再用最小输入跑一遍前向 / 反向 / 一步参数更新
    3. 断言新增的顶层模块里没有禁用项
为什么必须在子进程：`tests/` 自己就 import torch（对拍要用），同一个进程里 torch 早就在
`sys.modules` 里了，检查毫无意义 —— 要测「没有任何脏东西」，就得先有一个干净的房间。

⛔ 本文件是**验证工具**，允许 import torch —— 它不属于交付实现。
"""

import argparse
import ast
import json
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
TASKB_ROOT = TESTS_DIR.parent                      # task_b/
DEFAULT_PACKAGE = TASKB_ROOT / 'npnn'

# 静态层：除标准库与本项目模块外，唯一允许的第三方来源
ALLOWED_SOURCES = {'numpy'}

# 动态层：禁用清单（按顶层模块名比较）
BANNED = {
    'torch', 'torchvision', 'torchaudio', 'tensorflow', 'keras', 'jax', 'jaxlib',
    'autograd', 'theano', 'mxnet', 'chainer', 'cupy', 'numba',
    'scipy', 'pandas', 'matplotlib', 'sklearn', 'cv2', 'PIL', 'imageio', 'tqdm',
}

_STDLIB = set(sys.stdlib_module_names) | set(sys.builtin_module_names)


# ------------------------------------------------------------------ 静态层
def _imports_of(path: Path):
    """返回 (顶层模块名集合, 相对 import 次数, importlib/__import__ 的字符串参数集合)。"""
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    names, relative, dynamic = set(), 0, set()
    for node in ast.walk(tree):                    # walk 会进到函数内部，懒 import 也躲不掉
        if isinstance(node, ast.Import):
            names.update(alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:                         # from .linear import Linear —— 包内相对，允许
                relative += 1
            elif node.module:
                names.add(node.module.split('.')[0])
        elif isinstance(node, ast.Call):
            func = node.func
            if getattr(func, 'attr', None) == 'import_module' or getattr(func, 'id', None) == '__import__':
                dynamic.update(a.value for a in node.args
                               if isinstance(a, ast.Constant) and isinstance(a.value, str))
    return names, relative, dynamic


def _find_local(name: str, roots):
    """项目本地模块？在 roots 里找 <name>.py 或 <name>/__init__.py。"""
    for root in roots:
        for cand in (root / f'{name}.py', root / name / '__init__.py'):
            if cand.is_file():
                return cand
    return None


def static_check(package_dir: Path):
    roots = [package_dir.parent, package_dir]      # npnn/../config.py 与 npnn/xxx.py
    queue, seen, local_mods, violations = sorted(package_dir.rglob('*.py')), [], {}, []
    while queue:
        path = queue.pop(0)
        if path in seen:
            continue
        seen.append(path)
        names, _relative, dynamic = _imports_of(path)
        for name in sorted(names | dynamic):
            if name in ALLOWED_SOURCES or name in _STDLIB:
                continue
            origin = _find_local(name, roots)
            if origin is not None:
                local_mods[name] = origin      # 项目本地模块 → 递归检查它
                if origin not in seen:
                    queue.append(origin)
            else:
                try:
                    shown = path.relative_to(package_dir.parent)
                except ValueError:
                    shown = path
                violations.append((shown, name))
    return seen, local_mods, violations


# ------------------------------------------------------------------ 动态层
_DYNAMIC_SCRIPT = r'''
import json, sys
for p in reversed(sys.argv[1:]):
    sys.path.insert(0, p)
snapshot = set(sys.modules)
import npnn

import numpy as np
rng = np.random.default_rng(0)
drilled = []

# --- 演练：每实现一个新模块，就在这里加一段（最小输入 + 前向 + 反向 + 一步参数更新）---
if hasattr(npnn, 'normalize'):
    npnn.normalize((rng.random((2, 3, 32, 32)) * 255).astype(np.uint8))
    drilled.append('normalize')

if hasattr(npnn, 'Linear'):
    layer = npnn.Linear(8, 4)
    out = layer.forward(rng.standard_normal((2, 8)))
    layer.backward(np.ones_like(out))
    layer.weight -= 0.01 * layer.grads['weight']          # 一步参数更新
    drilled.append('Linear')

if hasattr(npnn, 'GELU'):
    act = npnn.GELU()
    out = act.forward(rng.standard_normal((2, 3)))
    act.backward(np.ones_like(out))
    drilled.append('GELU')

if hasattr(npnn, 'CrossEntropyLoss'):
    ce = npnn.CrossEntropyLoss()
    ce.forward(rng.standard_normal((3, 5)), np.array([0, 2, 4]))
    ce.backward()
    drilled.append('CrossEntropyLoss')

delta = sorted({m.split('.')[0] for m in set(sys.modules) - snapshot})
print(json.dumps({'drilled': drilled, 'delta': delta}))
'''


def dynamic_check(package_dir: Path, extra_paths=()):
    paths = [str(package_dir.parent)] + [str(p) for p in extra_paths]
    proc = subprocess.run([sys.executable, '-I', '-c', _DYNAMIC_SCRIPT, *paths],
                          capture_output=True, text=True, encoding='utf-8')
    if proc.returncode != 0:
        return {'error': proc.stderr.strip()[-500:] or f'退出码 {proc.returncode}'}
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    payload['banned'] = [m for m in payload['delta'] if m in BANNED]
    return payload


# ------------------------------------------------------------------ 报告
def check(package_dir: Path, extra_paths=()):
    scanned, local_mods, violations = static_check(package_dir)
    dynamic = dynamic_check(package_dir, extra_paths)
    ok = not violations and not dynamic.get('banned') and 'error' not in dynamic
    return {'scanned': scanned, 'local': local_mods, 'violations': violations,
            'dynamic': dynamic, 'ok': ok}


def report(result, title):
    # 打印一律 ASCII：Windows 控制台默认代码页会吃掉中文，而这份输出要直接贴进英文报告
    print(f'=== {title} ===')
    local = [p.name for p in result['local'].values()] or ['none']
    print(f'[static]  parsed {len(result["scanned"])} file(s); '
          f'project-local modules followed: {local}')
    for path, name in result['violations']:
        print(f'  [VIOLATION] {path} imports {name!r}')
    if not result['violations']:
        print('  [ok] every import is numpy / stdlib / project-local')

    dyn = result['dynamic']
    if 'error' in dyn:
        print(f'[dynamic] FAILED - subprocess error: {dyn["error"]}')
    else:
        drilled = '+'.join(dyn['drilled']) or 'nothing yet'
        extra = [m for m in dyn['delta'] if m not in _STDLIB]
        print(f'[dynamic] subprocess exercised: {drilled}')
        print(f'          {len(dyn["delta"])} new top-level modules; non-stdlib: {extra}')
        print('  [ok] no banned module imported' if not dyn['banned']
              else f'  [VIOLATION] banned module(s) imported: {dyn["banned"]}')
    print('verdict: ' + ('PASS' if result['ok'] else 'FAIL') + '\n')
    return result['ok']


def selftest():
    """反向验证：造一个只含 `import torch` 的假 npnn，守卫**必须**判它不通过。"""
    print('### negative control: a test that has never failed is not a test ###\n')
    with tempfile.TemporaryDirectory() as tmp:
        fake = Path(tmp) / 'npnn'
        fake.mkdir()
        (fake / '__init__.py').write_text('import torch\n', encoding='utf-8')
        bad = check(fake)
    report(bad, 'fake npnn containing `import torch` (must FAIL)')
    print('[selftest] ' + ('PASS - the guard rejected the injected violation'
                           if not bad['ok'] else 'FAIL - the guard MISSED the violation'))
    return not bad['ok']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument('--extra-path', type=Path, action='append', default=[])
    parser.add_argument('--selftest', action='store_true')
    args = parser.parse_args(argv)

    ok = report(check(args.package, args.extra_path), f'purity check: {args.package}')
    if args.selftest:
        ok = selftest() and ok
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
