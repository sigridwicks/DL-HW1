# 项目约定

中文沟通。每个 cell 只依赖 code #1+#2，不许跨 cell 读变量或输出；公共逻辑一律进 `task_a/src/`。

实验单变量、同预算锚点；1 seed 只筛，3 seed 定论，坍缩结果照留；终局 45k/5k 决策、50k 重训后 test 只测一次。

`experiments.py` 的字段名是分析 notebook 的契约，勿改。

提交前 UTF-8 无 BOM、不留异常输出、不提交数据/权重/凭据。
