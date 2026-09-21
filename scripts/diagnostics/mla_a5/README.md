# A5 MLA prefix 精度定位代码

适用问题：PR #40131 在 A5 可以运行，但 DeepSeek-V2-Lite-W8A8 的 GSM8K 精度约 25.8%；`ASCEND_USE_FIA=1`、NZ 关闭；关闭 radix 后约 32.8%。目标是定位精度问题，不是修复启动错误。

只需要把 `mla_a5_diag.py` 复制到 A5，例如 `/tmp/mla_a5_diag.py`。本诊断包由本地准备，没有连接 A5、修改远端文件或推送 PR。

本目录包含：

- `mla_a5_diag.py`：完整运行时探针、安装/恢复、CPU 分析和原用例启动包装器。
- `probe-hooks-3ba07dd.patch`：基于 PR 提交 `3ba07ddc43` 的两个源码文件接入位置，供查看探针改动；安装由 Python 脚本完成，包含诊断模块的放置和备份。
- `README.md`：安装、运行、分析、对照实验、结果判读和恢复方法。
- `local-validation.json`：对两个目标提交进行本地语法、安装和恢复校验的记录。

`tranfiles` 用于传递诊断文件。运行用例时保留当前 PR 工作区的代码版本。在该工作区可以这样取出脚本和说明，无需切换分支：

```bash
git fetch https://github.com/McZyWu/sgl-sglang.git tranfiles
git show FETCH_HEAD:scripts/diagnostics/mla_a5/mla_a5_diag.py > /tmp/mla_a5_diag.py
git show FETCH_HEAD:scripts/diagnostics/mla_a5/README.md > /tmp/mla-a5-diagnostic-README.md
```

脚本的 `install` / `restore` 使用 Python 标准库；`analyze` 使用 CPU PyTorch；只有显式执行 `run-case` 才会运行现有测试并启动它的服务。可以使用与原测试相同的 Python / 容器 / CANN 环境。

## 1. 安装临时探针

在 A5 的原运行容器中执行，仓库路径按实际修改：

```bash
REPO=/home/wzy/sgl-sglang
DIAG=/tmp/mla_a5_diag.py
cd "$REPO"
git rev-parse HEAD
python "$DIAG" install --repo "$REPO"
git diff --stat
```

已在本地校验 `1344e1e2a5` 和 `3ba07ddc43` 两个提交上的安装、Python 语法和逐字节恢复。其他版本只有所有代码定位点都唯一匹配才会安装；匹配失败时不写入文件。

安装修改两个文件：

- `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`
- `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`

增加模块 `python/sglang/srt/hardware_backend/npu/attention/_mla_a5_diag.py`；备份保存在仓库根目录 `.mla-a5-diag-backup`。备份保留安装前已有的本地修改。测试文件和权重配置不会被修改。

## 2. 第一组：采集原始路径

保留原来的卡选择、CANN 环境和权重，保持 radix 开启。下面包装器运行原 `test_npu_hicache_mla`，完整 1319 题，保留其 5-shot、parallel=128、512 最大输出 token 和 HiCache 参数，运行时固定服务 seed=42。它会清除 `GITHUB_EVENT_NAME`，避免误走 PR smoke 分支；会保留原 `PYTHONPATH` 并前置当前仓库的 `python` 路径。

```bash
export ASCEND_USE_FIA=1
export SGLANG_USE_FIA_NZ=0
export SGLANG_MLA_DIAG=observe
export SGLANG_MLA_DIAG_LAYERS=0,13,26
export SGLANG_MLA_DIAG_STEPS=2
export SGLANG_MLA_DIAG_DIR=/tmp/mla-a5-observe-$(date +%Y%m%d-%H%M%S)
mkdir -p "$SGLANG_MLA_DIAG_DIR"

python -u "$DIAG" run-case --repo "$REPO" --seed 42 > "$SGLANG_MLA_DIAG_DIR/test.log" 2>&1
```

若原测试文件中的权重路径已改好，上述命令直接复用它。也可给 `run-case` 加 `--model /实际的/DeepSeek-V2-Lite-W8A8`，只在本次运行中覆盖路径，不修改测试文件。

`observe` 保持原 FIA 和 `npu_attention_update` 的结果；探针只采集张量。CPU 拷贝会引入同步，采集批次的性能不适合做基准。如果仅打开采集就恢复精度，要优先关注 stream / 同步 / 生命周期问题，不能把这当作算子已修好。

如果测试仍因低于 34% 返回失败，这是当前问题的预期表现。只要服务完成过 prefix 计算，采集文件仍然可分析。

## 3. 离线分析

```bash
python "$DIAG" analyze "$SGLANG_MLA_DIAG_DIR" > "$SGLANG_MLA_DIAG_DIR/analysis.stdout.jsonl"
```

生成 `analysis.json`。此步骤只用 CPU，不启动模型、不执行 NPU 算子。参考计算采用 FP32，包含 NoPE 与 RoPE 两部分点积，并使用真实 `layer.scaling`。完整 attention 的 causal 边界是 `prefix_len + query_position`，不是普通左上角三角 mask。

输出字段及解释：

| 字段 | 检查内容 |
|---|---|
| cache 记录的 `kv` / `rope` | 同一次 `npu_kv_rmsnorm_rope_cache` 直接返回的 KV，与其写入 cache 后读回的 KV；已按 cache dtype 对齐 |
| `slot_mapping` | prefix 分页展开的物理 token 地址，是否与该请求的 token 映射一致 |
| `gather_kv` / `gather_rope` | prefix gather 的值，与从请求 token 映射直接读取的值是否一致 |
| `current_output_vs_fp32` / `current_lse_vs_fp32` | 当前 token causal FIA 的 output / LSE 与 CPU FP32 参考 |
| `prefix_output_vs_fp32` / `prefix_lse_vs_fp32` | prefix 无 mask FIA 的 output / LSE 与 CPU FP32 参考 |
| `*_output_vs_rounded_ref` | 参考输出先舍入到 FIA 输出 dtype 后再比较，帮助区分 BF16 表示精度与更大的偏差 |
| `prefix_weight_vs_fp32` | LSE 导出的 prefix 合并权重与参考值；两段 LSE 相同常量偏移会抵消，不能只因 LSE 绝对值有差异就定因 |
| `merge_vs_explicit_same_fia_parts` | 使用完全相同的 FIA output/LSE，NPU 合并与 CPU 显式 FP32 合并之间的差异 |
| `explicit_same_fia_parts_vs_fp32_full` | 绕过 NPU 合并算子后，FIA 的部分结果是否仍导致整体偏差 |
| `merged_vs_fp32_full` | 最终 NPU 合并输出与完整 attention FP32 参考 |
| `reference_split_vs_full` | CPU 分段参考合并与 CPU 完整 attention 的一致性，应该只有小的浮点误差 |

每项包含 `max_abs`、`mean_abs`、`rms_rel`、`mean_signed`、`nonfinite_mismatch` 等。没有设置统一的“算子失败阈值”：BF16 attention 输出与 FP32 合并的预期误差不同；首先看偏差在哪一步显著增加，并和同配置 A3 数据比较。

判断顺序：

1. cache 写回、地址或 gather 不一致：先查数据和 metadata，避免拿错误输入责怪 attention 算子。
2. FIA output/LSE 与参考一致，而 `merge_vs_explicit_same_fia_parts` 明显异常：指向 `npu_attention_update` 或其参数约定。
3. 合并与显式公式一致，但某段 FIA output/LSE 或 prefix 权重明显异常：向前查对应 FIA 调用及 mask / 长度 / RoPE 参数。
4. 所有已采样点正常：不能排除未采样的层、head、请求或后续 decode。可以扩大采集范围；此诊断并不覆盖整个模型的全部算子。

## 4. 第二组：只替换 prefix 最后的合并输出

仍使用同一权重、同一 seed、同一原用例配置：

```bash
export SGLANG_MLA_DIAG=replace_merge
export SGLANG_MLA_DIAG_DIR=/tmp/mla-a5-replace-merge-$(date +%Y%m%d-%H%M%S)
mkdir -p "$SGLANG_MLA_DIAG_DIR"

python -u "$DIAG" run-case --repo "$REPO" --seed 42 > "$SGLANG_MLA_DIAG_DIR/test.log" 2>&1
python "$DIAG" analyze "$SGLANG_MLA_DIAG_DIR" > "$SGLANG_MLA_DIAG_DIR/analysis.stdout.jsonl"
```

如果第一组用 `--model` 覆盖权重，第二组也传同一个路径。

该模式仍执行原 `npu_attention_update` 以便比较，但把送给后续计算的合并结果换成 NPU 上的 PyTorch FP32 公式：

```python
lse = torch.logaddexp(lse_current.float(), lse_prefix.float())
output = (
    output_current.float() * (lse_current.float() - lse).exp().unsqueeze(-1)
    + output_prefix.float() * (lse_prefix.float() - lse).exp().unsqueeze(-1)
)
```

原有的输出 dtype 转换、FIA 调用、cache、投影和 decode 路径保持原逻辑。**合并替换作用于所有层、所有 TP rank、所有 prefix 批次，不受采集层数或次数限制。** `replacement_vs_cpu_explicit` 可核对这个 NPU PyTorch 公式与 CPU 公式是否一致。

若仅换合并输出就恢复精度，并且局部数据也显示原合并结果异常，才有较强依据定位到合并链路。若精度仍低，应继续看 FIA、KV 数据和 DeepSeek 公共路径，不能直接把 FIA 判为错误。

## 5. 扩大采集范围

默认每个 TP rank 对 layer 0/13/26 各采两次 prefix 批次；每批最多两条请求、两个 head、八个 query 位置。参考计算保留这些 query 所需的全部 key。混合空/非空 prefix 时优先各采一条。超出 16384 个 prefix+current key 的请求跳过。

```bash
export SGLANG_MLA_DIAG_LAYERS=all
export SGLANG_MLA_DIAG_STEPS=1
export SGLANG_MLA_DIAG_HEADS=4
# 可选：SGLANG_MLA_DIAG_REQUESTS、SGLANG_MLA_DIAG_ROWS、SGLANG_MLA_DIAG_MAX_KEYS
```

扩大范围会增加同步、运行时间和 dump 大小。更改后使用新的输出目录重新运行。

## 6. 返回哪些结果

优先提供两组的 `analysis.json`、`rank*-pid*.jsonl` 和测试最终 accuracy。`test.log` 包含实际参数、采集路径及错误信息。`.pt` 保存抽样真实输入输出，在需要进一步重放时再使用；不必一开始发送全部文件。

若没有 `.pt`，检查 `[MLA_DIAG] config`、`prefix_enter`、`diagnostic_error`：可能实际运行的不是被修改源码、没有命中目标 prefix 分支，或采集参数不覆盖当前层。`run-case` 已强制优先加载指定仓库的 Python 源码。分析失败会返回非零状态，错误原因写入 `analysis.json`，不要将缺数据理解为通过。

## 7. 恢复

测试服务退出后：

```bash
python "$DIAG" restore --repo "$REPO"
unset SGLANG_MLA_DIAG SGLANG_MLA_DIAG_LAYERS SGLANG_MLA_DIAG_STEPS SGLANG_MLA_DIAG_DIR
```

恢复操作只还原探针安装涉及的文件；如果安装后又手工修改了这些文件，会拒绝覆盖，保留备份让你自行处理。外部 `/tmp` 采集文件不会删除。`ASCEND_USE_FIA` / NZ 环境变量按你的原运行配置管理。

本地已完成安装/还原与语法检查；未在 A5 或其他 NPU 上执行本诊断代码。当前本地 Python 没有 PyTorch，因此 CPU 数值分析部分也未在本机执行，需在你的原测试环境运行验证。
