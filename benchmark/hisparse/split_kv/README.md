# Ascend split-KV：两机 PD 自动启动与验证

普通KV BS6已用固定`437a8431`在209/217完成全部流程：D exit0、实际并发6，五轮正式中位数**198.473719 token/s**，全部6/6成功、每条64000/1024/cached63872；原生与forced逐token一致、共16次真实回退，性能cache专项6/6正确。全部轮值与边界见[验证记录](VALIDATION_zh.md)。对照的HiSparse BS16为310.265693；两组BS不同，旧203.723894另为单机16die、选定三轮统计，不能将差值全部归因于HiSparse。

已准备容器可用[BS6预设脚本](run_native_bs6_209_217.sh)直接复现，无需Codex或host signal-trace包装器。在209的P容器源码根目录执行并保持运行：

```bash
bash benchmark/hisparse/split_kv/run_native_bs6_209_217.sh prefill
```

在217的D容器源码根目录执行，自动完成参考、强制回退、缓存语义检查、2轮预热和5轮正式：

```bash
bash benchmark/hisparse/split_kv/run_native_bs6_209_217.sh decode
```

此预设对应本次核实的IP、16个可见die、模型及`/workspace/splitkv-pd-live`前缀，保持native/BS6/graph8/max-total163840/mem0.92。可用`SPLITKV_WORKSPACE`、`MODEL_PATH`及已有CANN/kernel/MF环境路径覆盖目录；其它机器、IP或设备范围请使用下方通用`run.sh`参数。每次创建新run，不覆盖已有结果，也不清理占用端口的其它任务。D结束后P仍常驻，可在其前台终端按Ctrl-C停止。

本目录从两个已准备好的容器分别运行一条命令：P 容器常驻 Prefill；D 容器自动依次运行普通参考、强制回退功能验证、正式压测。脚本不依赖 Codex、远程登录或外部 PASS 文件。2026-11-04 已在 209/217 的 A3 上完成 kernel 构建、真实 MF TP16 小探针、三个 CopySFA 用例和完整 BS16 controller：固定自然语言三组正确且恢复后逐 token 一致，共17次实际回退；五轮正式输出吞吐中位数310.27 tokens/s，全部轮次保留。这个数值来自两机各TP16，旧203.72来自单机TP16，资源与统计规则有差异。早期router注册失败、随机原生输入分叉及MF旧连接native段错误记录保留，范围和未解决项见验证记录；新增性能cache专项与扩BS仍待后续实测。

本次修改及检查范围见 [验证记录](VALIDATION_zh.md)，历史压测资产的来源和还原方式见 [冻结客户端说明](frozen_client/README.md)。

运行前确认两台机器独立、卡空闲且属于本任务；不要占用别人的服务。确认模型目录实际为同一 GLM-5.2 W4A8 checkpoint、量化和 tokenizer。两侧安装匹配的 Python、torch/torch_npu、CANN、memfabric_hybrid、SGLang 依赖和 `sglang_router`。Decode 必须安装配套 split-K/V `sgl-kernel-npu` wheel，并设置 `KERNEL_PYTHON_PATH` 指向 wheel 的独立 target 目录；参见同目录 `setup_kernel.sh`。它会在加载模型前检查 kernel schema 的 K/V 双 tensor 和双指针 ABI，失败直接退出。

两侧容器分别准备新的独立目录，不覆盖已有仓库：

```bash
TASK_WORKSPACE="$PWD/splitkv-workspace"
test ! -e "$TASK_WORKSPACE" || { echo "Already exists: $TASK_WORKSPACE" >&2; exit 1; }
mkdir -p "$TASK_WORKSPACE"
git clone --branch feat/hisparse-mf-split-kv-adapt https://github.com/McZyWu/sgl-sglang.git "$TASK_WORKSPACE/sglang"
git clone --branch feat/hisparse-mf-split-kv-adapt https://github.com/McZyWu/sgl-kernel-npu.git "$TASK_WORKSPACE/sgl-kernel-npu"
git -C "$TASK_WORKSPACE/sgl-kernel-npu" submodule update --init --recursive
cd "$TASK_WORKSPACE/sglang"
bash benchmark/hisparse/split_kv/setup_kernel.sh "$TASK_WORKSPACE/sgl-kernel-npu" "$TASK_WORKSPACE/kernel-target"
export KERNEL_PYTHON_PATH="$TASK_WORKSPACE/kernel-target"
```

`setup_kernel.sh` 使用独立构建目录和 wheel target，默认 A3 的 `KERNEL_SOC=910`；A5 可按实际设备设置 `KERNEL_SOC=950`。可用 `CANN_ENV` 指定实际 Ascend 环境脚本；先核对工具链、容器设备和精确 submodule，遇到依赖问题时保留构建日志。P/D 两侧都必须使用相同 checkpoint 与配套 kernel，不从目录名推断权重相同。

`KERNEL_PYTHON_PATH` 就是上面安装命令的第二个目录 `$TASK_WORKSPACE/kernel-target`，不是另行下载的目录。若安装输出明确写着 `/home/wzy/splitkv-workspace/kernel-target`，应设置这个实际路径；文中的 `/path/to/isolated-kernel-target` 只是待替换示例。两机分别使用各自容器内已安装的路径。

B103 容器中的 `memfabric_hybrid==1.3.0` 可能带有零字节的 `libmf_hybm_accoffload.so` 占位库，触发 `file too short` 和共享池初始化失败。可在每侧容器中用官方随包构建器生成私有 A3 扩展，保留原 site-packages：

```bash
bash benchmark/hisparse/split_kv/setup_memfabric.sh "$TASK_WORKSPACE/mf-target" --soc A3
export MF_PYTHON_PATH="$TASK_WORKSPACE/mf-target"
```

该命令要求新的空目录、实际 A3 设备与 MF 1.3.0 的 wheel 构建标记，保存源码/产物 SHA、工具链、日志及退出码，并核对原安装库未变化。`run.sh` 会在 source CANN 后追加该私有 Python 路径和 MF/torch/torch_npu 动态库路径。217 已完整执行此封装，外层和 `setup.exit` 均为 0，私有 ELF/CDLL 与原库未变检查通过。该次新产物未再次运行 TP16；当前生产仍使用此前通过 TP16 的私有库，不声称两次构建逐字节相同。已有正常匹配的 MF 安装无需重复构建；A5 不使用此 A3 专用辅助脚本。

在已有完整 SGLang 测试依赖的 Linux 环境，可运行以下 CPU 检查；此前的隔离环境已通过其中 2 项新回退测试与 6 项既有 MF CPU 测试（8 项），不代表设备通过。客户端自检只使用模拟服务和 CPU：

```bash
env -u CUDA_VISIBLE_DEVICES TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}" \
  python -m pytest test/registered/unit/hardware_backend/npu/test_sparse_kv_retraction.py \
  test/registered/unit/hardware_backend/npu/test_memfabric_sparse_kv.py
python benchmark/hisparse/split_kv/test_client.py
python benchmark/hisparse/split_kv/test_frozen_bench.py
python benchmark/hisparse/split_kv/test_no_thp_launch.py
python benchmark/hisparse/split_kv/test_spawn_startup.py
```

请在 source CANN 后、framework 仓根执行。测试进程中 `CUDA_VISIBLE_DEVICES` 应 unset，不要设为空字符串；`env` 只改变该测试进程，不会污染后续 NPU 启动环境。

若旧版在启动时出现 `current process has finished its bootstrapping phase`，先更新 framework 分支。`no_thp_launch.py` 的 THP 设置和 SGLang 启动现在只在 `__name__ == "__main__"` 时执行；spawn 子进程以 `__mp_main__` 重放脚本时不会重复启动服务。`test_spawn_startup.py` 使用真实 spawn 和假的服务/系统调用验证此入口，不加载模型或初始化 NPU。这个修复只更新 Python 文件，已安装的配套 kernel target 可继续使用。

两容器均应使用同一 framework 提交和补丁。`P_IP`/`D_IP` 必须分别是各自容器内可绑定或访问、并且两机之间可达的地址；确认相应地址及防火墙允许 HTTP、bootstrap、MF store 和传输所需端口。`--devices` 是**容器内可见编号**而非未经核实的宿主卡号；`--interface` 是每个节点用于本机 TP 的 HCCL/Gloo 接口（例如 `lo`），MF TransferEngine 的节点地址通过 `SGLANG_HOST_IP` 明确设置为各自 IP。P 默认占用端口 41000、8998、24669；D 占用 30001、8000。启动前脚本检查本机端口是否空闲，已占用时退出，不清理已有进程。

两个 host 参数必须是完整 IPv4 地址，例如 `192.168.25.209` 与 `192.168.25.217`，不能填 `809` 这样的简称或端口号。脚本在导入 NPU runtime 前检查地址格式，并尝试在当前角色的地址上绑定临时 TCP socket；只有本机地址会做绑定检查，对端地址只检查格式。请在各自容器中用 `ip -br -4 addr` 核实地址，主机名和 IPv6 暂不支持；格式正确仍不代表两容器网络已经互通。

`run.sh` 默认 source `/usr/local/Ascend/ascend-toolkit/set_env.sh`。若现场不同，先在容器执行 `export CANN_ENV=/实际路径/set_env.sh`；若还需 ATB 环境，设置 `ATB_ENV`。每侧使用各自的 `--model` 本地路径。下面的 IP 和设备列表是示例，替换为**现场确认的值**。在 P 容器执行并保持进程运行：

```bash
cd /path/to/your/sglang
bash benchmark/hisparse/split_kv/run.sh prefill \
  --model /path/to/GLM-5.2-w4a8 --prefill-host P_IP --decode-host D_IP \
  --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --interface lo
```

在 D 容器执行，先确认 `KERNEL_PYTHON_PATH` 是已安装的 split wheel target：

```bash
cd /path/to/your/sglang
export KERNEL_PYTHON_PATH=/path/to/isolated-kernel-target
bash benchmark/hisparse/split_kv/run.sh decode \
  --model /path/to/GLM-5.2-w4a8 --prefill-host P_IP --decode-host D_IP \
  --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --interface lo
```

P 自动创建 Ascend transfer 的配置存储：framework 的 `start_disagg_service` 在 Prefill 调用 `memfabric_hybrid.create_config_store(ASCEND_MF_STORE_URL)`；两侧使用同一 `tcp://P_IP:24669`，无需另起手工 store。Decode 中的 TP 共享 KV 另使用进程内创建的 localhost store。若容器网络不允许上述地址绑定、访问或传输，脚本会失败，需先修复实际网络环境。

两侧的 `--tp`、`--bs`、端口、内存比例等参数须一致；默认 TP16、DP1、BS16、MTP3/Q4、mode1、C12288、BF16 KV、context 67584、page 128、max-total-tokens 163840、静态显存比例 0.92。扩大 BS 时在两侧同时指定 `--bs N`，并根据现场显存和容量设 `--max-total-tokens`；不能把不同配置的数据当成同一基线。`--help` 显示完整参数。以上启动参数来自历史单机 203 TPS 配置的已知部分；PD 结构与历史单机不同。

本脚本固定 Q4：decode 捕获 token 行数为 `graph-bs * 4`，须能被 TP 整除；`graph-bs` 默认等于 `bs`，可单独设更大的捕获桶。TP16 时默认图桶应为4的倍数；默认 `bs=18` 会在模型加载前被参数预检拒绝。满足此关系仍不保证具体模型的图能工作：本轮 BS20 原生 verify 图启动阶段出现 HCCL/AICPU 参数错误，开始捕获时可用显存约14.83GB，未发现 OOM 证据，尚未产生BS20性能数据。默认推荐已完成全流程的BS16，不把BS20失败记为零TPS，也不把相邻模型支持BS20扩大为当前GLM路径已通过。

如需跑不启用 HiSparse 的原生 KV 对照，在 P、D 两侧都增加 `--kv-mode native --bs 6 --graph-bs 8`。原生 KV 仍使用模型自身的 DSA/MLA attention cache，不会切换成 dense attention。`--bs 6` 是运行时并发上限；`--graph-bs 8` 是目标 Q4 verify 图的捕获桶，脚本保持 `--max-running-requests 6`。NPU draft decode 的单 token 行按 TP16 对齐到16槽，draft-extend 与 target verify 使用8槽；这不改变客户端发出的请求数。Native 模式每阶段显式关闭 sparse-KV offload 和 fused LIM Copy-SFA，并清除 LIM/GVA 专属环境覆盖；保留 MTP、Ascend PD 传输和性能阶段 decode radix prefix cache。它不配置 `C12288` sparse resident cache。reference、强制回退精度校验以及性能阶段的输入、缓存门禁和2轮预热+5轮正式测量仍照常执行。`--graph-bs` 默认等于 `--bs`，未指定 `--kv-mode` 时仍为 `hisparse`；BS6 native 对照需实际通过两机精度、cache 和 decode admission 检查后再报告性能。

后续隔离诊断的 BS20 target verify 图已成功捕获，之前的 HCCL 错误没有复现；随后明确遇到 draft decode 的空桶断言：每请求1个token，20行不满足TP16对齐。配套 Python 修复仅在 NPU 普通 EAGLE draft decode 中把桶向上补齐，BS20使用32槽draft图；target verify保持20×4，draft extend按实际宽度保持20×4。额外槽沿用已有零号padding行，输出裁回实际请求；BS16桶不变，CUDA/ROCm行为不变。CPU桶选择测试已通过，真实BS20图、精度和性能仍须本轮验证；这不被描述成修复了此前未复现的HCCL错误，也不覆盖multi-layer draft路径。

环境记录保留 `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` 的原始设置；未设置时 JSON 为 `null`，框架 `environ.py` 的当前默认值为128。本脚本不为记录而更改它。本轮BS20失败后的环境重建读到128，不支持“沿用64导致80个verify token超限”的猜测；死进程的有效值没有被事后直接读取。

Decode 所有阶段显式设置 `--disaggregation-decode-extra-slots 0`，避免默认的额外请求槽同时扩大 C12288 resident 预算。性能阶段启用实验性的 `--disaggregation-decode-enable-radix-cache`，让共享前缀有机会复用物理槽；参考与强制回退阶段保持 Decode radix cache 关闭。此组合的 MF prefix 复用尚未实机验证。脚本在性能结束时从 Decode 原始打屏提取最大 `#running-req`，缺失或小于目标 BS 时以非零退出，避免把客户端排队并发误称实际 Decode BS。

`--disable-decode-prefix-cache` 仅用于诊断，需同时设 `--max-total-tokens` 至少为 `BS * (64000 + 1024 + 128)`；这只是保守预检，实际池大小、空闲显存与 admission 仍需查看运行日志。该模式不保证满足 63872 cached tokens 门禁，不应与 prefix100 正式数据混用。

Decode 控制端执行：

1. 等 P readiness；启动关闭 KV offload 的原生 D 和路由器。除 `/health` 外，还等待 `/workers` 中预期的 P、D URL 以正确角色注册且健康，再发送请求。使用独立冻结的自然语言取值输入，测试 512/64、4096/64 各四条相同输入，以及 64000/64 两条 greedy 请求；每条必须正确输出文首密钥的逐行重复前缀，每组 token 必须一致，保存为 `reference-short.json`、`reference-long.json`、`reference-target64k.json`。
2. 停止本次启动的 D 和路由器，启动开启 KV offload 的 forced-retraction D（`SGLANG_TEST_RETRACT=1`、间隔 3，`cpu_tensor` 备份）及路由器。三组请求分别逐 token 比对对应的原生 reference，且每组响应 `meta_info.num_retractions` 总数大于零；结果保存为对应 `functional-*.json`。这是这三组输入的 greedy 一致性检查，不代替完整模型评测。
3. 停止本次功能模式的 D 和路由器，启动不含强制回退开关的性能 D 和路由器。默认先在 decode prefix cache 开启的实际配置下做一次独立语义检查：64000-token 取值输入 seed 输出 1，再发 BS 条并发请求，各输出 64，要求答案正确、与原生 target64k 逐 token 一致且每条 cached_tokens 为正整数；保存为 `performance-prefix-accuracy.json`，不计入吞吐测量。随后每轮先对 P、D 分别调用 `/flush_cache`，再发历史随机 64000 输入、1 输出的 prefix seed，随后并发测量。默认调用随仓冻结的历史 `bench_serving` 客户端，固定两轮 warmup 和五轮 formal，保留每轮完整报告与原始结果。任一请求错误、输入/输出长度不符或每条 cached tokens 不等于 63872 时退出非零。可用 `--client stdlib` 显式选择本目录的 SSE 客户端用于诊断，诊断客户端还严格要求结束原因是 length。显式关闭 decode prefix cache 时，此 cache 语义专项标为 skipped；仍不把该诊断模式当作正式对齐成绩。

脚本为每个自己创建的服务分配独立 session 和随机运行标记，核对标记及进程启动时间后逐个发送 SIGTERM，等待后对仍存活的本次进程发送 SIGKILL；不使用 `pkill`、不停止其他容器。同 session 中若进程暂时读不到标记，会最多等待一秒并复查退出与身份；持续未知的活进程仍拒绝发信号，并记录 stat 字段供排障。清理失败会报错并保留证据。D 结束后，可在 P 的命令窗口按 Ctrl-C 停止本任务的 Prefill。任何阶段失败后保留原始日志；重试应启动一次新运行，不复用已有结果。

## 输入、输出与比较口径

默认输入为 `fixtures/prefix100_64000_token_ids.json.gz`，解压后 64000 个 token ID；其 SHA256 与 manifest 在运行时校验。它从核验过的历史 warm/formal fixture 导出，16 条历史正式请求与 warm prefix 的 token ID 一致。可用 `--token-ids-json /path/file.json` 或 `.json.gz` 传入 JSON 整数数组（也接受 `{"input_ids": [...]}`）；覆盖后仅检查长度和 tokenizer 词表范围，**不再声称与历史输入一致**。

上述默认输入和覆盖参数仅用于性能。精度阶段使用 `fixtures/precision/` 中独立的 512/4096/64000 token 自然语言取值输入，校验文件 SHA、长度、词表及 tokenizer SHA；生成方法和验收范围见 [精度输入说明](fixtures/precision/README.md)。64 是输出 token 上限，不是完成六十四行。原历史随机 token 切片在 P 自身计算的单 token 路径上也会重复分叉，失败日志保留；这不作为强制回退前后的稳定 oracle，也不声称已修复该原生非一致现象。性能的冻结随机输入、prefix100、C12288、预热、五轮正式测量及 cached63872 门禁保持原样。

每次运行默认写到当前目录下 `split-kv-runs/prefill-<UTC>/` 或 `decode-<UTC>/`，可用 `--output-dir` 指定目录。包含 `config.json`、`runtime.json`、白名单环境、各阶段完整服务/router 日志、`controller.log`、固定输入副本、每条响应、压缩 SSE 原始事件、每轮 JSON、`exitcode.txt` 和 `summary.json`。每轮保留客户端完整报告；汇总五轮正式输出 TPS 的中位数、范围及总体标准差。stdlib 诊断模式报告首 token 时间 TTFT、首尾 token 间隔的 TPOT、总延迟以及服务端返回时的 MTP 接受长度/接受率。输出吞吐是客户端测得的成功输出 token 数除以测量窗口；它**不是 scheduler 的 decode-only TPS**，也不是论文根据 ITL 计算的 TPS。还需披露两机 PD 与历史单机、两轮 warmup/五轮全部保留与旧 clean 三轮筛选的差别。运行目录创建之前的失败以及 SIGKILL 无法写入 `exitcode.txt`，此时以 shell 退出码和终端首错为准。

日志应另外核对实际 KV cache 分配、prefill/decode batch/吞吐、MTP 接受长度和服务的真实配置。请求回退检查不代替完整模型评测；长度/cache 门禁也不等于性能达标。已完成的设备小探针范围和未完成项见验证记录，不把启动就绪或 HTTP 200 当作 PD 通过。

## 小规模设备检查

在空闲、已授权的设备上，先 source CANN，设置当前 framework 的 `PYTHONPATH` 和配套 `KERNEL_PYTHON_PATH`。下面两个脚本会实际使用 NPU，首次运行是否通过需要以原始日志和退出码判断：

```bash
export PYTHONPATH="$PWD/python:$KERNEL_PYTHON_PATH${PYTHONPATH:+:$PYTHONPATH}"
export MF_HYBM_USE_VMM_SEGMENT=1
export HCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo

# Single-device Q4/cross-page, 128-head and C12288 eager/graph oracle checks.
ASCEND_RT_VISIBLE_DEVICES=0 python benchmark/hisparse/split_kv/probe_copy_sfa.py \
  --kernel-checkout /path/to/sgl-kernel-npu

# Real TP-shared MF K/V snapshots, owner restore and per-rank index/scale.
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  torchrun --standalone --nproc-per-node=16 \
  benchmark/hisparse/split_kv/probe_mf_retraction.py
```

MF 探针使用两层、384 个物理槽，以原始字节校验跨页备份、旧槽覆盖、非 owner 快照污染后的 owner 恢复，以及各 rank 的 index/scale；`reset_requests` 仅记录调用，未构造完整 manager。Copy-SFA 探针复用 kernel 仓库的原有断言并输出实际 CPU oracle 误差。两者不能代替完整 PD 或性能验证。
