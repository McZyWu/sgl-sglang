# Ascend split-KV：两机 PD 自动启动与验证

本目录从两个已准备好的容器分别运行一条命令：P 容器常驻 Prefill；D 容器自动依次运行普通参考、强制回退功能验证、正式压测。脚本不依赖 Codex、远程登录或外部 PASS 文件。**本脚本尚未经过真实 NPU、MF 或两机 PD 实测**；服务是否能跑通、精度和吞吐需要以实际日志和退出码判断。

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

Decode 所有阶段显式设置 `--disaggregation-decode-extra-slots 0`，避免默认的额外请求槽同时扩大 C12288 resident 预算。性能阶段启用实验性的 `--disaggregation-decode-enable-radix-cache`，让共享前缀有机会复用物理槽；参考与强制回退阶段保持 Decode radix cache 关闭。此组合的 MF prefix 复用尚未实机验证。脚本在性能结束时从 Decode 原始打屏提取最大 `#running-req`，缺失或小于目标 BS 时以非零退出，避免把客户端排队并发误称实际 Decode BS。

`--disable-decode-prefix-cache` 仅用于诊断，需同时设 `--max-total-tokens` 至少为 `BS * (64000 + 1024 + 128)`；这只是保守预检，实际池大小、空闲显存与 admission 仍需查看运行日志。该模式不保证满足 63872 cached tokens 门禁，不应与 prefix100 正式数据混用。

Decode 控制端执行：

1. 等 P readiness；启动普通 D 和路由器，四条相同短输入的 greedy 输出必须一致，保存为 reference。
2. 停止本次启动的 D 和路由器，启动 forced-retraction D（`SGLANG_TEST_RETRACT=1`、间隔 3，`cpu_tensor` 备份）及路由器。短请求输出必须逐 token 等于 reference，且响应 `meta_info.num_retractions` 总数大于零。
3. 停止本次功能模式的 D 和路由器，启动不含强制回退开关的性能 D 和路由器。每轮先对 P、D 分别调用 `/flush_cache`，再发一条 64000 输入、1 输出的 prefix seed，随后并发测量。默认调用随仓冻结的历史 `bench_serving` 客户端，固定两轮 warmup 和五轮 formal，保留每轮完整报告与原始结果。任一请求错误、输入/输出长度不符或每条 cached tokens 不等于 63872 时退出非零。可用 `--client stdlib` 显式选择本目录的 SSE 客户端用于诊断，诊断客户端还严格要求结束原因是 length。

脚本为每个自己创建的服务分配独立 session 和随机运行标记，核对标记及进程启动时间后逐个发送 SIGTERM，等待后对仍存活的本次进程发送 SIGKILL；不使用 `pkill`、不停止其他容器。清理失败会报错并保留证据。D 结束后，可在 P 的命令窗口按 Ctrl-C 停止本任务的 Prefill。任何阶段失败后保留原始日志；重试应启动一次新运行，不复用已有结果。

## 输入、输出与比较口径

默认输入为 `fixtures/prefix100_64000_token_ids.json.gz`，解压后 64000 个 token ID；其 SHA256 与 manifest 在运行时校验。它从核验过的历史 warm/formal fixture 导出，16 条历史正式请求与 warm prefix 的 token ID 一致。可用 `--token-ids-json /path/file.json` 或 `.json.gz` 传入 JSON 整数数组（也接受 `{"input_ids": [...]}`）；覆盖后仅检查长度和 tokenizer 词表范围，**不再声称与历史输入一致**。

每次运行默认写到当前目录下 `split-kv-runs/prefill-<UTC>/` 或 `decode-<UTC>/`，可用 `--output-dir` 指定目录。包含 `config.json`、`runtime.json`、白名单环境、各阶段完整服务/router 日志、`controller.log`、固定输入副本、每条响应、压缩 SSE 原始事件、每轮 JSON、`exitcode.txt` 和 `summary.json`。每轮保留客户端完整报告；汇总五轮正式输出 TPS 的中位数、范围及总体标准差。stdlib 诊断模式报告首 token 时间 TTFT、首尾 token 间隔的 TPOT、总延迟以及服务端返回时的 MTP 接受长度/接受率。输出吞吐是客户端测得的成功输出 token 数除以测量窗口；它**不是 scheduler 的 decode-only TPS**，也不是论文根据 ITL 计算的 TPS。还需披露两机 PD 与历史单机、两轮 warmup/五轮全部保留与旧 clean 三轮筛选的差别。运行目录创建之前的失败以及 SIGKILL 无法写入 `exitcode.txt`，此时以 shell 退出码和终端首错为准。

日志应另外核对实际 KV cache 分配、prefill/decode batch/吞吐、MTP 接受长度和服务的真实配置。短请求回退检查不代替真实 MF 多卡单算子验证、长输入数值精度或模型评测；长度/cache 门禁也不等于性能达标。当前仓库只做了 CPU 模拟与静态检查，没有实机通过数据。
