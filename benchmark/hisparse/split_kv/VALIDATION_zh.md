# 本次交接的验证范围

记录日期：2026-11-04。本次提交提供回退修复、两机 PD 控制器、冻结压测客户端和容器构建脚本。真实 MF TP16 小探针、CopySFA 三个设备用例和配套 kernel 构建已通过；**完整 PD、整模精度与性能仍未完成，没有本次有效实测 TPS。**

| 项目 | 结果 | 范围 |
| --- | --- | --- |
| 原始 NoneType 回退问题 | CPU 重现后修复 | 原版读取 `self.k_buffer=None`；修复改读权威 MF split K/V |
| Linux 完整测试模块 | 8/8，通过，exit 0 | 2 项新增 retraction 测试、6 项既有 MF CPU 测试；MF/设备边界使用模拟 |
| 自动控制器自检 | 通过，exit 0 | 真实本机 HTTP/SSE、错误/cache 门禁、默认冻结客户端及 stdlib 的 2 warmup + 5 formal 模拟流程、活跃 BS、异常退出码；最新测试计数见本次检查日志 |
| 冻结资产与报告自检 | 3/3，通过，exit 0 | 原始 SHA、BS16 pickle 原字节、BS20 扩展、seed 1 输出、正式 1024 输出及 cached63872、输入缓存命中 |
| 静态检查 | 通过 | Ruff 0.15.1、isort 7.0.0、codespell 2.4.1、AST、空白/EOF、Bash 语法、文件大小与 diff check |
| kernel 构建及安装 | 通过，exit 0 | 217 A3/CANN 9.1.0 构建；两侧安装同一 wheel，检查 split K/V 算子 schema |
| 真实 MF 回收/恢复小探针 | 16/16 rank 通过，exit 0 | 每 rank 39 项检查，原始字节校验跨页 K/V、owner 恢复和每 rank index/scale；reset 为 spy |
| CopySFA eager/graph | 3/3，通过，exit 0 | Q4 跨页、128 head、C12288 Q4 首填；独立 CPU oracle 最大绝对误差分别为 0.00066914、0.00065416、0.00059916 |
| PD 首次尝试 | 失败，exit 1，已补启动门禁 | P/D 服务 ready 后 router 尚无 P worker，首组请求 503；不能记作精度或性能结果 |
| 完整 PD、模型精度、吞吐 | 验证中 | 等待修正控制器后重试；请求逐 token 对照也不能替代完整模型评测 |

Linux 8 项测试使用 Python 3.12.13、torch 2.10.0+cpu、torch_npu 2.10.0.post6、memfabric_hybrid 1.3.0。采集日志结束时间为 `2026-11-03T23:16:50+00:00`，pytest 输出 `8 passed, 15 warnings in 12.28s`。修复文件的 SHA256 为 `3faf59ed016a26bfe5447f7bf4dfa7c02a4f663fccfb911164aac20e9319a5c9`；新增 retraction 测试文件为 `339177370819502c9ff25944f8478ae31c24b8e5fc8fefeda02357e49f9585fa`。本次发布前核对过这两个文件与当时通过测试的版本一致。

全量 `scripts/lint/check_registered_tests.py` 检查退出 1：基线已存在的 `test/registered/unit/hardware_backend/test_npu_sparse_dispatch.py` 缺 CI 注册。本次没有修改该文件，也未将这一检查描述为通过。新增 retraction 测试包含 CPU CI 注册和 `unittest.main()`。

修复对应的调用链如下。新增搬运发生在回退/恢复路径，普通 decode 的计算路径未添加操作。

```mermaid
flowchart TD
  A[ScheduleBatch.retract_decode] --> B[release_req / backup_kv_cache]
  B --> C[Req.offload_kv_cache]
  C --> D[allocator.get_cpu_copy]
  D --> E[NPUMLATokenToKVPool.get_cpu_copy]
  E --> F[MF split K/V → bounded NPU scratch → independent CPU snapshot]
  G[load_cpu_copy] --> H[TP owner restores MF split K/V]
  H --> I[Each rank restores indexer and optional scales]
  I --> J[publish and reset request resident state]
```

历史参照为旧单机 TP16/DP1、BS16 的 `203.723894` output tokens/s，使用选定三轮子集的中位数。本次脚本冻结相同客户端和 64000/1024、prefix100、C12288、mode1、NEXTN3/Q4、BF16 KV 等已知条件，但使用两机 PD，并保留全部五轮正式结果。性能 D 额外启用实验性的 radix prefix 复用，避免 163840-token 池把 16 路 HTTP 请求实际限制为约两条独立 64K decode；所有 D 阶段额外请求槽固定为 0。以上差异须与实际服务配置和活跃 BS 一起报告，不能把旧单机数值或 CPU 模拟值当成本次 PD 结果。

可重复检查命令见 [README](README.md)。运行产生的 `config.json`、`runtime.json`、环境、服务日志、完整旧客户端报告、逐请求 JSONL、`summary.json` 和退出码，是容器实测结果的依据。

## 启动包装器的 spawn 回归修复

用户随后提供的日志在 `no_thp_launch.py:11` 重入 SGLang 启动，原始异常是 `An attempt has been made to start a new process before the current process has finished its bootstrapping phase`。调用链为 `Engine._launch_scheduler_processes → proc.start → spawn.prepare → runpy.run_path(__mp_main__) → no_thp_launch.py → runpy.run_module(sglang.launch_server)`。顶层执行的包装器在 spawn 导入主模块期间再次启动服务。日志中的另一处 `RecursionError` 出现在 `torch_npu` 异常钩子，不能据此把首错归为 KV 或 attention kernel 故障。

修复将 THP 系统调用、参数设置和服务启动全部放入 `main()`，由 `if __name__ == "__main__"` 保护。新增 3 项 CPU 单测覆盖普通/`__mp_main__` 导入无副作用、参数转发与单次启动，以及 prctl 失败时停止启动。另提供 `test_spawn_startup.py`，用真实 multiprocessing spawn、假的 SGLang 服务和 ctypes 边界，验证只启动一个父服务入口及一个子 worker。

本地 Windows Python 3.12 回放结果：旧 `90a6d765` 包装器退出 1，出现相同 bootstrapping 错误，记录到两次服务入口；修复版退出 0，记录到一次 prctl、一次服务入口和一次子 worker，参数保持一致。3 项单测也全部通过。该回放未加载真实 SGLang 模型、调用真实 THP 系统调用或初始化 NPU；容器中的真实服务启动与 PD 验证仍需重跑。

## 地址预检与扩大功能对照

用户在实际参数中填入 `--prefill-host 809`，随后 Prefill 初始化经过 `get_local_ip_auto → get_zmq_socket_on_host`，在 `tcp://809:*` 报 `Cannot assign requested address`。脚本现要求完整 IPv4，并在 NPU runtime 导入前只检查当前角色地址能否本地绑定。CPU 自检新增错误简称、地址类型、真实 loopback 绑定、对端地址不在本机绑定以及 `EADDRNOTAVAIL` 提示覆盖。

功能对照现使用关闭 offload 的原生 Decode 作为参考；开启 offload 并强制回退后，分别对 512/64、4096/64 各四条 greedy 请求逐 token 比较并检查回退次数。控制器的 13 项 CPU 自检通过，仍不代表真实模型已通过这些对照。性能阶段维持两轮 warmup、五轮 formal，每轮只清缓存一次；所有阶段合计 11 次清缓存被自检覆盖。

新增两个设备探针供本轮 209/217 验证使用：`probe_mf_retraction.py` 检查真实 MF/NPU/TP 的共享 K/V 快照和原始字节恢复；`probe_copy_sfa.py` 复用 kernel 仓库的三个 eager/graph 精度用例并打印误差。初次发布时仅静态检查，随后已取得上表范围内的真实设备结果。

## 209/217 实机准备与首轮 PD 修正

两侧为同一 B103 镜像、Python 3.12.13、torch 2.10.0+cpu、torch_npu 2.10.0.post6、MF 1.3.0；各使用 16 个 A3 die。模型 config、量化描述、权重索引和 tokenizer 配置的 SHA 在两侧一致；未对 404.8 GB 全量权重重新计算 SHA。Prefill 的 BF16 KV 分配约 16.77 GB，draft KV 约 0.22 GB。

配套 kernel 基线为 `b47332ce9736a7920139ad8f29364b27ef416dea`；构建时只在任务副本限制编译并发和规范化 shell LF。实际 wheel SHA256 为 `5de65d7eb4ee9acda2f86007eb02ada47ed71e4c0c390645c8dd70a6cac1286c`，两侧主 SO SHA 为 `0b27a873ddd35b789cbe649c1dd17509b6ffac4e62b88829ce856bbb502096f7`。

MF 1.3.0 wheel 的 AccOffload 库为官方零字节占位。用随包官方构建器在私有前缀生成 718752 字节 A3 ELF，SHA256 为 `7e30b856a5e1a37bece3ef18f9a8b60f054901ef0e38aa00176d7cbc668d1550`，原 site-packages 未修改。真实 MF TP16 探针使用此私有库通过。新增 `setup_memfabric.sh` 封装同一构建机制，封装自身尚待独立完整执行。

首次 D 运行目录为 `/workspace/splitkv-pd-live/runs/decode-20261104T073438446307Z`。首错为 router `No available servers: No prefill workers available`；`/health` 已返回 200，后台 `/workers` 尚未注册 P。单独实际查询确认 router 0.3.2 异步注册，因此新增精确 URL、角色和健康状态等待，不能靠固定延时。退出阶段还暴露 `/proc/stat` 读取活进程、随后 `environ` 已清空并变成 zombie 的竞态；清理逻辑新增状态与进程身份复核。

功能对照扩大为 512/64、4096/64 各四请求和 64000/64 两请求，参考与强制回退逐组比较；性能仍为两轮 warmup 与五轮全部保留的 formal。全部阶段共清缓存 13 次。这些控制器测试包含模拟服务，实际整模结果以新运行日志为准。

第二次运行目录为 `/workspace/splitkv-pd-live/runs/decode-20261104T075657552274Z`。正确通过 router worker 门禁后，native 512/64 首组四请求均 HTTP 200，但 `reference-short-1` 的第 2 token 为 241，其他三条为 112；其后输出也不同，因此原生一致性门禁失败，未运行强制回退或性能。`reference-short-1` 与 `reference-short-3` 的 cached_tokens 都是 384，但只有前者不同，不能仅据此把原因归为缓存。独立原生诊断将保存顺序/并发请求的 logprob 与批次信息，尚无原因结论。

第二次退出也遇到 token 缺失，说明仅立即重读 stat 仍不足以覆盖现场进程生命周期。新清理逻辑对同 session 的缺失 token 最多等待一秒，期间检查状态、birth、session 和标记；未知活进程持续存在仍报错，带 state/ppid/pgrp/session/flags/birth 证据。23 项 CPU 控制器自检通过，覆盖时间虚拟的延迟标记、持续未知和身份变化；此补充仍待真实退出路径验证，不宣称现场根因已确认。
