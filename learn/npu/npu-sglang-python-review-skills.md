# NPU、SGLang 与 Python 静态审查技能选型

日期：2026-11-17

## 目的与结论

本记录整理适合检视 Ascend NPU、SGLang Python 包装层和算子调用边界的现成 agent skills，并区分技能原文明确覆盖的内容与针对本仓库提出的补充建议。有限检索未找到一份同时完整覆盖 Python 代码质量、SGLang 仓库约定和 NPU 算子正确性的现成技能。建议采用分层审查：Python 通用清单帮助发现语言层问题；SGLang 仓库技能约束改动位置、运行时状态和测试方式；Ascend 技能按改动属于框架迁移、AscendC 还是 Triton 选择；Trail of Bits 可用于高风险安全差异审查。它们是审查辅助，不是目标分支、依赖版本或芯片行为的权威替代品。

## Ascend 官方技能：按改动类型使用

### NPU 迁移适配

[Ascend`npu-adapter-reviewer`](https://github.com/Ascend/agent-skills/blob/master/skills/npu-adapter-reviewer/SKILL.md) 面向 GPU 到 NPU 的迁移审查。其思路包括扫描 CUDA 与 `torch.cuda` 依赖、识别适配堵点，并产出自动适配、验证和报告。适合迁移 PR 的初步清点与覆盖面核对。技能报告模板中的预填通过项、示例性能提升数字都只是占位示例，必须依据本次源代码和真实日志重写，不能当作验证证据。它的目标是迁移适配流程，因此不能当成只读代码审查流程，也不能把自动替换建议视为 API 语义的真理；每个改动仍须追到实际调用方、当前 Ascend 插件和受支持版本。

### AscendC 算子

[Ascend `ascendc-operator-code-review`](https://github.com/Ascend/agent-skills/blob/master/skills/ascendc-operator-code-review/SKILL.md) 专门面向 AscendC，覆盖数值正确性、指针和内存安全、资源生命周期、输入约束、并发、Tiling 和动态 shape、ABI 等。适合通过调用链和数据流检查 host/kernel 间契约及边界条件。技能要求结合配套 references 使用。目录内实际包含 `SKILL.md` 与 `references`；原文报告样式还提到 `agents/ascendc-ops-reviewer/style/...` 跨目录路径，因此不能假设仅复制该技能目录就带齐全部资源，应逐一检查相对路径和跨目录依赖。其中人为定义的分数阈值（如“超过60”）是启发式，不是统计置信度或缺陷发生概率，不能直接作为通过门槛。若被审改动只是 Python 调度或 PyTorch 封装，不应机械套用 AscendC 检查项。

### Triton 算子

[Ascend `triton-operator-code-review`](https://github.com/Ascend/agent-skills/blob/master/skills/triton-operator-code-review/SKILL.md) 关注 Host/Device 分工、mask 与越界、核函数类型、dtype 和归约精度、atomic 控制流、隐式同步及访存。适合检查 Triton kernel 与调用端的 shape/dtype/布局契约。硬件版本和 dtype 支持矩阵需参考技能配套资料，并结合项目实际编译器、CANN 与芯片版本核实。原文中的部分规则较宽泛，不能脱离目标版本解释为普遍硬件定律。

## SGLang 仓库技能：遵循目标分支惯例

SGLang 仓库已有以下技能文件：

- [`large-class-style/SKILL.md`](https://github.com/sgl-project/sglang/blob/main/.agents/skills/large-class-style/SKILL.md)：`ModelRunner` 是明确列出的 frozen orchestration-only 文件；Scheduler、TokenizerManager、ModelRunner 三大类的 `__init__` 遵循 `init_*` 拆分约定。适合判断改动应落在何处、是否破坏初始化边界。
- [`sglang-runtime-context/SKILL.md`](https://github.com/sgl-project/sglang/blob/main/.agents/skills/sglang-runtime-context/SKILL.md)：理解配置状态、运行时上下文、状态生命周期和覆盖入口。改动 `server_args`、模型覆盖、模块级状态或每次 forward 状态时，先明确状态所属层级，避免引入隐式全局状态。
- [`env-var-conventions/SKILL.md`](https://github.com/sgl-project/sglang/blob/main/.agents/skills/env-var-conventions/SKILL.md)：新增、改名或迁移环境变量时，核对集中定义、读取和覆盖约定，避免散落的 `os.getenv` 或兼容别名遗漏。
- [`kernel-organization/SKILL.md`](https://github.com/sgl-project/sglang/blob/main/.agents/skills/kernel-organization/SKILL.md)：核对 kernel API、注册元数据、测试/benchmark 和模型专用实现的归属及后端边界。
- [`write-sglang-test/SKILL.md`](https://github.com/sgl-project/sglang/blob/main/.agents/skills/write-sglang-test/SKILL.md)：按 `CustomTestCase`、CI 注册、测试目录和合适 fixture 编写测试；其要求不是简单以 pytest 命令替代仓库测试布局。
- [`mechanical-refactor-verify/SKILL.md`](https://github.com/sgl-project/sglang/blob/main/.agents/skills/mechanical-refactor-verify/SKILL.md)：对文件迁移、拆分、重命名等机械重构构造可复现证据，防止“看起来只是搬代码”时悄悄混入行为修改。

这些技能来自 SGLang，但 `main` 上的要求不一定适合旧 Ascend 分支或特定 PR 基线。审查时应先确定目标分支、提交和相邻实现，再按对应版本的惯例判断；不能为了迎合新规范对旧分支做无关的大规模重构。

## Python 通用技能与安全差异审查

[第三方 `python-code-review`](https://github.com/minhhh/agent-skills/blob/main/skills/python-code-review/SKILL.md) 提供 Python 专项清单，包含可变默认值、异常吞噬、资源管理、None/类型安全、async 错误、测试质量和常见性能问题，并依赖同仓库 [`code-review-principles/SKILL.md`](https://github.com/minhhh/agent-skills/blob/main/skills/code-review-principles/SKILL.md) 与 [`python-dev/SKILL.md`](https://github.com/minhhh/agent-skills/blob/main/skills/python-dev/SKILL.md)。这是个人社区仓库，非 Python 官方规范；独立使用会缺少依赖上下文。原文将缺少类型标注、独立 await 串行化等问题直接赋予很高严重性，存在过度分级风险；应按公开 API、实际并发语义和调用方影响重新定级。它偏 pytest 的建议也要适配 SGLang 的 `CustomTestCase` 和 CI 注册习惯。

[Sentry `code-review`](https://github.com/getsentry/skills/blob/main/skills/code-review/SKILL.md) 是产品维护者视角的通用 review 技能，涉及运行时、性能、安全、兼容性和测试，也含 Python/Django 示例；可作为温和的通用审查底线，但不提供全面的 Python 语法清单或 Ascend 内核细则。

[Trail of Bits `differential-review`](https://github.com/trailofbits/skills/blob/main/plugins/differential-review/skills/differential-review/SKILL.md) 面向安全风险较高的代码差异，强调变更风险分级、历史/blame、调用影响范围、测试覆盖和具体攻击场景。应将它用于认证、输入边界、内存安全或安全修复回归等高风险 diff；它需要配套 methodology、reporting 等资料，且产出重点是安全审计，不是 Python 类型、性能和算子数值的完整 review。

## 针对 SGLang NPU 改动的补充审查建议（自拟）

下列项目是本记录针对仓库场景提出的补充，不是上述技能原文保证覆盖的能力：

1. 追完整调用链：模型/调度入口 → Python adapter/manager → `torch.ops` 或 kernel wrapper → host 校验/元数据 → device kernel → 输出消费者。确认参数顺序、shape、dtype、stride/连续性、设备归属、返回值和原位写约定一致。
2. 核对 NPU stream/event 依赖和隐式同步：CPU 侧准备、H2D/D2H 拷贝、异步 kernel、跨 stream 消费之间必须有可证明的顺序；不要把“调用成功”当作异步数据已可读。
3. 核对 KV cache 与临时 buffer 的容量、slot/page 映射、生命周期及并发复用，检查请求结束、取消、重试和长度变化时是否遗留或覆盖状态。
4. 对齐 dtype、shape、layout、动态长度和边界输入；检查掩码、越界保护、归约精度及 fallback 行为，并区分 host 校验与 device kernel 实际假设。
5. 保持 GPU 路径不受影响，除非已有 GPU 缺陷且改动确有必要；用相同配置的精度与性能基线验证行为，不能用不同 batch、长度、版本或环境变量的结果作直接比较。
6. 记录静态事实与推断的区别。源码审查无法证明硬件并发、CANN 特定语义或性能结论时，应标注证据缺口并建议窄范围验证，不把静态推测写成已验证结果。

## 适用边界

技能选择应以改动语言和执行层为准：Python 调度/适配优先 Python 与 SGLang 规范；AscendC/Triton kernel 再叠加对应 Ascend 技能；安全敏感 diff 才叠加 Trail of Bits。静态 review 可以发现代码路径、契约和覆盖缺口，但不能替代目标芯片编译、精度验证、性能基线或特定硬件行为验证。

## 仓库落地与每次修改的入口

根 `AGENTS.md` 要求每次代码改动前和收尾前重读 `.agents/skills/npu-sglang-python-review/SKILL.md`，并按改动域读取匹配规则。固定版本的原始资料保存在 `.agents/review-sources`，由 `sources.lock.json` 记录来源、commit、license 与 SHA；它们只作审查参考，不自动执行迁移。迁移示例数字不是证据，AscendC 人工分值不是概率。Triton shared API 文档已正确路由。SGLang 六项既有技能继续按目标分支使用。若 AscendC 上游报告样式缺失，采用本地 file/line/trigger/effect/evidence/minimal-fix finding 格式。
