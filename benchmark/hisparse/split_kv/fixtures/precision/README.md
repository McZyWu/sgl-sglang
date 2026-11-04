# 固定自然语言取值用例

这三份输入仅用于精度与回退恢复对照，不用于性能测量。性能继续使用上级目录的历史 `prefix100_64000_token_ids.json.gz`、冻结客户端和每请求 63872 cached tokens 门禁。

模型为现场 GLM-5.2 W4A8 的 tokenizer。用 chat template 构造 user 消息，显式设置 `enable_thinking=False`、`add_generation_prompt=True`。文首定义唯一密钥 `731954`；中间重复固定自然语言说明；尾部要求每行输出文首密钥，重复六十四行且不解释。尾部没有再次给出密钥。模板 prefix/suffix 分别为 24/25 token，中间填充 token 重复并截取到总输入恰好 512、4096、64000 token。详细字符串、生成方法及 tokenizer 文件 SHA 见 manifest。

| Case | 输入 token | 输出 token 上限 | 并发请求 | 原始 JSON SHA256 |
| --- | ---: | ---: | ---: | --- |
| short | 512 | 64 | 4 | 7d0963fb6f9448c77d304cdd6f1eb382b05133446766fd58071d8fa41d8d0303 |
| long | 4096 | 64 | 4 | 18c8dcc3f7f10a6fba36f80d4947590b5f6ceb7dfed8ffe01716554160e97273 |
| target64k | 64000 | 64 | 2 | 8858ecfa3bc7cdb9a8a1d49dded624d88dc6d5248fb12dfd8a342b144223790a |

每份 gzip 内保留实机诊断时的原始 JSON 字节；manifest 同时记录 gzip 和解压内容的 SHA256。运行时校验这些 hash、token 长度和词表范围，并核对实际 tokenizer 文件。精度输入不受性能输入的 `--token-ids-json` 覆盖影响。

64 是输出 token 上限，不是已经输出六十四行。已采原生输出为十二次 `731954\n` 后再接 `731954`，共 64 token。正确性要求输出是无限重复 `731954\n` 的有效前缀且至少含一个完整行；允许在末行中间截断，不接受前导解释、错误行或仅在任意位置包含密钥。原生组内输出还必须逐 token 一致；强制回退组逐 token 对齐原生组，并实际发生回退。

采用独立精度输入的原因是，历史性能随机 token 输入的 512-token 切片在原生 PD 上重复运行会分叉。额外 teacher-force 试验将首个生成 token 附加到输入，构成 513 输入、只生成一个 token：该 token 由 P 计算并交接，D 未做 target forward，四次结果仍为 `112/241/112/112`。因此不能把原生随机输入的跨请求非一致直接归为本次 D 回退修复故障；其模型数值或状态根因尚未解决。失败数据已保存，不将自然语言用例通过写成随机输入一致性已修复。

2026-11-04，209 P + 217 原生 D 的独立诊断中，short 的预热、四次顺序和四次并发均正确且 exact；long 四请求、target64k 两请求也正确且 exact。输入在强制回退候选测试前冻结。这些结果只覆盖固定取值用例，不是 GSM8K 等通用模型评测，也不是正式性能数据。完整 controller、强制回退和性能结果以本轮新日志为准。
