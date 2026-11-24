# FGW-mHC

Ascend 950 的 TileOPs mHC 前向算子原型。支持四条残差流、BF16 激活和 FP32 权重，hidden 必须是 64 的倍数。

- `mhc_pre(phi, x, b, alpha_pre, alpha_post, alpha_res, sinkhorn_repeat, sinkhorn_eps=0.02)` 返回 `(x_res, x_layer, h_post)`。Pre 分三次启动，在 NPU 上完成分块投影和平方和、门控和 Sinkhorn、残差和层输入投影。
- `mhc_post(x_layer_out, h_post, x_res)` 以 FP32 计算广播乘加，返回 BF16 的 `[B, 4 * H]`。输入 `x_res` 由 Pre 完成残差混合。

实现参照 [tilelang-asc PR #25](https://github.com/yangsichan/tilelang-asc/pull/25) 的 Ascend 方言、Vector 分块与 JIT 编译方式，计算遵循 [TileOPs mHC 参考](https://github.com/yyttt6/TileOPs/blob/ea0d2999a830243fd398e74bb30427a064078bac/workloads/sequence_modeling/mhc.py) 的非 shifted 语义。Pre 第一版使用 Vector SIMT 投影，后续可用 Cube 实现相同计算边界。JIT 缓存按输入形状和 Pre 标量参数特化。

## 在 Ascend 上验证

在 Linux Ascend 950 环境安装匹配的 CANN、PyTorch、torch_npu 和 [PR #25 所用的 TileLang Ascend checkout](https://github.com/yangsichan/tilelang-asc/tree/053ddb470aefcbc6972d994739983734eeb4eaf8)，然后在仓库根目录运行：

```bash
python -m pytest -q tests
python bench_mhc.py --list-cases
python bench_mhc.py --warmup 30 --repeat 100
```

Benchmark 使用 TileOPs manifest 的三个 BF16 用例，先检查数值，再报告预热后的 NPU event 中位耗时。Post 对照是 Torch NPU 的 `addcmul` 加 dtype 转换；Pre 对照是同规格的 eager 组合计算。输出 CSV 可重定向到文件。


## 运行并计算加速比

```bash
git clone https://github.com/zhangwenhao91/FGW-mHC.git
cd FGW-mHC
bash run_npu.sh --device npu:0 --output results.csv
```

需要在已激活的 Ascend Python 环境安装 `pytest`。脚本先运行正确性测试，再运行 benchmark。CSV 的 `speedup` 为 `baseline_us / tilelang_us`，两个延时都使用预热后 NPU event 样本的中位数。可用 `PYTHON` 选择解释器。

默认 Pre 基线为 `cann_eager_composite`，Post 为 `cann_addcmul_with_casts`。Post 通过 PyTorch NPU 调用 CANN 乘加路径，计入同规格所需的 FP32 输入转换与 BF16 输出转换。默认基线的加速比对应这些组合路径。

如果目标机已有同规格的 AscendC 扩展，可以用下面的命令直接计时。Python 模块导出的 `mhc_pre`、`mhc_post` 需采用与本仓同样的参数顺序、shape、dtype 和返回值，在当前 NPU stream 上完成全部计算。必要的重排和类型转换应放在适配函数内部，随算子一起计时。

```bash
bash run_npu.sh --baseline-module my_ascendc_mhc --output ascendc-results.csv
```
