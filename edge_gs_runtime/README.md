# edge_gs_runtime

`edge_gs_runtime` 是一个面向多用户边缘视频帧渲染的阶段 A 在线原型。它把离线生成的 trace 输入转成逐帧任务，然后依次经过：

- `TraceReader` 读取任务到达
- `CostModel` 估计每帧基础渲染开销和 `g(q)` 参数
- `QoSPlanner` 做慢回路的目标帧率/质量分配
- `RealTimeScheduler` 做快回路的实时选帧
- `GPUWorker` 执行 `dry_run` 或真实渲染
- `MetricsCollector` 导出摘要和事件日志

## 目录

- `apps/run_stage_a.py`: 运行入口
- `apps/prepare_sample_inputs.py`: 从现有 `es/systemSimulation` 输出生成 runtime 可用的样例 CSV
- `core/`: runtime、task、session、metrics
- `control/`: planner、scheduler、cost_model
- `trace/`: trace CSV 读取
- `worker/`: dry run 执行器与 ES 真实渲染适配层
- `sample_inputs/`: 便于快速联调的输入样例
- `outputs/`: 运行输出

## 输入文件

运行至少需要两份 CSV：

- `trace CSV`
  需要包含 `Frame_ID, User_ID, R, D, Pred_Cost, Param_a, Param_b, Param_c`
  如果带 `Model` 列，runtime 会把它当作 `scene_id`
- `cost CSV`
  需要能解析出场景列和 `base_cost_mean/Pred_Cost + a/b/c`

如果你当前没有完整的上游 `cleanAndMergeData.py` 依赖文件，可以先用：

```bash
cd /root/3dgs-streaming/mini-splatting/edge_gs_runtime
/root/miniforge3/envs/mini_splatting/bin/python apps/prepare_sample_inputs.py
```

它会从现有的 `es/systemSimulation/01 traceGeneration/simulation_trace_bicycle.csv` 生成：

- `sample_inputs/generated/trace_runtime_from_systemsim.csv`
- `sample_inputs/generated/cost_runtime_from_trace_means.csv`

这里的 cost CSV 是从现有 trace 聚合得到的“场景级平均成本”版本，主要用于把 runtime 直接跑通。因为当前仓库里缺少原始 `viewports_20260105.json`、`render_times_*.csv`、`fit_params_per_view.csv`，所以不能完全复现最原始的 cost-field 清洗流水线。

## Dry Run

```bash
cd /root/3dgs-streaming/mini-splatting/edge_gs_runtime
PYTHONPATH=. /root/miniforge3/envs/mini_splatting/bin/python apps/run_stage_a.py \
  --trace_csv sample_inputs/generated/trace_runtime_from_systemsim.csv \
  --cost_csv sample_inputs/generated/cost_runtime_from_trace_means.csv \
  --dry_run \
  --summary_path outputs/generated_dry_run/summary.json \
  --events_path outputs/generated_dry_run/events.json
```

输出内容：

- `summary.json`: 整次运行的统计摘要
- `events.json`: 逐事件日志
- `outputs/frames/`: 每个成功任务的 dry-run 占位结果

## 真实渲染

`GPUWorker` 现在已经支持通过 ES 适配层接入真实渲染：

```bash
cd /root/3dgs-streaming/mini-splatting/edge_gs_runtime
PYTHONPATH=. /root/miniforge3/envs/mini_splatting/bin/python apps/run_stage_a.py \
  --trace_csv sample_inputs/generated/trace_runtime_from_systemsim.csv \
  --cost_csv sample_inputs/generated/cost_runtime_from_trace_means.csv \
  --model_root /root/3dgs-streaming/mini-splatting/ms/eval \
  --dataset_root /root/3dgs-streaming/mini-splatting/gs/dataset \
  --camera_split test \
  --summary_path outputs/generated_real/summary.json \
  --events_path outputs/generated_real/events.json
```

当前真实渲染适配层做的是：

- 按 `scene_id` 加载模型目录下的 `cfg_args`
- 用 `Scene + GaussianModel + gaussian_renderer.render` 执行真实渲染
- 用 `task_id % num_cameras` 稳定地选一个相机视角

注意：

- 真实渲染需要 CUDA 环境
- 当前 trace 还没有完整的真实视点矩阵，所以真实渲染先采用“离散相机索引映射”方案接通执行链路
- 在当前这台机器上，如果 `torch.cuda.is_available()` 为 `False`，真实渲染会直接报清晰错误并退出

## 输出说明

- `summary.json`: 成功率、丢帧率、平均排队时延、平均执行时长等
- `events.json`: `ARRIVAL/START/FINISH/DROP` 全量事件
- `outputs/frames/*.txt`: dry run 结果
- `outputs/frames/*.png`: 真实渲染结果

## 环境

当前联调用的是：

- Python: `/root/miniforge3/envs/mini_splatting/bin/python`
- 推荐启动方式：在 `edge_gs_runtime` 目录下加 `PYTHONPATH=.`
