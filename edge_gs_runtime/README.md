# edge_gs_runtime

`edge_gs_runtime` 是一个面向多用户边缘视频帧渲染的阶段 A 在线原型。它把离线生成的 trace 输入转成逐帧任务，然后依次经过：

---

## 当前进展（2026-04-03）

### 已完成

#### 离线标定链路（`es/per_view_gq_pipeline/`）

- 在球坐标空间 (radius × yaw × pitch = 32×32×32) 均匀采样 32768 个视口，计算每个视口的 3D 相机位置 `(x, y, z)` 写入 `viewports_20260105.json`
- 逐视口在多个质量等级 `q` 下实测渲染耗时，拟合 `g(q) = a·q² + b·q + c`（Polynomial，取 R² 最高结果）
- 合并坐标与拟合参数，生成场景级 `simulation_cost_field_*.csv`，目前已完成 **bicycle** 和 **room** 两个场景

#### 在线运行时（`edge_gs_runtime/`）

所有核心模块均已实现并端到端跑通 dry run：

| 模块 | 文件 | 状态 |
|------|------|------|
| `TraceReader` | `trace/trace_reader.py` | ✅ 完成 |
| `CostModel` | `control/cost_model.py` | ✅ 完成，KDTree + IDW 插值，分场景，有均值降级回退 |
| `SessionRegistry` | `core/session.py` | ✅ 完成，维护活跃用户状态与 mean cost |
| `QoSPlanner`（慢回路） | `control/planner.py` | ✅ 完成，负载感知：超载时先降质量再降帧率 |
| `RealTimeScheduler`（快回路） | `control/scheduler.py` | ✅ 完成，速率准入控制 + deadline 可行性过滤 + EDF 选帧 |
| `GPUWorker` | `worker/gpu_worker.py` | ✅ dry_run 可用；真实渲染适配层已写好，依赖 CUDA |
| `MetricsCollector` | `core/metrics.py` | ✅ 完成，全量事件日志 + 统计摘要 |
| `StageARuntime` | `core/runtime.py` | ✅ 完成，主循环串联所有模块 |
| `run_stage_a.py` | `apps/run_stage_a.py` | ✅ 完成，命令行入口 |
| 多模型 feasibility 输入生成链 | `apps/build_feasibility_cost_csv.py` + `apps/generate_feasibility_trace.py` | ✅ 完成 |

#### 已跑通的实验

**Dry Run**（feasibility 双场景，bicycle + room，4 用户，10s）：

```
total_arrivals:       2700
total_success:        1316   (48.7%)
total_dropped:        1384   (51.3%，全部为 QUEUE_TIMEOUT)
avg_queue_delay:      0.0050s
avg_sojourn_time:     0.0066s
avg_exec_time:        0.0016s
num_users_served:     4
```

**真实渲染**（feasibility 双场景，bicycle @ iter 30000 / 194 cameras，room @ iter 30000 / 311 cameras，4 用户，10s）：✅ 已接通

```
total_arrivals:       2700
total_success:        1310   (48.5%)
total_dropped:        1390   (51.5%，其中 QUEUE_TIMEOUT: 1385，EXEC_TIMEOUT: 5)
avg_queue_delay:      0.0055s
avg_sojourn_time:     0.0081s
avg_exec_time:        0.0026s
num_users_served:     4
```

产出真实渲染帧：`outputs/frames/scene_bicycle_user_*_task_*_<camera_name>.png`

---

## TODO List

### 高优先级

- [ ] **扩充 Cost Field 场景覆盖**：目前只有 bicycle + room，需补充剩余 11 个场景的离线标定（`es/per_view_gq_pipeline/` 流水线），以支持完整多场景实验
- [x] **接通真实渲染链路**：`ESRenderAdapter` 已接通，bicycle/room 双场景真实渲染端到端验证通过，产出 `.png` 帧文件
- [ ] **降低丢帧率**：当前 `load_budget=1.0` 下丢帧率约 51%（QUEUE_TIMEOUT 为主，另有少量 EXEC_TIMEOUT），需调研原因并改进：Planner 降档策略调参、trace 负载水平调整、或引入更激进的准入控制
- [ ] **真实视点矩阵接入**：目前真实渲染用 `task_id % num_cameras` 选固定相机，trace 中的 `(x, y, z)` 位置尚未映射到真实 view matrix，需要建立 cost field 视口到 3DGS 相机的对应关系

### 中优先级

- [ ] **实验结果可视化**：目前输出为原始 JSON，缺少可直接用于论文的统计图表脚本（成功率 vs 负载、延迟分布、质量退化曲线等）
- [ ] **Planner 策略增强**：当前为基于场景均值的贪心降档，可引入前瞻性预测或连续优化以减少不必要的质量损失
- [ ] **trace 视点矩阵补全**：随机游走目前只更新 `(x, y, z)` 相机位置，在 orbit 模式下已自洽，但后续真实渲染需要完整的 view/projection matrix，需在 trace 中补写或在 worker 中实时计算

### 低优先级

- [ ] **多 GPU 扩展**：当前原型为单 GPU，若需验证多 GPU 负载均衡，需扩展 `GPUWorker` 与调度器
- [ ] **配置文件支持**：当前用 argparse，若需统一管理实验配置，可封装为 YAML 配置加载层
- [ ] **单元测试覆盖**：核心模块（CostModel、Planner、Scheduler）目前缺少自动化单元测试

---

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

如果你要模拟“不同用户访问不同模型”，推荐使用本目录新增的 feasibility 输入生成链，而不是单模型样例。

如果你当前没有完整的上游 `cleanAndMergeData.py` 依赖文件，可以先用：

```bash
cd /root/3dgs-streaming/mini-splatting/edge_gs_runtime
/root/miniforge3/envs/mini_splatting/bin/python apps/prepare_sample_inputs.py
```

它会从现有的 `es/systemSimulation/01 traceGeneration/simulation_trace_bicycle.csv` 生成：

- `sample_inputs/generated/trace_runtime_from_systemsim.csv`
- `sample_inputs/generated/cost_runtime_from_trace_means.csv`

这里的 cost CSV 是从现有 trace 聚合得到的“场景级平均成本”版本，主要用于把 runtime 直接跑通。因为当前仓库里缺少原始 `viewports_20260105.json`、`render_times_*.csv`、`fit_params_per_view.csv`，所以不能完全复现最原始的 cost-field 清洗流水线。

## Multi-Model Feasibility Inputs

如果你的目标是验证“多用户访问不同模型”的 runtime 输入链，推荐使用：

- `apps/build_feasibility_cost_csv.py`
- `apps/generate_feasibility_trace.py`

默认会基于下面两个已经生成好的 cost-field 场景：

- `bicycle`
- `room`

并产出：

- `sample_inputs/feasibility/cost_runtime_bicycle_room.csv`
- `sample_inputs/feasibility/trace_runtime_bicycle_room.csv`

### 1. 聚合多模型 cost CSV

```bash
cd /root/3dgs-streaming/mini-splatting/edge_gs_runtime
/root/miniforge3/envs/mini_splatting/bin/python apps/build_feasibility_cost_csv.py
```

默认输入来自：

- `/root/3dgs-streaming/mini-splatting/es/systemSimulation/pilot_cost_fields/simulation_cost_field_bicycle.csv`
- `/root/3dgs-streaming/mini-splatting/es/systemSimulation/pilot_cost_fields/simulation_cost_field_room.csv`

默认输出：

- `sample_inputs/feasibility/cost_runtime_bicycle_room.csv`

### 2. 生成多模型 trace CSV

```bash
cd /root/3dgs-streaming/mini-splatting/edge_gs_runtime
/root/miniforge3/envs/mini_splatting/bin/python apps/generate_feasibility_trace.py
```

默认行为：

- `User 0,1 -> bicycle`
- `User 2,3 -> room`
- 总时长 `10s`
- 模式 `SIMULTANEOUS`
- FPS 从 `30, 50, 60, 90` 采样

默认输出：

- `sample_inputs/feasibility/trace_runtime_bicycle_room.csv`

这份 trace 会显式写入：

- `Model`
- `x`
- `y`
- `z`

这样 runtime 会在对应场景 cost-field 内做空间查询，而不是退化成场景均值。

### 3. 用 feasibility 输入跑 dry run

```bash
cd /root/3dgs-streaming/mini-splatting/edge_gs_runtime
PYTHONPATH=. /root/miniforge3/envs/mini_splatting/bin/python apps/run_stage_a.py \
  --trace_csv sample_inputs/feasibility/trace_runtime_bicycle_room.csv \
  --cost_csv sample_inputs/feasibility/cost_runtime_bicycle_room.csv \
  --dry_run \
  --summary_path outputs/feasibility_bicycle_room/summary.json \
  --events_path outputs/feasibility_bicycle_room/events.json
```

这个链路的设计目标不是覆盖全部 13 个场景，而是先验证：

- 多模型 cost CSV 能被 `CostModel` 正确分场景读取
- 多用户 trace 能同时访问不同模型
- `edge_gs_runtime` 能完整消费这两份输入并跑通

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
