# Per-View g(q) Pipeline

这个目录整理了当前推荐使用的 per-view `g(q)` 流水线，用来把已经训练完成的 ES 模型继续加工成：

- 多档剪枝模型 `point_cloud_pruned_{q}p.ply`
- 每视点渲染计时 CSV
- 每视点 `g(q)` CSV
- 每视点拟合参数 `fit_params_per_view.csv`
- 最终 `simulation_cost_field_{scene}.csv`

这条链是从两部分代码整合出来的：

- viewport 生成：`es/regression/00 generate_vp_matrix_20260105.py`
- 上游剪枝：`es/20251117_get_g(q)_fun/01 gen_pruned_model.py`
- 中下游计时与回归：`es/regression/01 render_by_q.py`、`02 process_csv.py`、`03 g_q_regression.py`

同时，这里已经内置了 cost-field 合并逻辑，不再依赖外部 `cleanAndMergeData.py`。

## 适用场景

推荐在下面这种流程里使用：

`训练完成的 ES 模型 -> 剪枝 -> 计时 -> 计算 g(q) -> 拟合参数 -> 生成 cost-field -> 喂给 edge_gs_runtime`

如果你的目标是给 [`edge_gs_runtime`](\/root/3dgs-streaming/mini-splatting/edge_gs_runtime) 提供 `cost CSV`，这套目录就是现在建议使用的正式链路。

## 目录中的脚本

0. [`00_generate_vp_matrix_20260105.py`](\/root/3dgs-streaming/mini-splatting/es/per_view_gq_pipeline/00_generate_vp_matrix_20260105.py)
   - 生成 `viewports_20260105` 风格的 viewport JSON
   - 默认生成 `32 x 32 x 32 = 32768` 个视点
   - 可以独立运行，也可以由 `run_pipeline.py --generate_viewports` 触发

1. [`01_gen_pruned_model.py`](\/root/3dgs-streaming/mini-splatting/es/per_view_gq_pipeline/01_gen_pruned_model.py)
   - 输入训练好的模型目录和 viewport JSON
   - 基于多视点 importance 生成 `point_cloud_pruned_{q}p.ply`
   - 会在模型目录下缓存 `importance_cache_<viewport>.pt`
   - 同时记录一份剪枝耗时统计 `prune_times.csv`

2. [`02_render_by_q.py`](\/root/3dgs-streaming/mini-splatting/es/per_view_gq_pipeline/02_render_by_q.py)
   - 对每个剪枝模型、每个视点、每次重复渲染计时
   - 生成 `render_times_{scene}.csv`

3. [`03_process_csv.py`](\/root/3dgs-streaming/mini-splatting/es/per_view_gq_pipeline/03_process_csv.py)
   - 从 `render_times_{scene}.csv` 计算每条记录的 `g_q`
   - 生成 `render_times_{scene}_gq.csv`

4. [`04_g_q_regression.py`](\/root/3dgs-streaming/mini-splatting/es/per_view_gq_pipeline/04_g_q_regression.py)
   - 对每个模型、每个视点分别拟合 `Power / Exponential / Polynomial`
   - 生成 `fit_params_per_view.csv`
   - 额外输出随机视点的拟合图

5. [`05_build_cost_fields.py`](\/root/3dgs-streaming/mini-splatting/es/per_view_gq_pipeline/05_build_cost_fields.py)
   - 直接在脚本内部完成 cost-field 合并
   - 合并 `render_times_{scene}.csv + fit_params_per_view.csv + viewport position`
   - 生成 `simulation_cost_field_{scene}.csv`
   - 导出的列已经兼容 `edge_gs_runtime` 的 `CostModel`

6. [`run_pipeline.py`](\/root/3dgs-streaming/mini-splatting/es/per_view_gq_pipeline/run_pipeline.py)
   - 一键按顺序执行上述步骤
   - 支持跳过某些步骤
   - 支持可选的第 0 步自动生成 viewport JSON

## 输入要求

### 1. 模型目录结构

`--models_root` 需要指向一个“每个场景一个子目录”的根目录，例如：

```text
eval_sr_all_scenes/
  bicycle/
    point_cloud/
      iteration_30000/
        point_cloud.ply
  room/
    point_cloud/
      iteration_30000/
        point_cloud.ply
  truck/
    point_cloud/
      iteration_30000/
        point_cloud.ply
```

这套流水线会在每个场景目录下继续写入后处理产物，不会覆盖 `point_cloud/iteration_*`。

### 2. viewport JSON

`--viewports_json` 需要是一个 JSON 列表，每个元素至少包含这些字段：

```json
{
  "position": [x, y, z],
  "view_matrix": [...16 values...],
  "view_projection_matrix": [...16 values...],
  "resolution_x": 1920,
  "resolution_y": 1080,
  "fov_x": ...,
  "fov_y": ...,
  "z_near": ...,
  "z_far": ...
}
```

当前推荐直接使用：

- [`viewports_20260105.json`](\/root/3dgs-streaming/mini-splatting/es/viewports_20260105.json)

如果你想从头生成，也可以用新目录里的第 0 步：

```bash
cd /root/3dgs-streaming/mini-splatting/es/per_view_gq_pipeline
/root/miniforge3/envs/mini_splatting/bin/python 00_generate_vp_matrix_20260105.py \
  --output_json /root/3dgs-streaming/mini-splatting/es/viewports_20260105.json
```

### 3. CUDA 环境

这套链路的前两步会实际调用渲染相关 CUDA 代码：

- `01_gen_pruned_model.py`
- `02_render_by_q.py`

因此需要：

- `torch.cuda.is_available() == True`
- ES 相关扩展可正常导入

## 一键运行

### 典型用法

```bash
cd /root/3dgs-streaming/mini-splatting/es/per_view_gq_pipeline
/root/miniforge3/envs/mini_splatting/bin/python run_pipeline.py \
  --models_root /root/3dgs-streaming/mini-splatting/es/Training_SR/eval_sr_all_scenes \
  --viewports_json /root/3dgs-streaming/mini-splatting/es/viewports_20260105.json \
  --scenes bicycle room truck \
  --q_list 60 70 80 90 100 \
  --costfield_out_dir /root/3dgs-streaming/mini-splatting/es/systemSimulation/pilot_cost_fields
```

### 从第 0 步一起跑

如果你还没有 viewport JSON，可以让 `run_pipeline.py` 先自动生成：

```bash
cd /root/3dgs-streaming/mini-splatting/es/per_view_gq_pipeline
/root/miniforge3/envs/mini_splatting/bin/python run_pipeline.py \
  --models_root /root/3dgs-streaming/mini-splatting/es/Training_SR/eval_sr_all_scenes \
  --viewports_json /root/3dgs-streaming/mini-splatting/es/viewports_20260105.json \
  --generate_viewports \
  --scenes bicycle room truck \
  --q_list 60 70 80 90 100 \
  --costfield_out_dir /root/3dgs-streaming/mini-splatting/es/systemSimulation/pilot_cost_fields
```

### 只跑到回归，不生成 cost-field

```bash
/root/miniforge3/envs/mini_splatting/bin/python run_pipeline.py \
  --models_root /root/3dgs-streaming/mini-splatting/es/Training_SR/eval_sr_all_scenes \
  --viewports_json /root/3dgs-streaming/mini-splatting/es/viewports_20260105.json \
  --scenes bicycle room truck \
  --q_list 60 70 80 90 100 \
  --skip_costfield
```

### 从中间步骤继续跑

例如前两步已经完成，只想从 `process -> regression -> costfield` 继续：

```bash
/root/miniforge3/envs/mini_splatting/bin/python run_pipeline.py \
  --models_root /root/3dgs-streaming/mini-splatting/es/Training_SR/eval_sr_all_scenes \
  --viewports_json /root/3dgs-streaming/mini-splatting/es/viewports_20260105.json \
  --scenes bicycle room truck \
  --costfield_out_dir /root/3dgs-streaming/mini-splatting/es/systemSimulation/pilot_cost_fields \
  --skip_prune \
  --skip_render
```

## 每一步的输出

### Step 0: 生成 viewport JSON

`00_generate_vp_matrix_20260105.py` 默认会生成：

- `32` 个半径层
- `32` 个 yaw
- `32` 个 pitch

总计：

- `32 x 32 x 32 = 32768` 个视点

默认输出文件通常是：

- `viewports_20260105.json`

#### 单个 viewport 的字段结构

```json
{
  "yaw": ...,
  "pitch": ...,
  "radius": ...,
  "position": [x, y, z],
  "train": 1,
  "shs_python": 0,
  "rot_scale_python": 0,
  "scaling_modifier": 1.0,
  "resolution_x": 1920,
  "resolution_y": 1080,
  "fov_y": ...,
  "fov_x": ...,
  "z_far": 1100,
  "z_near": 0.009,
  "keep_alive": 1,
  "view_matrix": [...16 values...],
  "view_projection_matrix": [...16 values...]
}
```

### Step 1: 剪枝模型

`01_gen_pruned_model.py` 会在每个场景目录下生成：

- `importance_cache_<viewport_stem>.pt`
- `point_cloud_pruned_60.0p.ply`
- `point_cloud_pruned_70.0p.ply`
- `point_cloud_pruned_80.0p.ply`
- `point_cloud_pruned_90.0p.ply`
- `point_cloud_pruned_100.0p.ply`

如果 `--q_list` 不同，对应文件名也会变化。

它还会在当前执行目录生成一份汇总 CSV：

- `prune_times.csv`

#### `prune_times.csv` 表头

```csv
model_name,q,prune_time_s,remaining_points
```

#### 字段含义

- `model_name`: 场景名
- `q`: 保留比例，单位是百分数，例如 `60` 表示保留 60%
- `prune_time_s`: 本次剪枝耗时，单位秒
- `remaining_points`: 剪枝后剩余高斯点数

### Step 2: 每视点渲染计时

`02_render_by_q.py` 会在每个场景目录下生成：

- `render_times_{scene}.csv`

例如：

- `bicycle/render_times_bicycle.csv`

#### `render_times_{scene}.csv` 表头

```csv
model_name,view_index,q,repeat_idx,render_time_s,remaining_points
```

#### 字段含义

- `model_name`: 场景名
- `view_index`: 视点索引，对应 `viewports_json` 里的第几个视点
- `q`: 当前使用的剪枝保留率
- `repeat_idx`: 第几次重复渲染，用于计时统计
- `render_time_s`: 单次渲染耗时，单位秒
- `remaining_points`: 当前这个 pruned 模型的点数

### Step 3: 计算每视点 g(q)

`03_process_csv.py` 会在每个场景目录下生成：

- `render_times_{scene}_gq.csv`

它会自动把 `q=max(q_list)` 当作 baseline，然后计算：

`g_q = render_time_s / base_time`

并过滤掉：

- `q == baseline`
- `g_q >= 1.0`

#### `render_times_{scene}_gq.csv` 表头

输出列是在 `render_times_{scene}.csv` 基础上增加两列：

```csv
model_name,view_index,q,repeat_idx,render_time_s,remaining_points,base_time,g_q
```

#### 字段含义

- `base_time`: 该 `view_index` 在 baseline 质量下的耗时
- `g_q`: 当前质量相对 baseline 的时间比例

### Step 4: 拟合参数

`04_g_q_regression.py` 会在每个场景目录下生成：

- `fit_params_per_view.csv`
- `figures/{scene}/view_*.png`

其中 `fit_params_per_view.csv` 是后续生成 cost-field 的关键输入。

#### `fit_params_per_view.csv` 表头

```csv
model_name,view_index,method,param1,param2,param3,r2
```

#### 字段含义

- `model_name`: 场景名
- `view_index`: 视点索引
- `method`: 拟合方法，当前会有：
  - `Power`
  - `Exponential`
  - `Polynomial`
- `param1,param2,param3`: 对应拟合函数参数
- `r2`: 当前方法在该视点上的拟合优度

#### 当前三种函数

- `Power`: `g(q) = a * q^b`
- `Exponential`: `g(q) = exp(a * q + b)`
- `Polynomial`: `g(q) = a * q^2 + b * q + c`

注意：

- `q` 在拟合时会先除以 `100`，即用 `0.6, 0.7, ...` 这样的比例值进入回归
- `Polynomial` 是当前 `05_build_cost_fields.py` 默认优先使用的方法

### Step 5: 生成 cost-field

`05_build_cost_fields.py` 会在 `--costfield_out_dir` 下生成：

- `simulation_cost_field_bicycle.csv`
- `simulation_cost_field_room.csv`
- `simulation_cost_field_truck.csv`

#### `simulation_cost_field_{scene}.csv` 表头

当前输出会包含两套列名，以同时兼容实验链和 runtime：

```csv
model_name,Model,view_index,x,y,z,base_cost_mean,base_cost_std,method,param1,param2,param3,Param_a,Param_b,Param_c,r2
```

#### 字段含义

- `model_name`: 场景名，实验链原始命名
- `Model`: 场景名，给 `edge_gs_runtime` 用的兼容列
- `view_index`: 视点索引
- `x,y,z`: 对应视点位置
- `base_cost_mean`: baseline 质量下的平均渲染耗时
- `base_cost_std`: baseline 质量下的渲染耗时标准差
- `method`: 当前选用的拟合方法，默认是 `Polynomial`
- `param1,param2,param3`: 原始拟合参数列
- `Param_a,Param_b,Param_c`: runtime 兼容参数列
- `r2`: 当前选中拟合方法的拟合优度

## 与 edge_gs_runtime 的关系

### 可以直接用的部分

这套 pipeline 最终产出的：

- `simulation_cost_field_{scene}.csv`

现在已经可以直接作为 [`edge_gs_runtime`](\/root/3dgs-streaming/mini-splatting/edge_gs_runtime) 的 `cost CSV` 输入。

原因是 [`CostModel`](\/root/3dgs-streaming/mini-splatting/edge_gs_runtime/control/cost_model.py) 会识别这些逻辑列：

- 场景：`Model`
- 位置：`x`, `y`, `z`
- baseline cost：`base_cost_mean`
- 拟合参数：`Param_a`, `Param_b`, `Param_c`

### 还不能替代的部分

这套目录不会生成 runtime 需要的 `trace CSV`。

也就是说，要真正跑 [`run_stage_a.py`](\/root/3dgs-streaming/mini-splatting/edge_gs_runtime/apps/run_stage_a.py)，还需要另一份 trace，至少要有：

```csv
Frame_ID,User_ID,R,D,Pred_Cost,Param_a,Param_b,Param_c
```

更理想的 trace 还应带上：

```csv
Model,x,y,z
```

这样 runtime 才能按场景和空间位置去查 cost-field。

## 当前默认约定

- baseline 质量：`q = max(q_list)`，通常是 `100`
- warmup 丢弃：`repeat_idx < 3` 会在 build cost-field 时被丢掉
- cost-field 默认选用回归方法：`Polynomial`

## 常见目录产物示意

以 `bicycle` 为例，场景目录最终会长成这样：

```text
bicycle/
  point_cloud/
    iteration_30000/
      point_cloud.ply
  importance_cache_viewports_20260105.pt
  point_cloud_pruned_60.0p.ply
  point_cloud_pruned_70.0p.ply
  point_cloud_pruned_80.0p.ply
  point_cloud_pruned_90.0p.ply
  point_cloud_pruned_100.0p.ply
  render_times_bicycle.csv
  render_times_bicycle_gq.csv
  fit_params_per_view.csv
  figures/
    bicycle/
      view_123.png
      view_456.png
      ...
```

而 cost-field 会集中输出到：

```text
pilot_cost_fields/
  simulation_cost_field_bicycle.csv
  simulation_cost_field_room.csv
  simulation_cost_field_truck.csv
```

## 备注

- 这条链的输出组织方式是“每个模型目录内各自产物”，便于后续定位问题
- 如果你后续想扩展到全量 13 个场景，直接改 `--scenes` 即可
- 如果你想接 [`edge_gs_runtime`](\/root/3dgs-streaming/mini-splatting/edge_gs_runtime)，建议把这套 cost-field 产物和 trace 生成链再包成一层统一脚本
