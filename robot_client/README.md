# NZ100 GR00T Robot Client

这个目录用于部署在机器人电脑上，连接 GPU 电脑上的 GR00T policy server。

运行链路：

```text
机器人电脑采集 top image + wrist_left image + left/right joints + left/right gripper
-> ZMQ 发给 GR00T server
-> 接收 action chunk
-> 机器人电脑按控制频率执行动作
```

## GPU 电脑启动 GR00T Server

在 GR00T 仓库根目录启动：

```bash
python gr00t/eval/run_gr00t_server.py \
  --model-path /path/to/nz100/checkpoint \
  --embodiment-tag NZ100 \
  --host 0.0.0.0 \
  --port 5555
```

## 机器人电脑启动 Client

先安装客户端依赖，并确保机器人电脑也能 import 当前 GR00T 仓库里的 `gr00t`：

```bash
pip install -r NZ100/robot_client/requirements.txt
pip install -e .
```

mock 测试不会控制真实机器人：

```bash
PYTHONPATH=NZ100 python -m robot_client.main \
  --config NZ100/robot_client/configs/nz100_client.yaml \
  --mock \
  --once
```

真实运行：

```bash
source /home/f/ysrobot_ws2/common/install/setup.bash
PYTHONPATH=NZ100 python -m robot_client.main \
  --config NZ100/robot_client/configs/nz100_client.yaml
```

## GR00T Observation 格式

client 发送给 GR00T server 的 observation 是：

```python
{
    "video": {
        "top": top_image[None, None],
        "wrist_left": wrist_left_image[None, None],
    },
    "state": {
        "left_arm": left_joints[None, None],
        "left_gripper": [[[left_gripper]]],
        "right_arm": right_joints[None, None],
        "right_gripper": [[[right_gripper]]],
    },
    "language": {
        "annotation.human.task_description": [[language_instruction]],
    },
}
```

GR00T 返回分组 action 后，client 会拼成机器人执行用的 16 维：

```text
0:7    左臂 7 个关节
7      左夹爪，PLC 语义 1=开，2=关
8:15   右臂 7 个关节
15     右夹爪，PLC 语义 1=开，2=关
```

当前 NZ100 action horizon 是 40，配置里的 `open_loop_horizon` 默认也是 40。

## 配置

主要配置文件：

```text
NZ100/robot_client/configs/nz100_client.yaml
```

常用字段：

```yaml
policy_host: 192.168.168.150
policy_port: 5555
control_fps: 30
open_loop_horizon: 40
language_instruction: Open the blue cylindrical package, and close the package.
execution_mode: async_queue
```

可选执行模式：

```yaml
execution_mode: sync_chunk
execution_mode: async_queue
execution_mode: rtc_guidance
```

`sync_chunk` 是普通 chunk 推理；`async_queue` 会在本地维护动作队列并后台预取下一段 chunk；`rtc_guidance` 会把上一段剩余 chunk 传给 GR00T server。`rtc_guidance_weight: 0` 时使用 GR00T 原生 RTC/inpainting；大于 0 时才启用 OpenPI-style simple guidance。`rtc_use_vjp: true` 暂不支持。

真实 ROS2 话题配置也在同一个 YAML 中，包括相机、关节状态、左右臂轨迹和双夹爪 Modbus 话题。
