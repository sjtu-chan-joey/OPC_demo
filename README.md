# OPC Competition Demo: LLM Battery Closed-Loop Control

本文件夹是用于 OPC 比赛展示的精简交付包。它只包含：

- 已训练的 LoRA adapter：`adapters/qwen3_battery_lora/`
- 实时闭环控制 dashboard 前端：`dashboard/`
- 为了运行前端演示所需的最小 Python 后端与仿真代码：`battery_llm_control/`

本包没有打包本地 Qwen 基座模型，避免提交体积过大。评委如需运行真实 LLM 策略，请按下文下载基座模型到本地。

<div align="center">

https://github.com/sjtu-chan-joey/OPC_demo/blob/main/demo.mp4

*点击播放查看完整演示*

</div>


## 重要说明

这是一个测试模型和演示系统，用于展示“大模型参与电池充放电闭环控制”的原型流程。当前 adapter 只经过小规模仿真数据微调，性能、泛化能力、稳定性和工程安全性均不保证，不能直接用于真实电池或工业设备控制。

dashboard 中的仿真模型是轻量等效模型，用于比赛演示和策略对比，不等同于真实电化学模型或量产 BMS 控制器。

## 目录结构

```text
OPC_Competition_Demo/
  README.md
  requirements.txt
  adapters/
    qwen3_battery_lora/
      adapter_config.json
      adapter_model.safetensors
  battery_llm_control/
    dashboard.py
    policy.py
    protocol_space.py
    sim_core.py
  dashboard/
    index.html
    app.js
    styles.css
```

## 基座模型下载

本 adapter 对应的基座模型是：

```text
Qwen/Qwen3-4B-Instruct-2507
```

Hugging Face 页面：

```text
https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
```

推荐下载到以下本地目录：

```text
models/Qwen3-4B-Instruct-2507
```

可使用 Hugging Face CLI：

```powershell
pip install -U huggingface_hub
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 --local-dir models/Qwen3-4B-Instruct-2507
```

或使用 Git LFS：

```powershell
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 models/Qwen3-4B-Instruct-2507
```

下载后，adapter 配置会默认从 `./models/Qwen3-4B-Instruct-2507` 读取基座模型。

## 环境安装

建议使用 Python 3.10 到 3.12。

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

如需运行真实 LLM 推理，请确保安装 CUDA 版 PyTorch，并且：

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

输出应为：

```text
True
```

如果没有可用 GPU，dashboard 会自动回退到快速 LLM 代理策略，用于展示闭环流程，但这不是实际 LoRA 模型输出。

## 启动演示

在 `OPC_Competition_Demo` 目录下运行：

```powershell
python .\battery_llm_control\dashboard.py
```

浏览器打开：

```text
http://127.0.0.1:8765
```

页面中设置：

- `目标 Cycle`：演示停止的等效循环目标。
- `决策间隔 Cycle`：每推进多少等效 Cycle 后让 LLM 或策略重新决策。
- `播放速度`：前端请求后端实时步进的速度。

点击“启动”后，dashboard 会实时执行以下过程：

1. 仿真模型推进 1 分钟电池状态。
2. 到达决策 Cycle 边界时，LLM 读取最新 SOC、SOH、温度、内阻等状态。
3. LLM 输出下一段充电、放电或静置策略。
4. 同时运行 Oracle、保守策略、固定 1C 基线等对比策略。
5. 前端实时更新 SOC、SOH、容量、温度、电压、电流、Cycle 和多策略曲线。

## 比赛展示建议

推荐演示参数：

```text
目标 Cycle: 1.0
决策间隔 Cycle: 0.10
初始 SOC: 0.35
初始 SOH: 0.94
环境温度: 35 C
初始内阻: 58 mΩ
```

如果现场 GPU 环境不足，可以直接演示快速 LLM 代理模式，仍可展示实时闭环控制、策略对比和指标变化过程。

## 再次声明

该模型为测试模型，训练数据规模和验证范围有限，性能不保证。演示结果只能说明系统原型可运行，不能作为真实电池控制、安全评估或工程部署依据。
