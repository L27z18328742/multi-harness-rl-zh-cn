## 框架如何运作

框架由三部分组成。OpenEnv 是 harness、环境和训练器共同连接的接口，其采集代理负责记录 token。Harbor 提供任务和沙箱。TRL 根据代理记录的数据训练。本章先沿着一次 rollout 贯穿整个系统，再分别介绍每一部分。

### OpenEnv

OpenEnv 是 RL 环境的标准接口 [@openenv2026]。它通过客户端与服务器之间的通信，提供类似 [Gymnasium](https://gymnasium.farama.org) 的 `reset`、`step`、`state`，采用有类型的动作和观测，打包为 Docker，并可发布到 Hub。它最初由 Meta PyTorch 与 Hugging Face 合作开发，现在位于 [`huggingface/OpenEnv`](https://github.com/huggingface/OpenEnv)，由十二家组织组成的委员会指导，采用 BSD-3-Clause 许可。

本文中，OpenEnv 是 harness、环境和训练器共同连接的共享接口。它不负责训练，也不定义奖励。**采集代理**从任何调用模型 API 的智能体中，记录 RL 训练所需的信息。**Harbor 集成** [@openenv2026harbor] 将 Harbor 的容器化任务作为 OpenEnv 环境提供服务，并在每次 rollout 中选择 harness 和沙箱。

通过有类型的 `TrainingTrace` API 训练，需要 [OpenEnv 0.7.0](https://github.com/huggingface/OpenEnv/releases/tag/v0.7.0) 或更新版本；Harbor 可选依赖需要 **Python 3.12**。在 Python 3.10 或 3.11 下，`openenv[harbor]` 不会安装 Harbor 依赖。`harbor_env` 包也来自仓库的 `envs/` 目录，而非 OpenEnv wheel。若使用 main 分支检出版本：

<Code n="0" />

训练器、推理服务器的依赖，以及双 GPU 启动命令，见 TRL 的 [Harbor 示例](https://github.com/huggingface/trl/tree/main/examples/async_grpo_harbor)。

### 一次 rollout 的端到端流程

在深入各部分前，先看完整路径。下图以 [SmolDataEnvs](https://huggingface.co/collections/FineEnvs/smoldataenvs) 中的一个三轮 rollout 为例，展示它如何经过系统各部分。这是我们根据真实 Kaggle notebook 构建的数据分析任务集。切换 harness，只会改变 Harbor 将其接到代理的方式、模型请求与回复的格式，以及工具名称。

<Embed n="0" title="一次 rollout 如何经过 OpenEnv" desc="从训练器发出请求到权重更新，逐步展示一个三轮 rollout。" />

训练器和推理服务器 vLLM [@kwon2023vllm] 各使用一张 GPU。OpenEnv 服务器可以在它们附近运行，也可以放在 Hugging Face Space 中；每个 rollout 都有独立沙箱。实验硬件见[训练设置](#setup)。沙箱里的 harness 从不直接与 vLLM 或训练器通信，只通过采集代理访问模型，它为此持有的唯一凭据是一个会话 ID。

### 采集代理

harness 通过 base URL 和 API key 找到代理，方式与访问普通模型提供商相同。这个 API key 是为当前 rollout 创建的采集会话 ID；这使一个端口上的单个代理能够服务全部并发 rollout。携带未注册 key 的调用会收到 401。

<Embed n="1" title="采集代理" desc="一次模型调用经过采集代理的逐步流程。" />

如 [harness 表格](#harness-ecosystem-matrix)所示，harness 使用四种模型 API。代理依次根据请求路径、请求头、请求体结构判断所用格式。它使用从 NVIDIA Polar 网关 [@xu2026polar] 引入项目的转换器，将请求转换为 chat completions，再按调用方自己的格式回放回答。

<Embed n="2" title="四种 API 方言，统一为一种格式" desc="编码智能体的四种请求格式，以及训练器看到的统一形式。" />

对推理引擎的调用始终采用非流式。代理等待完整补全，保存后，如果 harness 要求流式，就以流的形式向它回放。除了首个 token 到达更晚，harness 察觉不到区别，代理也无需解析不完整的增量片段。

#### 记录哪些信息

每次调用时，代理都向引擎索取提示的 token ID、采样得到的 token ID，以及每个采样 token 的 logprob。它还会从完整分布采样，不采用 `top_p` 或 `top_k` 截断。截断分布会使实际采样偏离策略自身的分布，这是 RL 中已知的熵坍缩途径；vLLM 又是在截断后计算 processed logprobs，因此它们也与策略不一致。在我们的实验中，将 `top_p` 调整为 1.0，使重要性比率从 0.985–0.993 改善到 0.9984–0.9999。在 vLLM 上，需要以下两个服务器参数：

<Code n="1" />

[SGLang](https://github.com/sgl-project/sglang) 也可以使用。[¹](#sglang-note)

<p id="sglang-note" class="footnote">¹ SGLang 从 v0.5.17 起，会为非流式请求返回采样 token ID；v0.5.20 又将其加入流式请求。代理对引擎不使用流式，因此 v0.5.17 及更新版本应当可用，不过我们只测试过 SGLang main 分支构建版本。</p>

代理不会盲目信任这些信息。第一次遇到一个引擎时，它会发送探测请求并评定结果，也会检查 logprobs 是 raw 还是 processed。低于 `tokens` 级别的引擎仍然可以服务 rollout，但会被标记为仅供评估；请求它们的训练数据会触发错误。OpenAI、Anthropic、[Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers/index) 等托管 API 属于这一情况，因此可以用来评估 harness，却不能通过它们训练。

<Embed n="3" title="三个采集等级" desc="其中只有一个等级能够用于训练。" />

#### 从调用构建训练序列

harness 会重试、启动子智能体、压缩上下文，因此一个 rollout 很少是一段整洁的对话。代理将每次模型调用存为一个节点，再将它连接到更早的某次调用：这次早期调用的提示与补全，必须构成当前提示最长的精确 token 前缀。重试会表现为没有继续延伸的兄弟节点。子智能体有自己的系统提示，所以其第一次调用不会延伸任何已有节点，而会建立新根；压缩后的上下文也一样。

<Embed n="4" title="rollout 图" desc="把轨迹表示成模型调用图，并展示可训练的分支。" />

每条从根到叶的路径，都会成为一个训练序列。上下文 token 的损失掩码为 0，采样 token 为 1。如果某轮 logprobs 缺失或没有对齐，这一轮仍作为上下文留在序列中，但绝不作为训练目标。这些检查在调用到达时立即执行，而非等到导出时才做，因为代理仍知道具体轮次时，最容易诊断对齐错误。

<Embed n="5" title="一次 rollout 返回什么" desc="逐轮查看一个 rollout 交回的信息。" />

#### 采集你自己的 harness

代理没有任何特定依赖于 Harbor 的部分，所以调用这四种 API 之一的任意程序，都能被采集。启动代理，通过 `POST /sessions` 创建会话，将代理 URL 作为智能体的 base URL，将会话 ID 作为其 API key，再通过 `GET /sessions/{id}/rollout` 读取 rollout。

<Code n="2" />

代理默认绑定 localhost。绑定其他接口时，需要一个 admin key，用于保护创建和读取会话的路由。

代理是单进程。在我们的实验里，大约 200 个并发会话时，健康检查开始得不到调度；到 320 个时进程崩溃。达到模型调用预算的 rollout，也由代理自己停止，因此这次最终回复不会进入训练数据。

### Harbor

[Harbor](https://harborframework.com/) 由 Terminal-Bench [@merrill2026terminalbench] 背后的团队开发，负责让智能体执行容器化任务，并保持任务、harness 和沙箱彼此独立。[0.22.0 版本](https://github.com/harbor-framework/harbor/releases/tag/v0.22.0) 为四十多种 agent harness 提供适配器，并支持 [26 种执行后端](https://github.com/harbor-framework/harbor/blob/v0.22.0/src/harbor/models/environment_type.py)，从本地 Docker 到 [Daytona](https://www.daytona.io)、[Modal](https://modal.com)、E2B。约八十个任务数据集已采用其格式，包括 Terminal-Bench 和 SWE-bench [@jimenez2024swebench]。任意 harness 都可以在任意后端尝试任意任务，因此一个任务集能够用于多种 harness 的训练；运行时，一条命令就能指定数据集、智能体和后端。Adithya 在 [《RL 编码环境 101：为什么需要 Harbor》](https://x.com/adithya_s_k/article/2054961319179420035) 中解释了它为何适合作为编码 RL 的抽象。

Harbor 任务是一个包含指令、环境和测试脚本的目录。下方任务来自本文所有实验使用的 SmolDataEnvs。问题源于 [jupyter-agent 数据集](https://huggingface.co/datasets/jupyter-agent/jupyter-agent-dataset)中的 Kaggle notebook，每个问题都配有自动评分器。选择一个任务和文件，就能以智能体或验证器看到的方式阅读它。

<Embed n="6" title="逐文件查看三个 SmolDataEnvs 任务" desc="一个 Harbor 任务的六个文件，未经修改地取自训练集，并展示奖励如何输出。任务文件保留原文。" />

任务本身不指定 harness、沙箱后端或训练器，因为这些都是运行时选择的。[Harbor 可视化工具](https://huggingface.co/spaces/HuggingFaceH4/harbor-visualiser) 以同样方式开放完整的 5,000 个训练任务。

<HarborVisualizer />

> **说明：** Harbor 将自己的沙箱后端称为 *environment*。本文中，*sandbox* 指智能体运行的隔离空间，*environment* 指 OpenEnv 服务器。

### 通过 OpenEnv 提供 Harbor 服务

Harbor 已经能让智能体执行任务并返回分数。训练器还需要策略生成的 token 及其概率。Harbor 的 [RL 文档](https://github.com/harbor-framework/harbor/blob/v0.22.0/docs/content/docs/training-workflows/rl.mdx) 列出了两种获取方式：在推理服务器处拦截 token，或者让 harness 在结果中返回它们。第二种方式要求 harness 将 token 放进结果元数据；文档指出，Harbor 自有 harness [Terminus 2](https://github.com/harbor-framework/harbor/blob/v0.22.0/docs/content/docs/agents/terminus-2.mdx) 的相关支持仍在添加。本集成采用第一种方式，通过采集代理实现，因此适用于任何调用模型 API 的 harness。

训练器通过 HTTP 与环境通信，而不是在自己的进程内运行 rollout。早先的进程内实现，会让第三方智能体内部的异常传播到训练循环；一个 rank 崩溃后，其他 rank 就会无限等待在 NCCL barrier 上。现在，异常不会跨越这条边界。失败的 rollout 会返回 `ok=False` 和 `reward=None`，训练器处理返回值即可，无需捕获异常。

#### 四条命令

| 命令 | 功能 |
|---|---|
| `openenv harbor info` | 报告本机能运行什么：引擎是否返回 token ID、哪些沙箱后端凭据可用、哪些数据集能够解析及各自任务数、哪些 harness 已验证。不会启动任何服务。 |
| `openenv harbor rollout` | 不启动环境服务器，直接运行 rollout。如果这里成功、在 `serve` 下失败，问题就在服务层。 |
| `openenv harbor serve` | 环境服务器：提供 Task API、通过 MCP 暴露的 `run_rollout` 工具、网页界面，以及采集代理。 |
| `openenv harbor push` | 将同一服务器部署到 Hugging Face Space，配置放在 Space variables 中，凭据放在 secrets 中。 |

<Code n="3" />

`rollout` 必须指定 `--llm-url`，且没有默认值，因为过期端点可能产生看似正常、却不含 token ID 的 rollout。`serve` 可以不带这个参数启动，因为每个 rollout 自己指定引擎。这让一个部署既能服务训练——连接训练器的 vLLM——也能服务评估。这里的 `-n 5`（或 `--n-tasks 5`）从 `--task-index 0` 开始，将任务 0 至 4 各运行一次；它不是对一个任务采样五次尝试。E2B 示例与报告的实验一致，配套教程则使用 Daytona。每个 rollout 完成时会打印一行：

<Code n="4" />

<Embed n="7" title="从 serve 到 rollout" desc="哪些操作在启动时只发生一次，哪些操作在每个 rollout 中都会发生。" />

接下来两部分介绍各 harness 如何连接代理，以及如何检查奖励和采集结果。完整参考见 [OpenEnv 文档的 Harbor 页面](https://huggingface.co/docs/openenv/environments/harbor)。

<details>
<summary>各 harness 的接线方式、奖励选择与采集检查</summary>

#### 各 harness 如何接线

Harbor 已经知道如何安装和启动每种 harness。OpenEnv 只补充它们从哪里读取模型 URL 和 API key，并以每种 harness 对应一个表项的形式维护。因此，支持新 harness 只需增加表项，无需编写新环境。十种 harness 已在 SmolDataEnvs 测试集的 16,000 次 rollout 中，通过下述采集契约和轨迹检查。

<Embed n="8" title="各已验证 harness 如何访问代理" desc="每种 harness 从哪里获取代理 URL 和会话 ID。" />

多数 harness 读取环境变量，OpenEnv 只为智能体这次运行设置它们。因此，[Codex](https://github.com/openai/codex) 通过 `OPENAI_API_KEY` 收到的会话 ID，绝不会传到可能需要自身真实 key 的验证器。[Claude Code](https://code.claude.com/docs/en/overview) 和 [Gemini CLI](https://github.com/google-gemini/gemini-cli) 还会读取服务器进程里的变量，因此那里也要设置。[OpenCode](https://opencode.ai) 和 [Pi](https://pi.dev) 使用写入沙箱的配置文件；Terminus 2 在主机运行，直接接收 URL 和 key。

#### 选择奖励并检查采集

Harbor 验证器可以返回多个具名分数，而 GRPO [@shao2024deepseekmath] 要求每个 rollout 只有一个数值。如果只有一个分数，集成直接使用它；否则优先使用名为 `reward` 的分数，再否则要求调用方选择。集成绝不自行组合分数，因为无论选择怎样的权重，都会成为训练目标。早期一次实验包含“只要提交任何内容就加 0.2”的奖励项，策略学会立即提交；留出集准确率从 0.740 跌至 0.178，而训练奖励看起来仍然正常。如果验证器根本未运行，奖励会始终以 `None` 传到训练器，因此沙箱崩溃不会被当作答案错误。

代理自身的检查只能查看它采集到的数据。为了独立验证，还会逐调用比较每个 rollout 与 Harbor 自己写出的 [ATIF](https://docs.harborframework.com/core-concepts/agents/atif) 轨迹文件：轮数、每次调用的补全 token 数，以及哪些调用算作智能体步骤。任何不一致都会让 rollout 失败。

<Code n="5" />

这项检查发现，一种 harness 发送了空 tools 数组，导致推理服务器返回 400，rollout 被提前截断，却留下了一个看似结构正确的图。现在代理会在转发前移除空 tools 数组。

</details>

#### 部署并对其训练

在单台机器上，`serve` 监听两个端口：一个用于训练器和网页界面，一个用于采集代理。沙箱通常在其他地方运行，因此通过隧道让沙箱访问代理，使用 `--expose` 设置，默认由 [Gradio](https://www.gradio.app) 提供。当服务器自身运行在 Hugging Face Space 上时，只有一个端口，因此代理挂载到 `/capture`。这个 Space 必须公开，因为沙箱中的智能体不能发送私有 Space 要求的认证头。安全性来自代理只响应已注册会话 ID，且创建会话需要 admin key。沙箱自身仍位于 E2B 或其他 Harbor 后端，每个 rollout 一个。

训练前，应提高服务器并发限制 `max_concurrent_envs`。它限制单个服务器同时运行多少 rollout；默认值四，小于我们每个 GRPO 组里的八个 rollout。超过限制时，`serve` 会以 `CAPACITY_REACHED` 拒绝新 rollout。

OpenEnv 的 `HarborSessionFactory` 生成符合 TRL `HarnessRolloutWorker` 预期的会话，适用于智能体掌握循环的情况。该 worker 通过 [TRL #6947](https://github.com/huggingface/trl/pull/6947) 合并，位于 `trl.experimental`。截至 2026 年 10 月 1 日，它已在 main 分支中，但尚未进入最新发布版 v1.14.1；下一版本发布前，需要从 main 安装 TRL：

<Code n="6" />

每个会话请求一个 rollout，等待完成，再通过 `fetch_training_trace()` 返回有类型的 `TrainingTrace`，其中包含 token ID、行为策略 logprobs 和损失掩码。TRL 直接消费这个对象。给定准备好的任务数据集和模型 tokenizer，worker 的接线代码如下：

<Code n="7" />

将这个 worker 作为 `rollout_worker` 传给 `AsyncGRPOTrainer`。worker 在生成前使用 `sampling=` 调用工厂，让采集策略与训练保持一致。`harness_adapter=None` 让 harness 运行自己的循环。这个最小回调只转发正确性奖励；[训练设置](#what-we-rewarded)还加入了工具效率加分。[完整 TRL 示例](https://github.com/huggingface/trl/blob/main/examples/async_grpo_harbor/async_grpo_harbor.py) 还展示了任务发现、`AsyncGRPOConfig`、训练器和 checkpoint 保存。

我们的实验中，同一个 GRPO 组内所有 rollout 都使用相同 harness，因此 advantage 比较的是动作，而非 harness。每个 rollout 使用的采样温度，也与训练器重新计算 logprobs 时使用的温度一致。

可以尝试 [SmolDataEnvs Harbor Space](https://huggingface.co/spaces/FineEnvs/smoldataenv-multi-harness-harbor)：选择任务、连接模型、运行 harness，再检查 rollout 与轨迹。[训练教程](https://github.com/adithya-s-k/FineEnvs/tree/main/05-multi-harness-rl) 包含同样的环境源码，以及用于 HF Jobs 或本地集群的命令。
