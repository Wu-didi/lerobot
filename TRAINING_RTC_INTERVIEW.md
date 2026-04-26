# Training RTC Notes

## 1. 这次我做了什么

我在 `pi05` 这条策略链里加了一套 **training-time RTC** 机制。它的目标不是只在推理时做修正，而是把“真实部署里 action chunk 会重叠、会有延迟”这个事实，直接提前注入训练过程。

一句话版本：

> 我把原来只在推理时处理的 chunk overlap / delay 问题，改成了训练时就让模型见到“前缀动作已经确定、后缀动作还需要去噪”的场景，这样模型在真实 RTC 推理时更自然，不需要完全依赖额外 guidance。

## 清晰版小结

如果你看完整篇之前，只想先抓住 `training RTC` 最核心的流程，可以先看这一段。

### 它到底在做什么

`training RTC` 的本质不是给模型加一个新的大模块，而是**改变训练样本的组织方式**：

- 以前：把整个 action chunk 都当成“要预测的目标”
- 现在：把 chunk 前面一小段当成“已经确定的前缀条件”，只让模型去补全后面的部分

一句最直白的话：

> 普通训练是在教模型“从头写完整段动作”，training RTC 是在教模型“前面这段我已经写好了，你接着往后写”。


### 为什么要这么做

因为训练时模型的使用方式，和真实部署时模型的使用方式，并不完全一样。

假设：

- `chunk_size = 8`
- `n_action_steps = 4`

模型第一次输出：

`a0 a1 a2 a3 a4 a5 a6 a7`

但机器人通常不会把 8 步全执行完才再问模型，而是先执行前 4 步：

`a0 a1 a2 a3`

这时上一块的后半段：

`a4 a5 a6 a7`

在下一次推理时，其实天然就变成了“当前 chunk 的前缀候选”。  
也就是说，真实部署里模型更常面对的不是“整块从零开始生成”，而是：

- 前面几步已经基本确定
- 后面几步才需要继续补全

这就是 `training RTC` 想解决的问题。


### 训练时具体怎么做

训练逻辑在 `PI05Pytorch.forward(...)` 里。它每次训练大概做 5 步：

1. 随机采样一个 delay `d`
   - `d` 表示“当前 chunk 前面有多少步已经是已知前缀”
   - 例如 `d=0` 就退化成普通训练，`d=3` 就表示前 3 个 token 是冻结前缀

2. 把 chunk 切成两部分
   - prefix：前 `d` 个 token，作为已知条件
   - suffix：后面的 token，作为真正需要预测的部分

3. 给 prefix 和 suffix 不同的时间语义
   - prefix 的时间强行设成 `0`
   - suffix 还是正常的 flow matching 时间 `t`

4. 构造训练输入 `x_t`
   - 对 prefix，因为 `t=0`，它直接变成 clean action
   - 对 suffix，还是正常 noisy interpolation

5. 只对 suffix 算 loss
   - prefix 参与 forward，但不参与 loss
   - 因为 prefix 现在是条件，不再是目标

所以训练时模型真正学到的是：

> 给你一个已经确定的动作前缀，你要把后面的动作继续补完。


### 推理时怎么对应起来

推理逻辑在 `PI05Policy.predict_action_chunk(...)` 和 `PI05Pytorch.sample_actions(...)` 里。

流程是：

1. 当前 chunk 预测完以后，把整块动作先保存起来
2. 机器人只执行前 `n_action_steps`
3. 剩下没执行的尾部，作为下一次 chunk 的 prefix
4. 下一次 denoising 时，把这段 prefix 每一步都硬替换回去
5. 同时把这些 prefix token 的时间设成 `0`
6. 模型只需要继续补全 suffix

所以推理时的真实语义就是：

> 当前 chunk 的前半段不是重新生成的，而是继承自上一块 chunk 没执行完的尾部；模型真正负责的是把后半段补全。


### 为什么这样是有效的

它有效，核心是下面 4 点。

1. 它减少了 train-test mismatch。
   训练时模型见到的任务，终于更像真实部署时的任务了。原来训练是“整块都预测”，真实推理却是“前缀已知、后缀补全”，这两者差得很远；training RTC 就是在缩小这个差距。

2. 它和 flow matching 原本的时间语义一致。
   在这类模型里，`t=0` 本来就表示 clean sample。把冻结前缀设成 `t=0`，非常自然地表达了“这部分已经确定，不需要去噪，只是条件”。

3. 它把训练信号集中到真正困难的部分。
   prefix 已经给定了，如果还在 prefix 上算 loss，模型只是在学重复已有答案。mask 掉 prefix loss 后，模型会把容量更多用在 suffix 的补全上。

4. 它更容易学到 chunk 边界的连续性。
   普通 chunk 预测容易在边界处跳变。training RTC 强制模型在已有前缀基础上续写，所以更容易得到连续、平滑的 chunk-to-chunk 行为。


### 你可以怎么记住它

把它记成下面两句话就够了：

- **训练时**：前面一小段动作当条件，后面一大段动作当目标
- **推理时**：上一块剩下来的尾巴当条件，当前块后半段由模型补全

如果面试官问一句“training RTC 到底是什么”，最稳的回答可以直接说：

> training RTC 就是把真实部署里的 chunk overlap 和 delay 问题前移到训练。训练时我会随机冻结 chunk 前面的几个动作 token，把它们当作已经确定的 prefix，并把这些 token 的时间设成 0，只对后面的 suffix 算 loss。这样模型推理时就能更自然地利用上一块 chunk 的剩余动作，减少 train-test gap，提高 chunk 边界的连续性。


## 2. 代码改动主要在哪

核心代码在这几个地方：

- `src/lerobot/policies/pi05/configuration_pi05.py`
- `src/lerobot/policies/pi05/modeling_pi05.py`

具体职责：

- `PI05Config`
  - 增加了 `training_rtc: bool`
  - 增加了 `simulated_delay: int`
  - 对 `simulated_delay` 做了范围校验

- `PI05Pytorch.forward(...)`
  - 训练时加入 per-token delay / per-token time
  - 把前缀 token 冻结成 clean action
  - 只对非冻结 token 计算 loss

- `PI05Pytorch.sample_actions(...)`
  - 推理时支持把上一块 chunk 的未执行尾部作为当前 chunk 的 frozen prefix
  - 在每一步 denoising 里硬替换前缀，并把这些 token 的时间设成 0

- `PI05Policy.predict_action_chunk(...)`
  - 负责从上一块 chunk 中提取 overlap 尾部
  - 组装 `training_rtc_prev_chunk` 和 `training_rtc_delay`
  - 把它们传进底层 `sample_actions(...)`

- `PI05Policy.forward(...)`
  - 读取 `_training_rtc_mask`
  - 对冻结前缀做 masked loss


## 3. 它解决的是什么问题

普通 action chunk policy 在训练时通常是假设：

- 当前 chunk 是完整独立预测的
- 每个 token 都同等需要学习
- 推理和执行之间没有时间错位

但真机实时执行不是这样：

- 上一块 chunk 往往只执行了一部分
- 下一块 chunk 生成时，前面几步其实已经被“历史动作”决定了
- 真正需要模型补全的是后半段

所以 training-time RTC 的核心目标是：

> 让模型在训练时就学会“给定一个已经确定的动作前缀，继续补全后面的动作后缀”。


## 4. 我是怎么把它加进去的

### 4.1 配置层

在 `PI05Config` 里加了两个参数：

- `training_rtc`
  - 是否启用 training-time RTC
- `simulated_delay`
  - 训练时最大模拟延迟 `K`
  - 实际 delay 从 `{0, 1, ..., K-1}` 中采样

设计含义：

- `training_rtc=False`
  - 完全走原始 pi05 训练逻辑
- `training_rtc=True`
  - 在训练时构造“冻结前缀 + 需要预测的后缀”


### 4.2 训练时怎么做

训练逻辑在 `PI05Pytorch.forward(...)`。

核心步骤是：

1. 采样一个 delay `d`
   - `d in {0, ..., K-1}`
   - 用指数衰减权重采样，小延迟概率更大

2. 构造前缀 mask
   - 前 `d` 个 token 视为 frozen prefix
   - 后面的 token 视为 active suffix

3. 构造 per-token time
   - frozen prefix 的时间设成 `0`
   - active suffix 的时间仍然是正常 flow matching 时间 `t`

4. 构造训练状态 `x_t`
   - 对冻结前缀：
     - `t = 0`
     - 所以 `x_t = actions`
     - 也就是 clean action
   - 对非冻结部分：
     - 仍然是正常的 flow matching 插值
     - `x_t = t * noise + (1 - t) * actions`

5. 只对非冻结 token 算 loss
   - 冻结前缀本来就是已知条件
   - 不应该再当成需要预测的目标

这一步的核心想法是：

> 训练时把“前缀是条件，后缀是预测目标”这个结构显式做出来。


### 4.3 为什么 frozen prefix 的时间要设成 0

因为这套模型是 flow matching 风格：

- `t = 0` 对应 clean action
- `t > 0` 对应不同程度的 noisy action

所以把冻结前缀设成 `t = 0`，就等价于告诉模型：

> 这部分不是要你去噪出来的，它已经是最终确定的动作条件。

这样做比“单纯把动作塞进去”更自然，因为它和模型原本的时间语义一致。


### 4.4 为什么 loss 要 mask 掉 frozen prefix

因为 frozen prefix 的角色已经变了，它不再是“需要预测的目标”，而是“给模型看的条件”。

如果不 mask：

- 模型会在已经给定的前缀上继续学 identity mapping
- 会稀释真正应该学习的后缀部分
- 训练信号会被无效前缀占掉一部分

所以这里保留：

- 前缀参与 forward
- 前缀不参与 loss


### 4.5 推理时怎么接进去

推理逻辑在 `PI05Policy.predict_action_chunk(...)` 和 `PI05Pytorch.sample_actions(...)`。

流程是：

1. 上一次 chunk 推理完成后，把整块动作存到 `self._prev_action_chunk`

2. 下一次推理时，计算 overlap
   - `overlap = chunk_size - n_action_steps`

3. 从上一块 chunk 中取“没执行完的尾巴”
   - 只保留下一块还会重叠到的那一段

4. 计算 `inference_delay`
   - `inference_delay = min(overlap, simulated_delay - 1)`

5. 把这段尾部传给底层模型：
   - `training_rtc_prev_chunk`
   - `training_rtc_delay`

6. 在 `sample_actions(...)` 的 denoising loop 里：
   - 每一步都把前 `delay` 个 token 的 `x_t` 强制替换成上一块剩余动作
   - 同时把这些 token 的时间设成 0

所以推理时的语义是：

> 当前 chunk 的前缀不是重新生成的，而是继承自上一块 chunk 的剩余动作；模型真正负责补全的是后面的部分。


## 5. 它和“传统 RTC”有什么区别

这点面试里很容易被问。

### 传统 RTC

传统 RTC 是 **inference-time** 技术。

特点：

- 训练目标本身不变
- 推理时根据前一块 leftover、推理延迟、execution horizon 做修正
- 依赖 `RTCProcessor`
- 更像“外部 guidance / correction”

### training-time RTC

training-time RTC 是 **training + inference 联动** 的做法。

特点：

- 训练时就把 delay / overlap 结构喂给模型
- 模型自己学会“prefix conditioning”
- 推理时不一定需要重型 guidance
- 更像“把 RTC 约束内化到模型行为里”

一句话区分：

> 传统 RTC 是推理时纠偏，training-time RTC 是训练时就让模型学会这种纠偏结构。


## 6. 为什么我这样设计

我这版设计的核心考虑有 4 个。

### 6.1 尽量少改原模型主干

我没有重写整套 pi05，而是在原有 flow matching 结构上做最小侵入改动：

- 配置层新增两个参数
- 训练前向增加 per-token delay/time
- 推理时加 prefix hard replacement
- 上层 loss 做 mask

这样更容易维护，也更容易和原始 OpenPI / LeRobot 逻辑对齐。


### 6.2 保持和 flow matching 时间语义一致

不是单独硬插一个“prefix 标志位”，而是直接利用已有的时间变量 `t`：

- frozen prefix: `t = 0`
- active suffix: `t = sampled t`

这样模型不用学一套完全新的接口，仍然在原来的 flow matching 语义里工作。


### 6.3 控制 train-test gap

推理时 `inference_delay` 被 cap 到 `simulated_delay - 1`，原因是：

- 训练时模型只见过 `{0, ..., simulated_delay - 1}` 这段 delay 分布
- 推理如果给超过训练分布的 delay，会增加 out-of-distribution 风险

所以这里明确做了分布对齐。


### 6.4 把收益集中在真实有效场景

training-time RTC 真正有意义的前提是：

- `chunk_size > n_action_steps`

否则没有 overlap，也就没有“前一块尾部作为当前前缀”这个问题。

所以这套机制天然更适合：

- 大 chunk
- 分步执行
- 推理延迟不可忽略


## 7. 面试时可以怎么回答

### 7.1 30 秒版本

> 我在 pi0.5 上加了一套 training-time RTC。核心想法是把真实部署里 chunk overlap 和推理延迟造成的“前缀已经确定、后缀需要补全”这个结构，提前放进训练。具体做法是在训练时随机采样 delay，把前几个 token 设为 frozen prefix，并把它们的时间设成 0，只对后面的 active token 计算 loss。这样模型推理时可以直接利用上一块 chunk 的尾部作为条件，减少对额外 RTC guidance 的依赖。


### 7.2 1 到 2 分钟版本

> 原始 chunk policy 在训练时默认每个 action token 都是同等预测目标，但真实 RTC 场景里不是这样。上一块 chunk 往往已经执行了一部分，下一块 chunk 的前缀其实已经被上一块剩余动作决定了。我做的 training-time RTC，就是把这种结构显式加入训练。  
>  
> 在实现上，我先在配置里加了 `training_rtc` 和 `simulated_delay`。训练时在 `PI05Pytorch.forward()` 里随机采样一个 delay，构造 prefix mask，然后把前缀 token 的时间设成 0，让它们对应 clean action；其余 token 还是原始的 flow matching 时间。这样 `x_t` 里前缀是冻结条件，后缀才是需要预测的部分。同时我会把 frozen prefix 从 loss 里 mask 掉，因为它已经是条件而不是目标。  
>  
> 推理时，我在 `predict_action_chunk()` 里保存上一块 chunk，并把未执行完的 overlap 尾部作为 `training_rtc_prev_chunk` 传给底层模型。`sample_actions()` 在每一步 denoising 里都会把这段前缀硬替换回去，并设置成时间 0。这样模型就能以一种训练时见过的方式使用 chunk overlap。  
>  
> 这个方案和传统 RTC 的区别在于，传统 RTC 更像 inference-time correction，而 training-time RTC 是把这种 prefix conditioning 能力直接学进模型里。


### 7.3 如果面试官问“你自己的贡献点是什么”

可以答：

> 我的贡献不是简单把一个开关接上，而是把 training-time RTC 从配置、训练前向、推理前向到 loss 归约这几层真正串起来了。重点包括 per-token delay 建模、frozen prefix 的时间语义设计、loss masking，以及把上一块 chunk 的 overlap 尾部接入下一块推理。


## 8. 面试官很可能会问的问题

### Q1. training-time RTC 和普通 RTC 的本质区别是什么？

答：

> 普通 RTC 主要是推理时修正，training-time RTC 是训练时就把这种延迟和 overlap 结构建模进去。前者更像外部 guidance，后者更像模型内化了 prefix conditioning 能力。


### Q2. 为什么要把 frozen prefix 的时间设成 0？

答：

> 因为在 flow matching 里 `t = 0` 对应 clean sample。把 frozen prefix 设成 0，等价于用模型原生语义表达“这部分已经确定，不需要去噪，只是条件”。


### Q3. 为什么要 mask 掉 frozen prefix 的 loss？

答：

> 因为 frozen prefix 已经作为条件给定了，不应该再把它当成预测目标，否则模型会浪费容量去学 identity mapping，削弱对真正需要补全的 suffix 的学习。


### Q4. 为什么 delay 不是均匀采样，而是指数衰减？

答：

> 因为真实系统里小延迟通常更常见，大延迟更少见。指数衰减能让模型更聚焦常见情况，同时保留对较大 delay 的一定覆盖。


### Q5. 为什么推理时要把 `inference_delay` 限制到 `simulated_delay - 1`？

答：

> 因为训练时模型只见过那段 delay 分布，推理如果超过训练分布，就会出现明显的 distribution shift，所以这里做了 cap。


### Q6. 什么时候 training-time RTC 才真正有收益？

答：

> 当 `chunk_size > n_action_steps` 时最有意义，因为这时两次 chunk 之间存在 overlap。没有 overlap，就没有“上一块尾部作为当前前缀”的问题，收益会很有限。


### Q7. 这套改动会不会影响原始训练路径？

答：

> 不会。`training_rtc=False` 时，代码直接退回原始 pi05 路径。也就是说这套逻辑是显式开关控制的，不会污染默认行为。


### Q8. 这套方法的代价是什么？

答：

> 主要代价是训练逻辑更复杂了，而且收益依赖真实部署场景。如果系统几乎没有推理延迟，或者 chunk 没有 overlap，那么这套机制的收益会下降。另外，如果 `simulated_delay` 设得不合理，也可能导致训练分布和推理分布不匹配。


### Q9. 你怎么验证它是对的？

答：

> 我主要从三层看。第一层是配置和前向逻辑的正确性，比如 delay 范围、mask 形状、训练和非训练路径的兼容性。第二层是推理行为是否符合预期，比如有 overlap 时是否真的会把上一块尾部当作当前前缀，没有 overlap 或没有 previous chunk 时是否退化成原始行为。第三层是和传统 RTC 路径的关系是否保持兼容，不会破坏原有 RTCProcessor 逻辑。

如果面试官继续追问“有没有单元测试”，可以诚实说：

> 当前仓库里 RTC 本身有比较完整的测试，但 training-time RTC 这部分主要是通过代码路径检查和行为验证接进去的，后续还可以补更细的单元测试，比如 frozen prefix mask、per-token time、以及 no-overlap 退化行为。


## 9. 这个实现的局限性

你主动说出局限性，面试里会更稳。

- 当前 training-time RTC 主要接在 `pi05` 这条链里
- 它依赖 chunk overlap；如果没有 overlap，收益有限
- `simulated_delay` 是人为设定的，和真实部署不匹配时会影响效果
- 当前没有专门的 training-time RTC 单元测试文件
- 如果未来机器人控制频率、执行 horizon、真实 latency 分布变化很大，delay 采样策略可能需要重调


## 10. 最后给面试官的收束句

可以这样收：

> 这项工作的本质，是把“推理时才暴露出来的实时执行约束”前移到训练阶段，让模型在训练时就学会利用 chunk overlap 和 delay 结构。这样部署时它对 real-time chunking 更自然，也更少依赖额外的后处理修正。
