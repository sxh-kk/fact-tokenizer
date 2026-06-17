# 方案3\.0

# 训练

## 第一阶段：训练FACT Tokenizer（基于Ego\-Exo视频学习factorized interaction token）

我们首先利用**配对的Ego\-Exo人类视频**训练一个**FACT**** tokenizer**，目标是学习一种**对ego视角可预测、但又能在训练时吸收exo动作信息**的离散交互表示。

我们不再假设FACT token是单一的latent action token。因为仅通过future feature prediction学到的token可能同时包含：

- 手物交互变化；

- 目标物体变化；

- 任务进展；

- 操作阶段信息；

- ego/exo各自视角下的外观变化；

- 相机运动、遮挡和背景变化。

因此，我们将FACT token space显式分解为两部分：

- **shared action token**：用于编码跨视角一致的shared whole\-body interaction dynamics；

- **private view\-residual token**：用于吸收ego/exo各自视角下不可共享、不可部署或不应迁移的视觉残差信息。

第一阶段最终需要得到的是ego\-accessible的shared action token：

$q^{act}_{ego}$

该token后续作为Ego\-only WAM的训练标签。private residual只在tokenizer训练阶段使用，训练结束后不进入后续WAM和Action Head。



具体来说，给定一段ego transition和对应的exo transition：

- Ego片段：当前时刻及未来短窗口的第一视角视频

- Exo片段：与其同步的第三视角视频

我们分别使用两个 encoder：Ego encoder和Exo encoder：

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YmQwM2ZlNzRhMGI4NzExMjk3NjIxMTg0YWQwOGEwMDFfN2UxMzRjMzdhMzExZDc0YTRkNmM1NzM3NDczNmJhOGJfSUQ6NzY0OTA0NDU2MDYzODc3NDQ4MV8xNzgxMjc0MDgyOjE3ODEzNjA0ODJfVjM)

其中：

- $z_{\text{ego}}^{\text{act}}$和$z_{\text{exo}}^{\text{act}}$：表示用于进入shared action codebook的连续action latent；

- $r_{\text{ego}}^{\text{priv}}$,$r_{\text{exo}}^{\text{priv}}$：表示ego/exo各自视角下的 private residual latent；

    - 它们不参与离散化，也不作为后续WAM的预测目标。其作用是在训练过程中吸收view\-specific residual，使shared action token不必承担完整未来视觉重建的全部信息。



随后，我们不再使用单一共享VQ codebook，而是使用**共享 action codebook方案**

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NGNmYzBjZWQ5ODUwNDVjMWE3OWQ3NDk0ZjJhODU0ZmFfNGRkODg4MzY1NzZlOWMzMzQ5OGE3MWM3MzA4NTM2NzdfSUQ6NzY0OTA0NTE5Njk2OTM4MDc5Nl8xNzgxMjc0MDgyOjE3ODEzNjA0ODJfVjM)

ego action latent 和 exo action latent 都被量化到同一个 shared action codebook 中：

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ZDkxYTFjYmZlOTEzZTVhNTlkZDY0MDdmMDQzOGUxNmJfZmIyOGNmMWQ4Y2E4ZjBiODFiYmM0NzNmYjQ1NzcyNjFfSUQ6NzY0OTA0NTQzMDU0NTc3OTY4NV8xNzgxMjc0MDgyOjE3ODEzNjA0ODJfVjM)

第一阶段真正希望学习的是：

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NmM5ZGFlYjI2YTk5MTgzOGNhODI2NTViMjMwYjM5NWVfMmFkNWYxNDdlMTk5YTE4ZjI4YTEzNGE0MjRiZTgzOGFfSUQ6NzY0OTA0NTY3MTQ1ODI0NTgyNV8xNzgxMjc0MDgyOjE3ODEzNjA0ODJfVjM)

也就是两个视角在 shared action codebook 上应当**对应同一组交互动作原型**。

private residual 不参与跨视角动作对齐：

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NjgwNzg5NmFmYjc3ZWM4N2JhOTY1MGMwYzVjZmZlYzNfNDAzMTlmMzA2OGI4MmY0NDM1MjM3NmVhNDg3MmYzMzdfSUQ6NzY0OTA0NTc4MzEzOTkyOTMxNl8xNzgxMjc0MDgyOjE3ODEzNjA0ODJfVjM)

它们只负责解释各自视角下的视觉残差。



### （1）Shared\-action Swapped Future Reconstruction

为了让shared action token真正表示跨视角一致的交互动态，我们引入**shared\-action swapped future reconstruction**。

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=OWZiNzJlZWVhNjVkYTRjMDJkMjE4OTlhODAzZWQ0NTdfMTEzN2YzNTMxMGUyMzIxNDk0Mjk2MTY5YjVmNmQwYWJfSUQ6NzY0OTA0NjQwNzU3MTE2NDQwM18xNzgxMjc0MDgyOjE3ODEzNjA0ODJfVjM)

如果$q^{\mathrm{act}}$真正表示跨视角一致的交互动作，那么ego的未来应该可以由exo提供的shared action token加上ego自己的private residual来重建；exo的未来也应该可以由ego提供的shared action token加上exo自己的private residual来重建。

**对于ego future prediction：**

decoder 接收当前ego feature、exo\-derived shared action token 和 ego\-private residual：

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ZmJiMjc1NDBkZWViMTM3NzNlMGI3YzliYTgwYmQ2ZDlfOGExMDdhNDcxZjgyYTU0MDgyYmMzNmI4MzJlN2IyYWNfSUQ6NzY0OTA0NzQ2NDgyMTEzMjQ4NF8xNzgxMjc0MDgyOjE3ODEzNjA0ODJfVjM)

对应损失为：

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=OWJhYjg1MzJlMmJjZWZjMGM5Y2MyZTllOWNlYzU1MjlfODVmYjVjY2NmMDM0OWJhNzdlM2Q4NDE5YTdlY2RiNzVfSUQ6NzY0OTA0NzUwNjc4OTQyMDI2N18xNzgxMjc0MDgyOjE3ODEzNjA0ODJfVjM)

**对于 exo future prediction：**

decoder 接收当前exo feature、ego\-derived shared action token 和 exo\-private residual：

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=OWQ1ZWUyZmQ1MGZiNDI0MjRlYjVlZGE4NDQyMzIxOGVfZTc1MWM0YjhjMzZlYzc0NGE2NmRhOWFjYmQ1MTQzZjhfSUQ6NzY0OTA0NzYyMjE4Njk5NDYzOF8xNzgxMjc0MDgyOjE3ODEzNjA0ODJfVjM)

对应损失为：

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YjQwYmMwNGQ4NTMzMjUyMTE5NzZmMGU3NTRiMmM2ZWRfM2JmZTlmMGIzM2U4Yzk5MTY1Y2RiM2Y1OTM2MjQ4NzFfSUQ6NzY0OTA0NzY3NTQ5Njc5NTMzNl8xNzgxMjc0MDgyOjE3ODEzNjA0ODJfVjM)

因此：

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YTVmZjQxNzI0NDcxNTQ3MmFjMzdmOTA0MmE5OTJmZDBfNzRmMzQwOWU2ODU3NWEwODU2Nzk1OWM2MzNjZTZmODZfSUQ6NzY0OTA0Nzc5ODY1MDA0NzY4NV8xNzgxMjc0MDgyOjE3ODEzNjA0ODJfVjM)

这样，shared action token 必须表达两个视角都能够复用的交互动态，而 private residual 只负责解释本视角下的视觉残差。



### （2）Within\-view Reconstruction Warm\-up

在训练初期，可以加入较弱的within\-view future reconstruction作为warm\-up，使encoder、decoder和codebook先获得稳定的 transition representation。

对于ego branch：

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ZTJmZTYwMDdiNTAwZWExMmM5ZDkwZDg3MDVjYmIzYzVfNmIxYWQ5ODYxMTRjMWRkNzUyYWMxNzAwMThmZWZmMjhfSUQ6NzY0OTA0OTA5MDQyNDE2MzMwNl8xNzgxMjc0MDgyOjE3ODEzNjA0ODJfVjM)

对于 exo branch：

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MzhlODAxZWQ5MDYxOGVlYjVlZjRiOTkwY2UxMmM1N2ZfNGMyZTE4YWJmY2JlZGI1NWJlZGQwMWU5OTEzODdkMmNfSUQ6NzY0OTA0OTIyMTM0OTMxMzUxNV8xNzgxMjc0MDgyOjE3ODEzNjA0ODJfVjM)

训练过程中可以采用**权重调度**：

训练早期依赖 $L_{\mathrm{self}}$ 稳定 tokenizer；训练中后期逐渐提高 $L_{\mathrm{swap}}$ 的权重，使主要约束转向跨视角 shared action token 的一致性。



### （3）Confidence\-gated Exo Prototype Alignment

exo 视角通常可以提供更清晰的 whole\-body interaction cues。因此，在 shared action codebook 上引入 confidence\-gated Exo prototype alignment。

ego encoder 和 exo encoder 在 shared action codebook $C_{\text{shared}}^{\text{act}}$ 上分别产生 soft assignment distribution：

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NTBjN2Q3NjJlYjNiYjI0NTJmMjhkZGNkYzY3Yzk1NWJfZGQ3ZGQ5NzM3NWZlMDExOWJhNjEyMzFjODExYmY2ZWRfSUQ6NzY0OTA0OTc5NDIyNDA5ODI2NF8xNzgxMjc0MDgyOjE3ODEzNjA0ODJfVjM)

使用 exo branch 作为 从视角 A 提取 latent action，但用它去重建视角 B 的未来。，对 ego branch 的 shared action assignment 进行带置信度门控的 KL 对齐：

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NTcyYzM5M2Q0YTM4NjVmYzZlYjQ2YjVjNzdiMGE5NzlfYTJhNmZlM2I5MTBjNGU5ZDdkMzAxNWEzYjI5M2FmN2VfSUQ6NzY0OTA0OTk4MTk5MDUzODQ4Ml8xNzgxMjc0MDgyOjE3ODEzNjA0ODJfVjM)

其中：

$sg()$表示 stop\-gradient；$w_{\text{exo}}^{\text{act}}$表示 exo teacher 的置信度。

这个对齐只发生在：

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=N2NhNGJmNGMyMDYxOWZmZmFlNjJjOWVmNzM1Mjk5ZmZfZmY4ODgyZDE0ZGE3MDhiZmI1OTFjNDk4MjQ1YTk4MmZfSUQ6NzY0OTA1MDc2MzM3Njg3MjYyOF8xNzgxMjc0MDgyOjE3ODEzNjA0ODJfVjM)

之间。

exo 只在可迁移的 shared action prototype 层面充当 weak teacher。

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NjY3NDMwOGVhM2YzOTI3NWIwNDIzOTY2NmI0ODE3NDZfNmUzMTUwMWQwMTc4ZWY0MDBlYzZhZWRkMTkxY2QwMzVfSUQ6NzY0OTA3MDcwOTg3MTc5MTAzNl8xNzgxMjc0MDgyOjE3ODEzNjA0ODJfVjM)

### 其他辅助目标：

1. Hand\-object Relative Dynamics Prediction

2. Contact Prediction

3. Private Residual Separation

4. Codebook Usage Balancing



### （用于验证）Tokenizer Quality Validation

为了验证第一阶段得到的 token 是否真正具有 action\-centric 和 ego\-predictable 属性，需要进行 tokenizer quality validation。

1. **View\-invariance**

训练一个小分类器，用$q^{\mathrm{act}}$判断该 token 来自 ego 还是 exo。

如果分类准确率接近随机，说明 shared action token 中的 viewpoint\-specific 信息较少。

2. **Contact Purity**

统计每个 code 内部的 contact pattern 分布。

理想情况下，同一个 code 应该对应相似的 contact 阶段，例如 approach、contact onset、stable contact 或 release。

3. **Object Leakage**

训练一个分类器，用$q^{\mathrm{act}}$预测object category、scene id 或 background type。

如果预测准确率过高，说明 shared action token 中仍然编码了过多 object appearance 或 scene information。

4. **Ego Predictability**

训练一个轻量ego\-only predictor，根据当前和历史 ego observation 预测未来：

如果 token 无法由 ego history 稳定预测，那么它不适合作为部署阶段 WAM 的目标。

5. **Robot Relevance**

在少量机器人teleoperation 或 deployment 数据上，验证$q^{\mathrm{act}}$是否对机器人控制有预测价值。

可以检查：

用$q^{\mathrm{act}}$预测 end\-effector delta；

用$q^{\mathrm{act}}$预测 contact phase；

用$q^{\mathrm{act}}$预测 SONIC coarse trajectory type；

比较使用$q^{\mathrm{act}}$和不使用$q^{\mathrm{act}}$时Action Head的性能差异。



## 第二阶段：冻结shared action token space

在 FACT tokenizer 训练完成后，我们不直接冻结整个 tokenizer，而是冻结其中真正用于后续动作预测的 **shared action token space**。

我们冻结：

- shared action codebook：$C_{\text{shared}}^{\text{act}}$；

- shared action quantizer；

- ego action encoder的主体参数；

- exo action encoder的主体参数；

- private view\-residual latent / private residual branch / private residual projection head；

从这一阶段开始，所有后续训练都基于这个frozen tokenizer产生的 token label。

在 shared action token space 冻结后，我们使用 frozen tokenizer 对 human ego transition 生成稳定的 shared action token label。我们同时保存 frozen tokenizer 在 shared action codebook 上产生的 **hard token label** 和 **soft assignment distribution**。

除了 token label 本身，我们还为每个 token assignment 计算 **confidence weight **$c_t$ 用于衡量该 token label 的可靠性。

confidence weight 可以由以下计算得到：

- Token entropy

- Top\-1 / Top\-2 margin

- Nearest code distance

- Ego\-Exo consistency

- Temporal consistency

- Contact / dynamics consistency



### **Robot ego adapter（可选）**

参考psi0  WAM能否直接弥补gap问题？

（混合？）

LATTE tokenizer 的一阶段训练主要基于人类 Ego\-Exo 视频，而部署时输入来自机器人 ego camera。二者存在明显 domain gap，如果完全冻结 ego encoder，机器人 ego 图像可能无法稳定落到正确的 shared action token 上。

因此，我们在 frozen ego action encoder 前后加入一个轻量级的 robot ego adapter：$A_{robot}$

机器人 ego transition 经过该 adapter 后，被映射到同一个 frozen shared action codebook 中：

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YThlZWViYjQyY2U0YmJkMWE4MWJiMjE4MjAyNDRkZGNfMWYyYmZlMjI2YmJlYWE4ZWFkNDdlNzgxMWVhNzY3MzVfSUQ6NzY0OTA1ODUyMzI0MTg3NjcxN18xNzgxMjc0MDgyOjE3ODEzNjA0ODJfVjM)

其中：

- $A_{\text{robot}}$ 可以训练；

- $E_{\text{ego}}^{\text{act}}$ 的主体保持冻结；

- $C_{\text{shared}}^{\text{act}}$ 保持冻结；

- token ID 语义保持不变。

我们使用少量 robot ego video 或 robot teleoperation data 训练 robot ego adapter。

训练目标可以包含以下几类：

- Weak token distribution alignment

- Temporal consistency

- Contact / gripper phase prediction

- End\-effector\-object relative dynamics

- Robot feature to human action feature alignment

- Adapter regularization



在 shared action token space 固化，并完成 robot ego adapter 的训练后，我们使用 frozen tokenizer 对 human ego data 和 robot ego data 生成稳定的 shared action token label。



## 第三阶段：训练 Video DiT 增强的 Ego\-only WAM

在获得frozen shared action token space之后，我们训练一个**causal Ego\-only WAM**，使其能够仅根据当前和历史ego observation预测未来shared action token。

### （1）Video DiT 提取 denoising feature

我们在 WAM 前引入一个 **Video DiT / Video Dynamics Branch**，用于从当前和历史 ego 视频中提取面向未来交互动态的 **intermediate denoising feature****。**

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ZTRlODRjNmRmMGQ3ZjcyMDUxMWI1OGNjNmVjZjU0YTZfOTA4NTcxNzY4NGExYmJlYWJiMzUzOWViNDIyOTNjMzhfSUQ6NzY0OTc0MzIwNDQ2MzUyODkzNF8xNzgxMjc0MDgyOjE3ODEzNjA0ODJfVjM)

**输入：**

- 当前和历史 ego observation

- structured subgoal embedding（若有高层VLM Planner）

**输出：**

- ego 视频中提取的 intermediate denoising feature

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MmU1NDQ3MGM0NDZhZjY4MDM3NzljMzMyN2ZhYjMwNzJfZTgyZWI4ZTBmNGIxODIxZDA3Y2E1NDg5NDgxMGQ2ZjdfSUQ6NzY0OTc0MDU3Njg3MDkyNzMxNl8xNzgxMjc0MDgyOjE3ODEzNjA0ODJfVjM)

    其中：

    - $\epsilon$ 表示 latent noise；

    - $\tau$ 表示选取的 denoising timestep；

    - $h^{denoise}_t$ 表示 Video DiT 在 denoising 过程中的中间层动态特征。

该 feature 不需要恢复完整未来图像，而是应该编码与后续交互动作预测相关的视觉动态信息。



### （2）Denoising feature 输入 WAM 预测 shared action token

得到 $h^{denoise}_t$ 后，我们将其输入 WAM / token prediction head，预测未来 shared action token distribution：

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ODk1YzUzMzJiMjQ5YmUwY2E0OGNlZTc4MDQ4ZjBkMGVfYjZhNTllN2U4M2E5ZmEyMDU5ZDcyYzk0YjgwOTMyNDZfSUQ6NzY0OTc0MTI3NTAxMjU5ODc2NV8xNzgxMjc0MDgyOjE3ODEzNjA0ODJfVjM)

其中：

- $\hat{p}^{act}_{t:t+H}$ 是未来 shared action token distribution；

- $h^{denoise}_t$ 是 WAM 中间的连续视觉动态表征。

WAM 的输出仍然落在 frozen shared action token space 上。Video DiT 只是增强 WAM 的视觉动态建模能力，不改变 shared action token 的语义空间。



### （3）训练目标

训练时，frozen tokenizer为每个future token position提供soft label：

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MjRiYTc0NmZjNjZkMjMyMGFiMDk0NTRjYzA4OTcxYmNfNmFiOWEwMGQwMmZkN2I0ZGZkNTRiZGRhYTVkYWZhOGNfSUQ6NzY0OTA1NzkyNzQwNzUzNzM2M18xNzgxMjc0MDgyOjE3ODEzNjA0ODJfVjM)

WAM预测对应分布：

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MGY1Y2M4ODBkMTZkMjA2Nzk5ZmQ1YjgzZTU4NWU4OTZfZTM4NzdlZDIyZTgzYTAxZDY0OWU4ZmExMWY1MjU4MTRfSUQ6NzY0OTA1NzkyOTA1MTY4ODE1N18xNzgxMjc0MDgyOjE3ODEzNjA0ODJfVjM)

使用 confidence\-weighted KL loss 训练 WAM：

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MTQwMGMyZmY4Y2Q4N2YzNjlkYjI3ZDFmMTY2OGZkY2JfYTA1ZTI5ZDU5OTQ1OTNmZTM0MmIwZDYxZjYwOWY2NjhfSUQ6NzY0OTc0MjQ0NTMwODMwMDIzOF8xNzgxMjc0MDgyOjE3ODEzNjA0ODJfVjM)

其中：

- $p_{t+k,\text{label}}^{\text{act}}$ 是第二阶段 frozen tokenizer 生成的 soft action token label；

- $\hat{p}_{t+k}^{\text{act}}$ 是 WAM 预测的 token distribution；

- $c_{t+k}$ 是该 token label 的 confidence weight；

- $H$ 是 WAM 预测的 future token chunk 长度；

- $\epsilon$ 用于避免分母为 0。

该监督只作用于shared action token distribution，不预测private residual latent。WAM学习让自己的未来shared action token分布接近frozen tokenizer给出的soft label，并根据 label 可靠性调整监督强度。



#### 其他辅助目标：

1. Contact dynamics prediction

2. Hand\-object / gripper\-object relative dynamics prediction

3. Interaction\-centric future feature prediction

4. Temporal consistency regularization



需要强调的是：

- **Tokenizer可以看未来片段来定义token label**

- **WAM在训练和推理时都只能看当前与历史观测，不能看未来**

因此，WAM是一个**因果的动作预测模型**，而不是非因果的视频重建模型。



## 第四阶段：训练Action Head（实现FACT token到SONIC执行空间的对齐）

对齐目标？

在Ego\-only WAM训练完成后，我们冻结WAM主体，并使用少量机器人遥操作/部署数据训练**Action Head**，实现从shared action token到SONIC执行空间的对齐。

为了避免Action Head依赖 Video DiT feature而绕过FACT token，本阶段采用如下设计原则：

**FACT token是主动作语义通道**

**Video DiT feature只作为grounding context**

也就是说：

- WAM 预测的 shared action token distribution 决定“接下来应该发生什么 whole\-body interaction phase”；

- Video DiT denoising feature 只提供 object context、state grounding、局部视觉残差信息；

- Action Head 主要根据 shared action token 生成 SONIC motion chunk；

- denoising feature 只用于修正、对齐、gating 和环境 grounding，不作为主要动作语义来源。



**Action Head输入：**

- WAM输出的top\-k expected shared action embedding；

- 低带宽 Video DiT context（只作为 grounding / residual context）；

- 外部环境状态；

    - 当前 robot ego RGB / RGB\-D feature；

    - object feature；

    - object bbox / mask；

    - object pose；

- 机器人自身状态；

    - base pose / base velocity；

    - IMU；

    - joint position；

- action history。

- structured subgoal embedding（若有VLM Planner）



**Action Head输出：**

- 64D SONIC latent motion token chunk；

- left hand / gripper joint command；

- right hand / gripper joint command；

- state\-conditioned execution gating；



### （1）输入Action Head前的数据处理

#### **Top\-k token distribution interface**

对于每个未来 token position $t+k$，我们从 WAM 预测分布中取 top\-k 候选：

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MGI5NWI1MTYwMWU2M2Y4NjU2ZGUzNDgyZjMwZTgzZjNfNjliNDc3YjMxNWVkN2ZmZmY3YmVlNWRhYzVmZDE5MjhfSUQ6NzY0OTc3OTA4MDEyODI1MzExOV8xNzgxMjc0MDgyOjE3ODEzNjA0ODJfVjM)

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NDUxYjZjMTU4MGI3YTNjNjE4NTQ5YzBmZWRlYTQ2MzZfNjQ1OWJjZDgwNmU3MmU3NDJhZDQ5M2JiYWMyMWNhZjFfSUQ6NzY0OTc3OTA5ODQ1MzIxNjIwOV8xNzgxMjc0MDgyOjE3ODEzNjA0ODJfVjM)

为了使 Action Head 感知 WAM 的不确定性，我们不只传 token embedding，还传递完整 candidate information：

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NTAxNmMzZjQwMTE1YTUzZGQ5OTc3NTFiZjMzMzBlZDlfMWIxZGM0YTY5Mjc1YjY0ODZmOGRmNjJlZmRkYzdmNjhfSUQ6NzY0OTc3OTM2OTkyOTQ0NDMwMV8xNzgxMjc0MDgyOjE3ODEzNjA0ODJfVjM)

其中：

- $i_{t+k}^{j}$ 是 top\-k token ID；

- $e_{t+k}^{j}$ 是对应 token embedding；

- $\pi_{t+k}^{j}$ 是该 token 的 probability；

- $\log \pi_{t+k}^{j}$ 提供 log\-probability 信息；

- $H^{act}_{t+k}$ 是该位置 token distribution 的 entropy；

- $m^{act}_{t+k}$ 是 top\-1 / top\-2 margin；

- $PE(k)$ 是 horizon position embedding；

- $\delta^{phase}_{t+k}$ 是 phase transition signal。

其中 entropy 定义为：

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ZWFmYmQ5Njc1MWU2YzM3ZmM5ZmY3ZmI5ZGU3MWFkNDlfMmI4YmY5OGExM2U2ZjNhZWM2MDgxOTVjZmRkOTYxYjhfSUQ6NzY0OTc3OTYyNjk5ODQwMjI0NF8xNzgxMjc0MDgyOjE3ODEzNjA0ODJfVjM)

top\-1 / top\-2 margin 定义为：

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NzQ3YjA5NTM4NmMyMmZkYjcwNDEwMjRlYTcwM2I5MDNfZDFkZTcyZjhmYTIwOTczNWU0ZTQ1MjU1YTI3MDE1OWVfSUQ6NzY0OTc3OTY4MDA2NDYzODE5NV8xNzgxMjc0MDgyOjE3ODEzNjA0ODJfVjM)

phase transition signal 用于提示当前 token 相对于上一时刻或上一 horizon 是否发生阶段切换，可以包含：

- continue probability；

- switch probability；

- delay probability；

- token change indicator；

- previous executed token consistency。

因此，WAM 传给 Action Head 的不再是单个 embedding，而是一组结构化 top\-k candidate token set：

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YzQ2NmVlY2QxYjJjZGZiYWU5NzUwMTRkMGRjNjZhOGVfNmI4MmRmZTI5MGNlOGU1MWI3ZWIwZGViODE5ZWMzNDdfSUQ6NzY0OTc3OTc5NTU3NTk4MzMwNV8xNzgxMjc0MDgyOjE3ODEzNjA0ODJfVjM)



#### Video DiT feature 的低带宽上下文化

为了避免 Action Head 直接从高维 Video DiT feature 预测 SONIC motion token，我们不直接输入完整 $h^{denoise}_t$，而是进行 stop\-gradient 和低带宽 projection：

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ZDhkNzBlZjBiODRjODkyM2UzNTAxODg4MzAzZGY2YzZfMTE4YjA3NTE4NmUzZGRhNTA5NTdmZWMyMGQ4OTBhOTdfSUQ6NzY0OTc4MDI2ODU2MzQ4MzYyMV8xNzgxMjc0MDgyOjE3ODEzNjA0ODJfVjM)

其中：

- $sg(\cdot)$ 表示 stop\-gradient；

- $B_{\omega}$ 是低带宽 bottleneck projection；

- $\tilde{h}^{denoise}_t$ 是压缩后的 visual grounding context。

并限制：

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ZTVkNGNiYmQxNzczNzdjYWViODhjYjBmNDZlOThhMzdfNjVmYTVlZmNjMzczZTBlZjlhODVkZWQwZmZmNjNlNDNfSUQ6NzY0OTc4MDQ2MjEwOTQ5NDUwNF8xNzgxMjc0MDgyOjE3ODEzNjA0ODJfVjM)

它只应该提供：

- object\-centric context；

- contact\-region context；

- target visibility；

- local geometry / depth context；

- gripper\-object alignment context；

- visual residual information；

- execution grounding signal。

它不应独立承担主要动作语义预测。



### （2）Token\-as\-query 的 Action Head 结构

**Action Head 输入：**

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NGEzY2Y4YzUxMmIwMzVhMWMxYjc1M2ZmYWNlN2U4NzhfODlhNDc0MWVhNGE2M2ViNjE3NjlmMDRkZGIyNmUxMTdfSUQ6NzY1MDA2OTA4NTY2OTIxNTQyMF8xNzgxMjc0MDgyOjE3ODEzNjA0ODJfVjM)

其中：

- $\bar{e}^{act}_{t:t+H}$：WAM 预测的 shared action token embedding，作为主动作语义瓶颈；

- $\tilde{h}^{denoise}_t$：低带宽 Video DiT context，只作为 grounding / residual context；

- $f^{robot\ ego}_t$：当前 robot ego RGB / RGB\-D feature；

- $f^{object}_t$：object feature，包括 bbox、mask、pose、depth、tracking state 等；

- $s^{robot}_t$：机器人 proprioception，包括 base pose、base velocity、IMU、joint position、joint velocity 等；

- $a_{<t}$：历史动作，包括上一段 SONIC token、手部命令、执行状态等；

- $s_t$：structured subgoal embedding。



为了进一步防止 Video DiT feature 绕过 shared action token，我们不采用简单 concat 结构，而是采用 **token\-as\-query** 的 cross\-attention 结构。

首先，将 shared action token embedding 编码为 Action Head 的主 query：

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MGFkZjE0NzYyZWM1NmUxMjZhOTc4NTQ3N2MzNDEyOGZfNDFlMTBmNmVmMzMyYWJjMDlmYWNiNmY2NWIxNmZkNzBfSUQ6NzY1MDA2OTI4NjI2NjcwMjc5OV8xNzgxMjc0MDgyOjE3ODEzNjA0ODJfVjM)

然后，将低带宽 Video DiT context、object feature 和 robot state 编码为 key / value：

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ZDdiYWU0YWM4NWVjZTgzYmJjZjNhOTIxNDRkMjFjODBfNjFhNjE4NjViZTE1ZWQ1ZDhiYjVkZDk4NzMwYjUzNDdfSUQ6NzY1MDA2OTM0MDU0OTY2MzkyNF8xNzgxMjc0MDgyOjE3ODEzNjA0ODJfVjM)

再进行 cross\-attention：

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MDcyYjU3NmFjNDEyOTk1MDY5M2Y4YzEwMTVjYjRiMTRfMzUwOTgyZGU5YmI4Nzg1ZTE5NDBlNGU5Nzg0OTIxMDJfSUQ6NzY1MDA2OTQwNjkxOTQ3ODI0NF8xNzgxMjc0MDgyOjE3ODEzNjA0ODJfVjM)

其中：

- token branch 是 query；

- feature branch 只是 key / value context；

- **Action Head 必须先基于 shared action token 提出动作语义，再从 visual context 中查询执行细节**。





### （3）**State\-conditioned execution gating**

Action Head同时显式输出 **state\-conditioned execution gating** ：

- Execute：

    - 当前机器人状态已经满足执行条件，可以直接执行该 shared action token 对应的 SONIC motion chunk。

- Delay/refine：

    - 当前 shared action token 的语义是对的，但机器人还没有准备好直接执行，需要先进行靠近、对齐、姿态调整或稳定。

- Smooth/continue：

    - 当前仍在执行上一段 motion chunk，不应该突然切换动作，而应该根据 action history 平滑延续。

- Recover：

    - 当前视觉 grounding 或执行状态异常，需要进入恢复策略。



**训练监督**来自少量遥操作/部署数据中记录的：

- logged SONIC 64D token\_state

- 左右手关节命令



**训练方式：**

1. Teacher\-forced token training

让Action Head先学会：如果给定较可靠的shared action token和intermediate denoising feature，以及当前环境状态、机器人状态和历史动作，如何输出SONIC执行表示。

2. Mixed token training / scheduled sampling

训练早期主要使用 teacher\-forced token，训练后期逐渐增加 WAM\-predicted token 的比例，让 Action Head 适应 WAM 的预测噪声。



# 推理

## 第一阶段：VLM进行高层任务分解（可选）

给定用户输入的语言指令，如：“Make me a cup of tea and bring it to me\.”

VLM Planner接收：

- 用户语言指令；

- 当前或低频 ego 图像；

- 可选的 object detection / segmentation / scene graph；

- 可选的历史执行状态。

输出一组**structured subgoals：**

- subgoal id；

- subgoal type；

- target object；

- goal state；

- precondition；

- success condition；

- failure condition；

- constraints；

这一模块负责高层规划和任务分解，运行频率较低。

## 第二阶段：Ego\-only WAM预测action latent

随后，机器人**当前和历史机器人第一视角观测**输入WAM；

1. Video DiT / Video Dynamics Branch 提取 intermediate denoising feature

2. WAM 基于该 denoising feature 预测未来 shared action token distribution

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NzljZGUyYTc2NmVjZWU1YmQ4OTk4NWY4OWNmMGFjOTZfZGQzODUyNzkxYTI2MmZiNGI5YjgxZDhkZTBhYTA1NTJfSUQ6NzY0OTc0NDI4MDU3NTU2MDY3Nl8xNzgxMjc0MDgyOjE3ODEzNjA0ODJfVjM)

其中：

- $\hat{p}^{act}_{t:t+H}$ 是未来 shared action token distribution；

- 每个 $\hat{p}^{act}_{t+k}$ 都定义在 frozen shared action codebook 上；

- $h^{denoise}_t$ 是当前场景下的连续未来视觉动态先验。

然后将该分布转换为：top\-k expected embedding

WAM可以使用第二阶段得到的robot ego adapter或经过robot\-domain adaptation的视觉前端。

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YzJiNzU3MWI3NWJmMjQ1ZGQ1YWRhOTM3OTFjYWRjZjFfN2RmNmQ1MDMxNmJmZjQ1ZjRhNDg3OTU3OWZiZTJmYTNfSUQ6NzY0OTA1ODE4ODY2MjE0ODMxNF8xNzgxMjc0MDgyOjE3ODEzNjA0ODJfVjM)

其中：

- $A_{\text{robot}}$用于对齐robot ego visual domain；

- WAM输出仍然落在frozen shared action token space；

- shared action token ID的语义保持不变。

## 第三阶段：Action Head将action latent转为SONIC粗轨迹

WAM输出的top\-k shared action embedding和Video DiT denoising feature 会被送入Action Head，进一步转成SONIC所需的执行表示，包括：

- 64D SONIC latent motion token chunk和左右手关节命令

这一步相当于把高层的动作语义表示，转换成机器人底层控制器可以理解的动作接口。



## 第四阶段：SONIC底层生成并执行机器人轨迹

最后，Action Head输出的SONIC token和手部控制命令被输入：Frozen SONIC decoder/controller

由SONIC底层模块生成whole\-body joint command，并驱动机器人执行动作。

整个系统以**receding horizon**方式循环运行：

1. 观测当前状态

2. WAM预测下一段FACT token

3. Action Head转成SONIC token

4. SONIC执行短时间动作

5. 获取新观测

6. 再次预测下一段动作

因此推理闭环可以概括为：

**语言指令\+当前ego观测→子任务分解→WAM预测action latent→Action Head转为SONIC执行格式→SONIC 输出机器人轨迹**



# 消融实验





新的数采方案？
