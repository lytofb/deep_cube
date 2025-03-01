import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# -------------------------
# 1. 辅助函数和自定义环境
# -------------------------

def compute_reward(state, action):
    """
    自定义奖励函数示例。
    例如：若动作等于目标动作（这里设定为 3），则奖励为 1，否则为 0。
    你需要根据魔方复原的任务设计更合理的奖励函数。
    """
    target_action = 3  # 示例目标动作
    reward = (action == target_action).float()
    return reward


def env_step(state, action):
    """
    环境转移函数示例。
    根据当前状态和动作返回下一个状态，这里为了示例简单，直接返回原状态。
    在实际任务中，需要根据动作对魔方状态进行更新。
    """
    return state  # 仅作占位


def get_training_batch(batch_size, src_seq_len=8, input_dim=55):
    """
    构造一个训练批次：
      - src: 魔方状态张量，形状 (B, src_seq_len, input_dim)
      - decoder_start: decoder 初始 token（假定起始 token id 为 0），形状 (B, 1)
    """
    src = torch.randn(batch_size, src_seq_len, input_dim)
    decoder_start = torch.zeros(batch_size, 1, dtype=torch.long)
    return src, decoder_start


# -------------------------
# 2. 定义 Critic 网络（用于价值函数估计）
# -------------------------
class Critic(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc = nn.Linear(d_model, 1)

    def forward(self, hidden):
        # hidden: shape (B, d_model)
        return self.fc(hidden).squeeze(-1)  # 输出 shape (B,)


# -------------------------
# 3. Rollout 过程
# -------------------------
def rollout(actor, critic, batch_size, rollout_len=5):
    """
    模拟 rollout 过程，收集一段轨迹的数据。
    返回：
      states: 每一步的 (src, decoder_input) 数据
      actions: 每一步采样的动作 (rollout_len, B)
      logprobs: 每一步采样动作的 log 概率 (rollout_len, B)
      rewards: 每一步的奖励 (rollout_len, B)
      values: 每一步的价值估计 (rollout_len+1, B)
      old_logits: 每一步的原始 logits (rollout_len, B, num_moves)
    """
    states = []
    actions = []
    logprobs = []
    rewards = []
    values = []
    old_logits = []

    src, decoder_input = get_training_batch(batch_size)
    current_src = src
    current_decoder = decoder_input
    for t in range(rollout_len):
        # 前向传播：模型输出 logits，形状 (B, tgt_seq_len, num_moves)
        outputs = actor(current_src, current_decoder)
        # 取最后一步输出作为决策依据
        last_logits = outputs[:, -1, :]  # shape: (B, num_moves)
        old_logits.append(last_logits.detach())

        # 为了计算价值，我们需要一个隐藏表示；这里简单使用 last_logits 作为 proxy（实际中应从模型中抽取 hidden state）
        hidden = last_logits  # shape: (B, d_model) — 注意：这里维度可能不匹配，实际中请从模型中获得适当的隐藏状态
        value = critic(hidden)  # shape: (B,)
        values.append(value.detach())

        # 构造策略分布并采样动作
        dist = torch.distributions.Categorical(logits=last_logits)
        action = dist.sample()  # shape: (B,)
        actions.append(action)
        logp = dist.log_prob(action)  # shape: (B,)
        logprobs.append(logp.detach())

        # 计算奖励（基于当前状态和动作）
        reward = compute_reward(current_src, action)  # shape: (B,)
        rewards.append(reward)

        # 存储当前状态
        states.append((current_src, current_decoder))

        # 模拟环境步进：更新状态和 decoder 输入（这里简单地保持状态不变，并将采样动作拼接到 decoder 序列中）
        next_src = env_step(current_src, action)
        next_decoder = torch.cat([current_decoder, action.unsqueeze(-1)], dim=1)
        current_src = next_src
        current_decoder = next_decoder

    # 最后一步状态价值
    outputs = actor(current_src, current_decoder)
    last_logits = outputs[:, -1, :]
    hidden = last_logits
    final_value = critic(hidden)  # shape: (B,)
    values.append(final_value.detach())

    # 将列表转换为张量
    rewards = torch.stack(rewards, dim=0)  # (rollout_len, B)
    logprobs = torch.stack(logprobs, dim=0)  # (rollout_len, B)
    values = torch.stack(values, dim=0)  # (rollout_len+1, B)
    actions = torch.stack(actions, dim=0)  # (rollout_len, B)
    old_logits = torch.stack(old_logits, dim=0)  # (rollout_len, B, num_moves)

    return states, actions, logprobs, rewards, values, old_logits


# -------------------------
# 4. 计算优势和收益（使用 GAE）
# -------------------------
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """
    rewards: (T, B), values: (T+1, B)
    返回 advantages 和 returns，每个形状为 (T, B)
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    gae = 0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = advantages + values[:-1]
    return advantages, returns


# -------------------------
# 5. PPO 更新
# -------------------------
def ppo_update(actor, critic, optimizer_actor, optimizer_critic,
               states, actions, old_logprobs, rewards, values, old_logits,
               clip_eps=0.2, kl_coef=0.1, vf_coef=0.5, ppo_epochs=4):
    """
    进行 PPO 更新，更新 actor 和 critic 参数
    """
    # 计算优势和收益
    advantages, returns = compute_gae(rewards, values)
    # 将 rollout 数据展平，形状 (T*B,)
    T, B = advantages.shape
    advantages = advantages.view(-1)
    returns = returns.view(-1)
    old_logprobs = old_logprobs.view(-1)
    actions = actions.view(-1)
    old_logits = old_logits.view(T * B, -1)

    # PPO 更新循环
    for _ in range(ppo_epochs):
        # 为了更新，我们需要重新计算当前策略下的 logprobs 与价值估计
        # 这里我们简单地对每个存储的 state 进行前向传播，并取最后 token 的输出
        current_logprobs_list = []
        current_values_list = []
        current_logits_list = []
        for (src, decoder_input) in states:
            outputs = actor(src, decoder_input)
            last_logits = outputs[:, -1, :]  # shape: (B, num_moves)
            current_logits_list.append(last_logits)
            dist = torch.distributions.Categorical(logits=last_logits)
            current_logprobs_list.append(dist.log_prob(actions))  # 注意：这里假设 actions 与每个 state 对应的 batch 大小一致
            hidden = last_logits  # 同上，作为价值估计的 proxy
            value = critic(hidden)
            current_values_list.append(value)
        # 对 rollout 中每一步的输出取平均（作为一个简化处理）
        current_logprobs = torch.stack(current_logprobs_list).mean(dim=0)
        current_values = torch.stack(current_values_list).mean(dim=0)
        current_logits = torch.stack(current_logits_list).mean(dim=0)

        # 计算概率比
        ratio = torch.exp(current_logprobs - old_logprobs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        policy_loss = -torch.mean(torch.min(surr1, surr2))

        # KL 惩罚项：计算旧策略和新策略之间的 KL 距离
        old_dist = torch.distributions.Categorical(logits=old_logits)
        new_dist = torch.distributions.Categorical(logits=current_logits)
        kl_div = torch.distributions.kl_divergence(old_dist, new_dist).mean()

        # Critic 损失（价值函数的均方误差）
        value_loss = F.mse_loss(current_values, returns)

        total_loss = policy_loss + vf_coef * value_loss + kl_coef * kl_div

        optimizer_actor.zero_grad()
        optimizer_critic.zero_grad()
        total_loss.backward()
        optimizer_actor.step()
        optimizer_critic.step()

    return total_loss.item(), kl_div.item()


# -------------------------
# 6. PPO 训练主循环
# -------------------------
def ppo_train(actor, critic, optimizer_actor, optimizer_critic,
              num_iterations=100, rollout_len=5, batch_size=16):
    actor.train()
    critic.train()
    for iteration in range(num_iterations):
        # 采样一段 rollout
        states, actions, old_logprobs, rewards, values, old_logits = rollout(actor, critic, batch_size, rollout_len)
        # PPO 更新
        loss, kl_div = ppo_update(actor, critic, optimizer_actor, optimizer_critic,
                                  states, actions, old_logprobs, rewards, values, old_logits)
        if iteration % 10 == 0:
            avg_reward = rewards.mean().item()
            print(f"Iteration {iteration}: loss={loss:.4f}, KL={kl_div:.4f}, avg_reward={avg_reward:.4f}")


# -------------------------
# 7. 主程序入口
# -------------------------
if __name__ == "__main__":
    # 导入你的模型类和配置
    from models.hf_model_history_transformer import RubikSeq2SeqConfig, RubikSeq2SeqForConditionalGeneration

    # 根据预训练时使用的参数初始化配置
    config = RubikSeq2SeqConfig(
        input_dim=55,
        d_model=256,
        nhead=8,
        num_layers=6,
        num_moves=21,
        max_seq_len=50,
        dropout=0.2
    )
    # 实例化 actor 模型
    actor = RubikSeq2SeqForConditionalGeneration(config)
    # 加载预训练权重
    state_dict = torch.load("rubik_model_best.pth", map_location="cpu")
    actor.load_state_dict(state_dict)

    # 实例化 critic 网络
    critic = Critic(config.d_model)

    optimizer_actor = optim.Adam(actor.parameters(), lr=1e-5)
    optimizer_critic = optim.Adam(critic.parameters(), lr=1e-5)

    # 开始 PPO 训练
    ppo_train(actor, critic, optimizer_actor, optimizer_critic,
              num_iterations=100, rollout_len=5, batch_size=16)
