import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dataset_rubik import RubikDataset, collate_fn

from torch.utils.data import DataLoader
from utils import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN


# -------------------------
# 1. 辅助函数和自定义环境
# -------------------------
def compute_reward(state, action):
    """
    自定义奖励函数示例：
    假设当动作等于目标动作（这里设为 3）时奖励为 1，否则为 0。
    """
    target_action = 3  # 示例目标动作
    reward = (action == target_action).float()
    return reward


def env_step(state, action):
    """
    环境转移函数示例：
    根据当前状态和动作返回下一个状态。
    这里为了示例，直接返回原状态，实际中需要更新魔方状态。
    """
    return state  # 仅作占位


def get_training_batch():
    global train_dataloader_iter

    try:
        src_batch, tgt_batch = next(train_dataloader_iter)
    except StopIteration:
        # 如果迭代到头了，重新创建迭代器
        train_dataloader_iter = iter(train_dataloader)
        src_batch, tgt_batch = next(train_dataloader_iter)

    # src_batch.shape = (B, history_len+1, 55)
    B = src_batch.shape[0]

    # decoder_start 形状 (B, 1)，这里用0当作起始token
    decoder_start = torch.full((B, 1), SOS_TOKEN, dtype=torch.long)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_batch = src_batch.to(device)
    decoder_start = decoder_start.to(device)

    return src_batch, decoder_start


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
    假设 actor 模型支持参数 return_hidden=True，返回 (hidden, logits)
    """
    states = []
    actions = []
    logprobs = []
    rewards = []
    values = []
    old_logits = []

    src, decoder_input = get_training_batch()
    current_src = src
    current_decoder = decoder_input
    for t in range(rollout_len):
        # 获取 actor 输出：要求返回 (hidden, logits)
        # hidden: (B, tgt_seq_len, d_model), logits: (B, tgt_seq_len, num_moves)
        hidden, logits = actor(current_src, current_decoder, return_hidden=True)
        last_logits = logits[:, -1, :]  # shape: (B, num_moves)
        last_hidden = hidden[:, -1, :]  # shape: (B, d_model)
        old_logits.append(last_logits.detach())

        # 计算价值：使用最后 token 的 hidden 状态作为 critic 输入
        value = critic(last_hidden)  # shape: (B,)
        values.append(value.detach())

        # 构造策略分布并采样动作
        dist = torch.distributions.Categorical(logits=last_logits)
        action = dist.sample()  # shape: (B,)
        actions.append(action)
        logp = dist.log_prob(action)  # shape: (B,)
        logprobs.append(logp.detach())

        # 计算奖励
        reward = compute_reward(current_src, action)  # shape: (B,)
        rewards.append(reward)

        # 存储当前状态
        states.append((current_src, current_decoder))

        # 更新环境：这里简单地将动作拼接到 decoder 序列中，状态保持不变
        next_src = env_step(current_src, action)
        next_decoder = torch.cat([current_decoder, action.unsqueeze(-1)], dim=1)
        current_src = next_src
        current_decoder = next_decoder

    # 最后一步状态的价值估计
    hidden, logits = actor(current_src, current_decoder, return_hidden=True)
    last_logits = logits[:, -1, :]
    last_hidden = hidden[:, -1, :]
    final_value = critic(last_hidden)  # shape: (B,)
    values.append(final_value.detach())

    # 转换为张量
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
    参数说明：
      - states: 长度为 T 的列表，每个元素是一个元组 (src, decoder_input) ，shape 分别为 (B, ...)；
      - actions: tensor，shape (T, B)；
      - old_logprobs: tensor，shape (T, B)；
      - rewards: tensor，shape (T, B)；
      - values: tensor，shape (T+1, B)；
      - old_logits: tensor，shape (T, B, num_moves)。
    """
    # 计算优势和收益，返回的优势和 returns 形状均为 (T, B)
    advantages, returns = compute_gae(rewards, values)
    T, B = advantages.shape

    for epoch in range(ppo_epochs):
        current_logprobs_list = []
        current_values_list = []
        current_logits_list = []
        # 针对 rollout 中每个时间步计算当前策略下的输出
        for t in range(T):
            src_t, dec_t = states[t]  # src_t, dec_t 均形状 (B, ...)
            # 要求 actor 支持参数 return_hidden=True，返回 (hidden, logits)
            hidden_t, logits_t = actor(src_t, dec_t, return_hidden=True)
            last_logits_t = logits_t[:, -1, :]  # (B, num_moves)
            current_logits_list.append(last_logits_t)
            dist_t = torch.distributions.Categorical(logits=last_logits_t)
            # 注意：针对第 t 个时间步使用 actions[t]，形状 (B,)
            current_logprobs_list.append(dist_t.log_prob(actions[t]))
            last_hidden_t = hidden_t[:, -1, :]  # (B, d_model)
            current_values_list.append(critic(last_hidden_t))

        # 将各时间步数据堆叠：形状分别为 (T, B) 或 (T, B, num_moves)
        current_logprobs = torch.stack(current_logprobs_list, dim=0)  # (T, B)
        current_values = torch.stack(current_values_list, dim=0)  # (T, B)
        current_logits = torch.stack(current_logits_list, dim=0)  # (T, B, num_moves)

        # 将所有时间步与 batch 数据展平为 (T*B,)
        current_logprobs_flat = current_logprobs.view(-1)
        current_values_flat = current_values.view(-1)
        old_logprobs_flat = old_logprobs.view(-1)
        advantages_flat = advantages.view(-1)
        returns_flat = returns.view(-1)
        current_logits_flat = current_logits.view(T * B, -1)
        old_logits_flat = old_logits.view(T * B, -1)

        # 计算概率比率
        ratio = torch.exp(current_logprobs_flat - old_logprobs_flat)
        surr1 = ratio * advantages_flat
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages_flat
        policy_loss = -torch.mean(torch.min(surr1, surr2))

        # 计算新旧策略之间的 KL 散度
        old_dist = torch.distributions.Categorical(logits=old_logits_flat)
        new_dist = torch.distributions.Categorical(logits=current_logits_flat)
        kl_div = torch.distributions.kl_divergence(old_dist, new_dist).mean()

        # Critic 的均方误差损失
        value_loss = nn.functional.mse_loss(current_values_flat, returns_flat)

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
        # 采集 rollout 数据
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
    from models.hf_model_history_transformer import RubikSeq2SeqConfig, RubikSeq2SeqForConditionalGeneration

    # 假设你有一个数据目录 data_dir，并且想要history_len=8
    train_dataset = RubikDataset(data_dir='rubik_shards', history_len=8)

    # 使用collate_fn把Dataset取出的样本batch化
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=16,  # 可以按需调整
        shuffle=True,
        collate_fn=collate_fn
    )
    # 假设在某个文件或作用域中能访问到 train_dataloader
    train_dataloader_iter = iter(train_dataloader)  # 先创建一个迭代器

    # 初始化模型配置（与预训练时保持一致）
    config = RubikSeq2SeqConfig(
        input_dim=55,
        d_model=256,
        nhead=8,
        num_layers=6,
        num_moves=21,
        max_seq_len=50,
        dropout=0.2
    )
    # 实例化 actor 模型，并加载预训练权重
    actor = RubikSeq2SeqForConditionalGeneration(config)
    state_dict = torch.load("rubik_model_best.pth", map_location="cpu")
    actor.load_state_dict(state_dict)

    # 实例化 critic 网络
    critic = Critic(config.d_model)

    optimizer_actor = optim.Adam(actor.parameters(), lr=1e-5)
    optimizer_critic = optim.Adam(critic.parameters(), lr=1e-5)

    # 开始 PPO 训练
    ppo_train(actor, critic, optimizer_actor, optimizer_critic,
              num_iterations=100, rollout_len=5, batch_size=16)