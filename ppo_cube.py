import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dataset_rubik import RubikDataset, collate_fn

from torch.utils.data import DataLoader
from utils import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, IDX_TO_MOVE, move_idx_to_str, convert_state_to_tensor

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


# 假设之前已定义好各个动作的变换函数以及映射字典 move_funcs
def rotate_face(face, times=1):
    """
    对一个3x3的面进行旋转，face 的 shape 为 (9,)
    顺时针旋转一次的映射为：[6,3,0,7,4,1,8,5,2]
    """
    idx = torch.tensor([6, 3, 0, 7, 4, 1, 8, 5, 2], dtype=torch.long)
    new_face = face.clone()
    times = times % 4
    for _ in range(times):
        new_face = new_face[idx]
    return new_face

def move_U(state_54):
    assert state_54.numel() == 54, f"Expected state length 54, got {state_54.numel()}"
    new_state = state_54.clone()
    # U 面（indices 0:9）顺时针旋转
    new_state[0:9] = rotate_face(state_54[0:9], times=1)
    # 邻接面更新：根据 face_order = ['U','L','F','R','B','D']
    # 对于 U 动作：
    # 更新相邻面上行：F 上行 ← L 上行, R 上行 ← F 上行, B 上行 ← R 上行, L 上行 ← B 上行
    temp = state_54[9:12].clone()  # L 上行
    new_state[9:12] = state_54[36:39]  # L <- B
    new_state[36:39] = state_54[27:30]  # B <- R
    new_state[27:30] = state_54[18:21]  # R <- F
    new_state[18:21] = temp  # F <- L
    return new_state


def move_L(state_54):
    assert state_54.numel() == 54, f"Expected state length 54, got {state_54.numel()}"
    new_state = state_54.clone()
    # L 面（indices 9:18）顺时针旋转
    new_state[9:18] = rotate_face(state_54[9:18], times=1)

    # 将各相关面转为 3x3 矩阵
    U_face = state_54[0:9].view(3, 3)
    F_face = state_54[18:27].view(3, 3)
    D_face = state_54[45:54].view(3, 3)
    B_face = state_54[36:45].view(3, 3)

    # U 左列：U_face[:,0]
    U_left = U_face[:, 0].clone()
    # F 左列：F_face[:,0]
    F_left = F_face[:, 0].clone()
    # D 左列：D_face[:,0]
    D_left = D_face[:, 0].clone()
    # B 右列：B_face[:,2]，需要倒序，因为 B 面方向与 U 相反
    B_right = B_face[:, 2].clone()

    new_U_left = torch.flip(B_right, dims=[0])
    new_B_right = torch.flip(D_left, dims=[0])
    new_D_left = F_left.clone()
    new_F_left = U_left.clone()

    U_face[:, 0] = new_U_left
    B_face[:, 2] = new_B_right
    D_face[:, 0] = new_D_left
    F_face[:, 0] = new_F_left

    new_state[0:9] = U_face.view(-1)
    new_state[18:27] = F_face.view(-1)
    new_state[36:45] = B_face.view(-1)
    new_state[45:54] = D_face.view(-1)
    return new_state


def move_F(state_54):
    assert state_54.numel() == 54, f"Expected state length 54, got {state_54.numel()}"
    new_state = state_54.clone()
    # F 面（indices 18:27）顺时针旋转
    new_state[18:27] = rotate_face(state_54[18:27], times=1)

    U_face = state_54[0:9].view(3, 3)
    L_face = state_54[9:18].view(3, 3)
    R_face = state_54[27:36].view(3, 3)
    D_face = state_54[45:54].view(3, 3)

    # U 底行：U_face[2, :]
    U_bottom = U_face[2, :].clone()
    # L 右列：L_face[:,2]
    L_right = L_face[:, 2].clone()
    # R 左列：R_face[:,0]
    R_left = R_face[:, 0].clone()
    # D 顶行：D_face[0, :]
    D_top = D_face[0, :].clone()

    new_U_bottom = torch.flip(L_right, dims=[0])
    new_L_right = D_top.clone()
    new_D_top = torch.flip(R_left, dims=[0])
    new_R_left = U_bottom.clone()

    U_face[2, :] = new_U_bottom
    L_face[:, 2] = new_L_right
    D_face[0, :] = new_D_top
    R_face[:, 0] = new_R_left

    new_state[0:9] = U_face.view(-1)
    new_state[9:18] = L_face.view(-1)
    new_state[27:36] = R_face.view(-1)
    new_state[45:54] = D_face.view(-1)
    return new_state


def move_D(state_54):
    assert state_54.numel() == 54, f"Expected state length 54, got {state_54.numel()}"
    new_state = state_54.clone()
    new_state[45:54] = rotate_face(state_54[45:54], times=1)
    F_face = state_54[18:27].view(3, 3)
    R_face = state_54[27:36].view(3, 3)
    B_face = state_54[36:45].view(3, 3)
    L_face = state_54[9:18].view(3, 3)
    F_bottom = F_face[2, :].clone()
    R_bottom = R_face[2, :].clone()
    B_bottom = B_face[2, :].clone()
    L_bottom = L_face[2, :].clone()
    R_face[2, :] = F_bottom
    B_face[2, :] = R_bottom
    L_face[2, :] = B_bottom
    F_face[2, :] = L_bottom
    new_state[18:27] = F_face.view(-1)
    new_state[27:36] = R_face.view(-1)
    new_state[36:45] = B_face.view(-1)
    new_state[9:18]  = L_face.view(-1)
    return new_state

def move_R(state_54):
    assert state_54.numel() == 54, f"Expected state length 54, got {state_54.numel()}"
    new_state = state_54.clone()
    new_state[27:36] = rotate_face(state_54[27:36], times=1)
    U_face = state_54[0:9].view(3, 3)
    F_face = state_54[18:27].view(3, 3)
    D_face = state_54[45:54].view(3, 3)
    B_face = state_54[36:45].view(3, 3)
    U_right = U_face[:, 2].clone()
    F_right = F_face[:, 2].clone()
    D_right = D_face[:, 2].clone()
    B_left = B_face[:, 0].clone()
    new_F_right = U_right.clone()
    new_D_right = F_right.clone()
    new_B_left  = torch.flip(D_right, dims=[0])
    new_U_right = torch.flip(B_left, dims=[0])
    F_face[:, 2] = new_F_right
    D_face[:, 2] = new_D_right
    B_face[:, 0] = new_B_left
    U_face[:, 2] = new_U_right
    new_state[0:9]   = U_face.view(-1)
    new_state[18:27] = F_face.view(-1)
    new_state[36:45] = B_face.view(-1)
    new_state[45:54] = D_face.view(-1)
    return new_state

def move_B(state_54):
    assert state_54.numel() == 54, f"Expected state length 54, got {state_54.numel()}"
    new_state = state_54.clone()
    new_state[36:45] = rotate_face(state_54[36:45], times=1)
    U_face = state_54[0:9].view(3, 3)
    L_face = state_54[9:18].view(3, 3)
    R_face = state_54[27:36].view(3, 3)
    D_face = state_54[45:54].view(3, 3)
    U_top = U_face[0, :].clone()
    R_right = R_face[:, 2].clone()
    D_top = D_face[0, :].clone()
    L_left = L_face[:, 0].clone()
    new_U_top = torch.flip(R_right, dims=[0])
    new_R_right = torch.flip(D_top, dims=[0])
    new_D_top = torch.flip(L_left, dims=[0])
    new_L_left = torch.flip(U_top, dims=[0])
    U_face[0, :] = new_U_top
    R_face[:, 2] = new_R_right
    D_face[0, :] = new_D_top
    L_face[:, 0] = new_L_left
    new_state[0:9]   = U_face.view(-1)
    new_state[9:18]  = L_face.view(-1)
    new_state[27:36] = R_face.view(-1)
    new_state[45:54] = D_face.view(-1)
    return new_state

##########################################
# 定义动作到函数映射字典
##########################################
move_funcs = {
    "U": move_U,
    "U'": lambda s: move_U(move_U(move_U(s))),
    "U2": lambda s: move_U(move_U(s)),
    "D": move_D,
    "D'": lambda s: move_D(move_D(move_D(s))),
    "D2": lambda s: move_D(move_D(s)),
    "L": move_L,
    "L'": lambda s: move_L(move_L(move_L(s))),
    "L2": lambda s: move_L(move_L(s)),
    "F": move_F,
    "F'": lambda s: move_F(move_F(move_F(s))),
    "F2": lambda s: move_F(move_F(s)),
    "R": move_R,
    "R'": lambda s: move_R(move_R(move_R(s))),
    "R2": lambda s: move_R(move_R(s)),
    "B": move_B,
    "B'": lambda s: move_B(move_B(move_B(s))),
    "B2": lambda s: move_B(move_B(s)),
}

##########################################
# 修改 env_step 支持批量
##########################################
def single_env_step(state, action):
    """
    针对单个样本进行环境步进：
      - state: 形状 (55,) 的状态张量（前54为魔方状态，最后1为上一步动作）
      - action: 单个动作（标量 tensor）
    """
    move_str = IDX_TO_MOVE.get(int(action.item()), None)
    if move_str is None or move_str not in move_funcs:
        return state
    new_cube_state = move_funcs[move_str](state[:54])
    new_state = state.clone()
    new_state[:54] = new_cube_state
    new_state[54] = action  # 记录本次动作
    return new_state

def env_step(states, actions):
    """
    批量环境步进：
      - states: (B, 55) 状态张量
      - actions: (B,) 动作张量
    对每个样本调用 single_env_step
    """
    new_states = states.clone()
    B = states.size(0)
    for i in range(B):
        new_states[i] = single_env_step(states[i][-1], actions[i])
    return new_states


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load("rubik_model_best.pth", map_location=device)
    actor.load_state_dict(state_dict)

    # 实例化 critic 网络
    critic = Critic(config.d_model)

    optimizer_actor = optim.Adam(actor.parameters(), lr=1e-5)
    optimizer_critic = optim.Adam(critic.parameters(), lr=1e-5)

    # 开始 PPO 训练
    ppo_train(actor, critic, optimizer_actor, optimizer_critic,
              num_iterations=100, rollout_len=5, batch_size=16)