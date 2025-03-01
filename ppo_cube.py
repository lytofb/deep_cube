import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_rubik import RubikDataset, collate_fn
from utils import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, IDX_TO_MOVE, move_idx_to_str, convert_state_to_tensor

# 定义全局 device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


##########################################
# 1. 辅助函数和自定义环境
##########################################
def compute_reward(state, action):
    """
    自定义奖励函数示例：
    假设当动作等于目标动作（这里设为 3）时奖励为 1，否则为 0。
    """
    target_action = 3  # 示例目标动作
    reward = (action == target_action).float()
    return reward


def rotate_face(face, times=1):
    """
    对一个 3x3 的面进行旋转，face 的 shape 为 (9,)
    顺时针旋转一次的映射为：[6, 3, 0, 7, 4, 1, 8, 5, 2]
    """
    idx = torch.tensor([6, 3, 0, 7, 4, 1, 8, 5, 2], dtype=torch.long, device=device)
    new_face = face.clone().to(device)
    times = times % 4
    for _ in range(times):
        new_face = new_face[idx]
    return new_face


def move_U(state_54):
    assert state_54.numel() == 54, f"Expected state length 54, got {state_54.numel()}"
    new_state = state_54.clone().to(device)
    new_state[0:9] = rotate_face(state_54[0:9].to(device), times=1)
    temp = state_54[9:12].clone().to(device)
    new_state[9:12] = state_54[36:39].to(device)
    new_state[36:39] = state_54[27:30].to(device)
    new_state[27:30] = state_54[18:21].to(device)
    new_state[18:21] = temp
    return new_state


def move_L(state_54):
    assert state_54.numel() == 54, f"Expected state length 54, got {state_54.numel()}"
    new_state = state_54.clone().to(device)
    new_state[9:18] = rotate_face(state_54[9:18].to(device), times=1)
    U_face = state_54[0:9].view(3, 3).to(device)
    F_face = state_54[18:27].view(3, 3).to(device)
    D_face = state_54[45:54].view(3, 3).to(device)
    B_face = state_54[36:45].view(3, 3).to(device)
    U_left = U_face[:, 0].clone()
    F_left = F_face[:, 0].clone()
    D_left = D_face[:, 0].clone()
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
    new_state = state_54.clone().to(device)
    new_state[18:27] = rotate_face(state_54[18:27].to(device), times=1)
    U_face = state_54[0:9].view(3, 3).to(device)
    L_face = state_54[9:18].view(3, 3).to(device)
    R_face = state_54[27:36].view(3, 3).to(device)
    D_face = state_54[45:54].view(3, 3).to(device)
    U_bottom = U_face[2, :].clone()
    L_right = L_face[:, 2].clone()
    R_left = R_face[:, 0].clone()
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
    new_state = state_54.clone().to(device)
    new_state[45:54] = rotate_face(state_54[45:54].to(device), times=1)
    F_face = state_54[18:27].view(3, 3).to(device)
    R_face = state_54[27:36].view(3, 3).to(device)
    B_face = state_54[36:45].view(3, 3).to(device)
    L_face = state_54[9:18].view(3, 3).to(device)
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
    new_state[9:18] = L_face.view(-1)
    return new_state


def move_R(state_54):
    assert state_54.numel() == 54, f"Expected state length 54, got {state_54.numel()}"
    new_state = state_54.clone().to(device)
    new_state[27:36] = rotate_face(state_54[27:36].to(device), times=1)
    U_face = state_54[0:9].view(3, 3).to(device)
    F_face = state_54[18:27].view(3, 3).to(device)
    D_face = state_54[45:54].view(3, 3).to(device)
    B_face = state_54[36:45].view(3, 3).to(device)
    U_right = U_face[:, 2].clone()
    F_right = F_face[:, 2].clone()
    D_right = D_face[:, 2].clone()
    B_left = B_face[:, 0].clone()
    new_F_right = U_right.clone()
    new_D_right = F_right.clone()
    new_B_left = torch.flip(D_right, dims=[0])
    new_U_right = torch.flip(B_left, dims=[0])
    F_face[:, 2] = new_F_right
    D_face[:, 2] = new_D_right
    B_face[:, 0] = new_B_left
    U_face[:, 2] = new_U_right
    new_state[0:9] = U_face.view(-1)
    new_state[18:27] = F_face.view(-1)
    new_state[36:45] = B_face.view(-1)
    new_state[45:54] = D_face.view(-1)
    return new_state


def move_B(state_54):
    assert state_54.numel() == 54, f"Expected state length 54, got {state_54.numel()}"
    new_state = state_54.clone().to(device)
    new_state[36:45] = rotate_face(state_54[36:45].to(device), times=1)
    U_face = state_54[0:9].view(3, 3).to(device)
    L_face = state_54[9:18].view(3, 3).to(device)
    R_face = state_54[27:36].view(3, 3).to(device)
    D_face = state_54[45:54].view(3, 3).to(device)
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
    new_state[0:9] = U_face.view(-1)
    new_state[9:18] = L_face.view(-1)
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
# 修改 env_step 支持批量，并统一设备
##########################################
def single_env_step(state, action):
    """
    针对单个样本进行环境步进：
      - state: 形状 (55,) 的状态张量（前54为魔方状态，最后1为上一步动作）
      - action: 单个动作（标量 tensor）
    """
    move_str = IDX_TO_MOVE.get(int(action.item()), None)
    if move_str is None or move_str not in move_funcs:
        return state.to(device)
    new_cube_state = move_funcs[move_str](state[:54].to(device))
    new_state = state.clone().to(device)
    new_state[:54] = new_cube_state.to(device)
    new_state[54] = action.to(device)
    return new_state


def env_step_batch(states, actions):
    """
    批量环境步进：
      - states: (B, 55) 状态张量
      - actions: (B,) 动作张量
    对每个样本调用 single_env_step，并更新状态序列的最后一项
    """
    new_states = states.clone().to(device)
    B = states.size(0)
    for i in range(B):
        # 假设每个样本的状态序列为 (history_len+1, 55)，我们取最后一项进行更新
        last_state = states[i, -1, :]  # (55,)
        updated_state = single_env_step(last_state, actions[i])
        new_states[i, -1, :] = updated_state
    return new_states


##########################################
# 数据加载：保证数据在 device 上
##########################################
def get_training_batch():
    global train_dataloader_iter, train_dataloader
    try:
        src_batch, tgt_batch = next(train_dataloader_iter)
    except StopIteration:
        train_dataloader_iter = iter(train_dataloader)
        src_batch, tgt_batch = next(train_dataloader_iter)
    # src_batch.shape = (B, history_len+1, 55)
    B = src_batch.shape[0]
    decoder_start = torch.full((B, 1), SOS_TOKEN, dtype=torch.long, device=device)
    src_batch = src_batch.to(device)
    return src_batch, decoder_start


##########################################
# 2. 定义 Critic 网络（用于价值函数估计）
##########################################
class Critic(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc = nn.Linear(d_model, 1)

    def forward(self, hidden):
        return self.fc(hidden).squeeze(-1)


##########################################
# 3. Rollout 过程
##########################################
def rollout(actor, critic, batch_size, rollout_len=5):
    states = []
    actions = []
    logprobs = []
    rewards = []
    values = []
    old_logits = []

    src, decoder_input = get_training_batch()  # src: (B, history_len+1, 55)
    current_src = src
    current_decoder = decoder_input
    for t in range(rollout_len):
        hidden, logits = actor(current_src, current_decoder, return_hidden=True)  # hidden: (B, seq_len, d_model)
        last_logits = logits[:, -1, :]  # (B, num_moves)
        last_hidden = hidden[:, -1, :]  # (B, d_model)
        old_logits.append(last_logits.detach())
        value = critic(last_hidden)
        values.append(value.detach())
        dist = torch.distributions.Categorical(logits=last_logits)
        action = dist.sample()  # (B,)
        actions.append(action)
        logp = dist.log_prob(action)
        logprobs.append(logp.detach())
        reward = compute_reward(current_src, action)
        rewards.append(reward)
        states.append((current_src, current_decoder))
        next_src = env_step_batch(current_src, action)
        next_decoder = torch.cat([current_decoder, action.unsqueeze(-1)], dim=1).to(device)
        current_src = next_src
        current_decoder = next_decoder

    hidden, logits = actor(current_src, current_decoder, return_hidden=True)
    last_logits = logits[:, -1, :]
    last_hidden = hidden[:, -1, :]
    final_value = critic(last_hidden)
    values.append(final_value.detach())

    rewards = torch.stack(rewards, dim=0)
    logprobs = torch.stack(logprobs, dim=0)
    values = torch.stack(values, dim=0)
    actions = torch.stack(actions, dim=0)
    old_logits = torch.stack(old_logits, dim=0)

    return states, actions, logprobs, rewards, values, old_logits


##########################################
# 4. 计算优势和收益（使用 GAE）
##########################################
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards, device=device)
    gae = 0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = advantages + values[:-1]
    return advantages, returns


##########################################
# 5. PPO 更新
##########################################
def ppo_update(actor, critic, optimizer_actor, optimizer_critic,
               states, actions, old_logprobs, rewards, values, old_logits,
               clip_eps=0.2, kl_coef=0.1, vf_coef=0.5, ppo_epochs=4):
    advantages, returns = compute_gae(rewards, values)
    T, B = advantages.shape
    for epoch in range(ppo_epochs):
        current_logprobs_list = []
        current_values_list = []
        current_logits_list = []
        for t in range(T):
            src_t, dec_t = states[t]
            hidden_t, logits_t = actor(src_t, dec_t, return_hidden=True)
            last_logits_t = logits_t[:, -1, :]
            current_logits_list.append(last_logits_t)
            dist_t = torch.distributions.Categorical(logits=last_logits_t)
            current_logprobs_list.append(dist_t.log_prob(actions[t]))
            last_hidden_t = hidden_t[:, -1, :]
            current_values_list.append(critic(last_hidden_t))
        current_logprobs = torch.stack(current_logprobs_list, dim=0)
        current_values = torch.stack(current_values_list, dim=0)
        current_logits = torch.stack(current_logits_list, dim=0)
        current_logprobs_flat = current_logprobs.view(-1)
        current_values_flat = current_values.view(-1)
        old_logprobs_flat = old_logprobs.view(-1)
        advantages_flat = advantages.view(-1)
        returns_flat = returns.view(-1)
        current_logits_flat = current_logits.view(T * B, -1)
        old_logits_flat = old_logits.view(T * B, -1)
        ratio = torch.exp(current_logprobs_flat - old_logprobs_flat)
        surr1 = ratio * advantages_flat
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages_flat
        policy_loss = -torch.mean(torch.min(surr1, surr2))
        old_dist = torch.distributions.Categorical(logits=old_logits_flat)
        new_dist = torch.distributions.Categorical(logits=current_logits_flat)
        kl_div = torch.distributions.kl_divergence(old_dist, new_dist).mean()
        value_loss = nn.functional.mse_loss(current_values_flat, returns_flat)
        total_loss = policy_loss + vf_coef * value_loss + kl_coef * kl_div
        optimizer_actor.zero_grad()
        optimizer_critic.zero_grad()
        total_loss.backward()
        optimizer_actor.step()
        optimizer_critic.step()
    return total_loss.item(), kl_div.item()


##########################################
# 6. PPO 训练主循环
##########################################
def ppo_train(actor, critic, optimizer_actor, optimizer_critic,
              num_iterations=100, rollout_len=5, batch_size=16):
    actor.train()
    critic.train()
    for iteration in range(num_iterations):
        states, actions, old_logprobs, rewards, values, old_logits = rollout(actor, critic, batch_size, rollout_len)
        loss, kl_div = ppo_update(actor, critic, optimizer_actor, optimizer_critic,
                                  states, actions, old_logprobs, rewards, values, old_logits)
        if iteration % 10 == 0:
            avg_reward = rewards.mean().item()
            print(f"Iteration {iteration}: loss={loss:.4f}, KL={kl_div:.4f}, avg_reward={avg_reward:.4f}")


##########################################
# 7. 主程序入口
##########################################
if __name__ == "__main__":
    from models.hf_model_history_transformer import RubikSeq2SeqConfig, RubikSeq2SeqForConditionalGeneration

    train_dataset = RubikDataset(data_dir='rubik_shards', history_len=8)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn
    )
    train_dataloader_iter = iter(train_dataloader)

    config = RubikSeq2SeqConfig(
        input_dim=55,
        d_model=256,
        nhead=8,
        num_layers=6,
        num_moves=21,
        max_seq_len=50,
        dropout=0.2
    )
    actor = RubikSeq2SeqForConditionalGeneration(config).to(device)
    state_dict = torch.load("rubik_model_best.pth", map_location=device)
    actor.load_state_dict(state_dict)
    critic = Critic(config.d_model).to(device)
    optimizer_actor = optim.Adam(actor.parameters(), lr=1e-5)
    optimizer_critic = optim.Adam(critic.parameters(), lr=1e-5)

    ppo_train(actor, critic, optimizer_actor, optimizer_critic,
              num_iterations=100, rollout_len=5, batch_size=16)
