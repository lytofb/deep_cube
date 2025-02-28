import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# 这是你已有的 Dataset 和 collate_fn
from dataset_rubik import RubikDataset, collate_fn

# PPO相关
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer

# =============================
# 1. 数据集与 DataLoader
# =============================
def build_dataloader(
    data_dir,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    max_files=None
):
    dataset = RubikDataset(data_dir=data_dir, max_files=max_files)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return loader


# =============================
# 2. 加载已有的基础模型
#    （假设已转换到 HF 格式）
# =============================
def load_pretrained_model_and_tokenizer(model_path: str, device: str = "cuda"):
    """
    这里示例直接用 from_pretrained() 加载一个已转换好的模型。
    如果你的模型是自定义的 RubikSeq2SeqTransformer，需要手动封装/转换。
    """

    # 加载带价值头的模型
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path)

    # 放到指定设备
    model = model.to(device)

    return model


# =============================
# 3. 定义一个奖励函数
# =============================
def compute_reward(predictions, labels):
    """
    predictions: List[str], 模型生成的动作序列或token序列(已decode)
    labels:      List[str], 真实的/目标的动作序列或token序列(已decode)
    return:      List[float], 长度与batch相同
    """
    rewards = []
    for pred, lab in zip(predictions, labels):
        # 简单示例：完全匹配则+1，不匹配则-1
        # （实际可结合魔方“还原度”或启发式判断）
        if pred == lab:
            rewards.append(1.0)
        else:
            rewards.append(-1.0)
    return rewards


# =============================
# 4. PPO 训练过程
# =============================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 4.1 数据准备 ----
    # 参考 train.py 里的路径、参数
    train_dir = "rubik_train_shards"  # 或者 config.data.train_dir
    val_dir   = "rubik_val_shards"

    # 这里设置一个合理的 batch_size
    ppo_batch_size = 16
    num_workers = 4
    train_loader = build_dataloader(
        data_dir=train_dir,
        batch_size=ppo_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    # ---- 4.2 模型 & Tokenizer 加载 ----
    # 这里假设你已经将 "rubik_model_best" 转换到 HF 格式，并能用 from_pretrained() 加载
    model_path = "path/to/rubik_model_best_converted"
    model = load_pretrained_model_and_tokenizer(model_path, device=device)

    # ---- 4.3 PPO 配置与初始化 ----
    ppo_config = PPOConfig(
        steps=1024,             # 每个 epoch 的最大训练 step（可根据数据大小做调整）
        batch_size=16,          # 一次 PPO step 用多少条数据
        forward_batch_size=8,   # 前向计算时使用的 mini-batch 大小
        lr=1e-5,
        log_with=None,          # 若想要 tensorboard 或 wandb，可改成 'tensorboard'/'wandb'
        optimize_cuda_usage=True,
        # 其他可选超参: gamma, gae_lambda, cliprange, etc.
    )

    # 如果需要参考模型 ref_model 做KL惩罚，可在这里加载
    #   ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path).to(device)
    #   ppo_trainer = PPOTrainer(config=ppo_config, model=model, ref_model=ref_model, tokenizer=tokenizer)
    # 本例中先不使用参考模型
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=None,
    )

    # ---- 4.4 定义 PPO 训练循环 ----
    num_epochs = 2  # 可以根据实际需要调整
    step_count = 0  # 累计 step 数
    for epoch in range(num_epochs):
        print(f"===== PPO EPOCH {epoch+1} / {num_epochs} =====")

        for batch_idx, (src, tgt) in enumerate(train_loader):
            # 注：train_loader 返回 (src, tgt)，参考 train.py 里的 collate_fn
            # src, tgt 的形状: (B, seq_len)

            # 1) 先把 src/tgt 转成“可供 PPOTrainer 使用”的输入：
            #    可以把 `src` 序列当做 queries；把 `tgt` (或其下一步动作)当做 labels。
            #    这里演示：把 src 直接 decode 作为 prompt，tgt 也 decode 作为 ground truth。
            #    具体视你的魔方动作如何编码而定。
            src = src.to(device)
            tgt = tgt.to(device)

            # decode => List[str]
            # 假设 collate_fn 里直接给的是 token id（兼容 tokenizer），
            # 否则需要你自己的 state->text / action->text 转换。
            queries_text = tokenizer.batch_decode(src, skip_special_tokens=True)
            labels_text  = tokenizer.batch_decode(tgt, skip_special_tokens=True)

            # 2) 让模型生成回复(动作/预测token)。在 PPO 场景下，一般用 sampling。
            #    也可以限制 max_new_tokens = (长度)，top_k, top_p 等
            query_tensors = tokenizer(
                queries_text, padding=True, truncation=True, return_tensors="pt"
            ).input_ids.to(device)

            generation_outputs = model.generate(
                query_tensors,
                max_new_tokens=10,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )

            # 3) 解码模型预测
            pred_texts = tokenizer.batch_decode(generation_outputs, skip_special_tokens=True)

            # 4) 计算 Reward
            rewards = compute_reward(pred_texts, labels_text)

            # 5) 使用 PPOTrainer 的 step() 更新
            #    先把“新生成的部分”从 generation_outputs 中分割出来
            response_tensors = []
            for q_tensor, g_output in zip(query_tensors, generation_outputs):
                # 仅保留 q_tensor 之后的新生成 tokens 作为 response
                response_part = g_output[len(q_tensor):]
                response_tensors.append(response_part)

            rewards_tensors = torch.tensor(rewards, dtype=torch.float, device=device)

            # ppo_trainer.step() 会执行一次基于 PPO 的后向与更新
            stats = ppo_trainer.step(
                queries=query_tensors,
                responses=response_tensors,
                rewards=rewards_tensors
            )

            step_count += 1
            if step_count % 50 == 0:
                print(f"Epoch={epoch+1}, Step={step_count}, PPO stats={stats}")

            # 当 step_count 接近 ppo_config.steps 时，可以考虑退出本 epoch
            # 或者你想完整遍历完一个 epoch 的所有 batch 也行
            if step_count >= ppo_config.steps:
                break

        # 每个 epoch 结束后可保存模型
        save_path = f"ppo_rubik_model_epoch_{epoch+1}"
        model.save_pretrained(save_path)
        print(f"已保存 PPO 更新后的模型到 {save_path}")

        # 如果想要继续训练下一个 epoch，可以在此重置 step_count 或让其累加
        # 具体看你想如何分配 steps vs. epochs
        # 这里简单演示，继续往下累加 steps

    print("PPO 微调完成！")


if __name__ == "__main__":
    main()
