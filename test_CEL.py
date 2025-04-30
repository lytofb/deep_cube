import torch
import torch.nn as nn

def test_one_step_loss():
    B = 4           # batch size
    num_moves = 10  # 假设动作类别数

    # 构造随机 logits 和目标标签
    torch.manual_seed(0)
    logits = torch.randn(B, num_moves, requires_grad=True)  # (B, num_moves)
    target = torch.randint(0, num_moves, (B,))              # (B,)

    # CrossEntropyLoss 要求 logits 形状 (B, C)，target 形状 (B,)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, target)

    # 检查 loss 是标量
    assert loss.dim() == 0, f"Expected scalar loss, but got loss.dim()={loss.dim()}"

    # 检查能正确反向传播
    loss.backward()
    assert logits.grad is not None, "Expected gradients for logits, but got None"

    # 打印结果以供直观验证
    print(f"Loss value: {loss.item():.4f}")
    print(f"Gradient sample: {logits.grad[0]}")

def test_pkl():
    import pickle
    from collections import Counter
    gtcounter = Counter()
    f = open("first_mismatch_records.pkl", 'rb')  # pickle_data_path为.pickle文件的路径；
    info = pickle.load(f)
    for record in info:
        gtcounter[record["gt"][0]] += 1
    print(gtcounter)
    print(sum(gtcounter.values()))
    f.close()  # 别忘记close pickle文件

if __name__ == "__main__":
    # CELoss
    # Counter({11: 117, 8: 96, 2: 60, 0: 35, 17: 34, 1: 22, 5: 21, 3: 9, 4: 6})

    # CELoss DistributedWeightedSampler
    # Counter({17: 81, 11: 61, 8: 36, 2: 9, 3: 6, 4: 5, 0: 5, 1: 3, 5: 2}) 208

    # FocalLoss DistributedWeightedSampler
    # Counter({8: 125, 17: 89, 11: 69, 2: 48, 0: 11, 5: 7, 4: 4, 1: 3, 3: 3})

    # FocalLoss
    # Counter({19: 191, 8: 131, 11: 123, 17: 110, 2: 63, 5: 46, 0: 30, 1: 17, 3: 7, 4: 6})

    # FocalLoss MLP
    # Counter({19: 182, 8: 132, 17: 105, 11: 85, 2: 65, 0: 28, 5: 24, 3: 19, 1: 12, 4: 8})

    # CELoss MLP
    # Counter({19: 183, 11: 101, 2: 68, 17: 64, 5: 61, 8: 38, 0: 33, 3: 29, 4: 20, 1: 14})

    # CELoss DistributedWeightedSampler MLP
    # Counter({11: 125, 8: 115, 17: 104, 2: 60, 5: 54, 0: 17, 1: 14, 3: 2, 4: 1}) 492

    # CELoss DistributedWeightedSampler Prompt
    # Counter({8: 132, 11: 118, 17: 70, 2: 55, 3: 8, 1: 5, 0: 5, 5: 5, 14: 2, 4: 2}) 402

    # CELoss DistributedWeightedSampler MLP Dropout //2
    # Counter({11: 101, 8: 56, 2: 46, 0: 10, 1: 5, 17: 4, 3: 2, 14: 1}) 225

    # CELoss DistributedWeightedSampler MLP Dropout //1
    # Counter({11: 92, 2: 52, 17: 12, 5: 10, 0: 8, 8: 5, 3: 3, 1: 2, 4: 1}) 185

    # CELoss DistributedWeightedSampler position scale factor MLP Dropout //1
    # Counter({8: 42, 11: 42, 2: 19, 5: 18, 17: 12, 0: 11, 1: 5, 3: 4, 14: 1, 4: 1}) 155


    # {'U': 0, "U'": 1, 'U2': 2, 'D': 3, "D'": 4, 'D2': 5,
    #  'L': 6, "L'": 7, 'L2': 8, 'R': 9, "R'": 10, 'R2': 11,
    #  'F': 12,"F'": 13, 'F2': 14, 'B': 15, "B'": 16, 'B2': 17}
    test_pkl()
    # test_one_step_loss()
    # print("✅ test_one_step_loss passed")
