import os
import pickle
from collections import Counter

def scan_pkl_stats(data_dir):
    """
    扫描 data_dir 下所有 .pkl，统计：
      1) 每种 move（字符串）的出现次数
      2) 所有 move 的总数
      3) 每个 sample 中，在 SOS_TOKEN 之前（即 src_seq 最后一行）的 move 出现次数
    Args:
        data_dir (str): 存放 .pkl 文件的目录
        history_len (int): 每个 sample 的历史长度（与 Dataset 中保持一致）
    Returns:
        move_counter (Counter): 全部 steps 中 move 的分布
        total_moves (int): move_counter 的总和
        prev_move_counter (Counter): 每个 sample “在 SOS_TOKEN 之前一个 move” 的分布
    """
    move_counter      = Counter()
    first_char_counter = Counter()

    for fname in os.listdir(data_dir):
        if not fname.endswith('.pkl'):
            continue
        full_path = os.path.join(data_dir, fname)
        with open(full_path, 'rb') as f:
            # 一个 .pkl 里可能 dump 多个 list，每次 load 都取出来处理
            while True:
                try:
                    data_list = pickle.load(f)
                except EOFError:
                    break
                for item in data_list:
                    steps = item.get('steps', [])
                    # 1. 累加所有 steps 里的 move
                    for _, mv in steps:
                        move_counter[mv] += 1
                    first_char_counter[steps[1][1]] += 1

    total_moves = sum(move_counter.values())
    return move_counter, total_moves, first_char_counter


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',     type=str, default='rubik_shards',
                        help='pkl 存放目录')
    args = parser.parse_args()

    move_cnt, total, first_char_cnt = scan_pkl_stats(args.data_dir)

    print(f"—— 全部 MOVE 分布（共 {total} 次） ——")
    for mv, cnt in move_cnt.most_common():
        if mv is None:
            continue
        print(f"{mv:>6s} : {cnt}")

    print(f"\n—— 首字符分布 ——")
    for ch, cnt in first_char_cnt.most_common():
        print(f"{ch:>2s} : {cnt}")
