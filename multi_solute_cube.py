import os
import pickle
import random
import pycuber as pc
import multiprocessing

MOVES_POOL = [
    'U', 'U\'', 'U2', 'D', 'D\'', 'D2',
    'L', 'L\'', 'L2', 'R', 'R\'', 'R2',
    'F', 'F\'', 'F2', 'B', 'B\'', 'B2'
]

def inverse_move(move_str):
    """将 'R' 转为 'R'', 'R'' 转为 'R', 'R2' 不变。"""
    if move_str.endswith('2'):
        return move_str
    elif move_str.endswith('\''):
        return move_str[:-1]
    else:
        return move_str + '\''

def cube_to_6x9(cube):
    """将魔方序列化为 6×9 的二维数组，并去掉 '[ ]'。"""
    face_order = ['U', 'L', 'F', 'R', 'B', 'D']
    res = []
    for face_name in face_order:
        face_obj = cube.get_face(face_name)
        row_data = []
        for r in range(3):
            for c in range(3):
                raw_sticker = str(face_obj[r][c])  # 如 "[g]"
                color_char = raw_sticker.strip('[]')
                row_data.append(color_char)
        res.append(row_data)
    return res

def generate_single_case(min_scramble=1, max_scramble=25, seed_offset=0):
    """
    生成单条数据, 不使用求解器(用随机打乱+逆操作).
    seed_offset 可用于多机/多进程区分随机种子, 避免重复.
    """
    cube = pc.Cube()

    k = random.randint(min_scramble, max_scramble)
    scramble_ops = [random.choice(MOVES_POOL) for _ in range(k)]
    solution_ops = [inverse_move(m) for m in reversed(scramble_ops)]

    # 正向打乱
    for mv in scramble_ops:
        cube(mv)

    # 记录打乱态 -> 复原态的每一步
    steps = []

    # (b) 最终状态
    steps.append((cube_to_6x9(cube), None))

    # (a) 依次应用 solution_ops, 并记录
    for move in solution_ops:
        cube(move)
        steps.append((cube_to_6x9(cube), move))

    data_item = {
        'steps': steps  # [(6x9二维数组, move or None), ...]
    }
    return data_item

def generate_shard(shard_index, num_samples, out_dir, min_scr=8, max_scr=25, seed_offset=0, flush_size=10000):
    """
    生成一个分片(shard)，共 num_samples 条数据。但为了避免大内存占用，
    每当累积到 flush_size 条时，就把数据dump到磁盘(追加写入)并清空缓存。

    最终所有数据都保存在 out_dir/part_{shard_index:05d}.pkl 中，
    里面是多段pickle, 每段是一个小列表。
    """
    random.seed(shard_index + seed_offset * 1000000)

    shard_file = os.path.join(out_dir, f"part_{shard_index:05d}.pkl")

    # 如果文件已存在先删除，防止重复追加
    if os.path.exists(shard_file):
        os.remove(shard_file)

    buffer_data = []
    count = 0

    for _ in range(num_samples):
        item = generate_single_case(min_scr, max_scr)
        buffer_data.append(item)
        count += 1

        # 到达 flush_size 就写一次
        if count % flush_size == 0:
            with open(shard_file, "ab") as f:
                pickle.dump(buffer_data, f)
            buffer_data.clear()

    # 写出剩余的
    if len(buffer_data) > 0:
        with open(shard_file, "ab") as f:
            pickle.dump(buffer_data, f)
        buffer_data.clear()

    return shard_file

def generate_dataset_multiprocess(
        total_samples=10_000,
        samples_per_shard=100,
        num_processes=4,
        out_dir='output',
        min_scr=8,
        max_scr=25,
        flush_size=10_000
):
    """
    单机多进程+分块 生成大规模数据.
    total_samples: 需生成的总条数
    samples_per_shard: 每个分片包含多少条
    flush_size: 每多少条就往磁盘写一次(防止占用过多内存).
    """
    os.makedirs(out_dir, exist_ok=True)

    # 计算分片数量
    total_shards = (total_samples + samples_per_shard - 1) // samples_per_shard

    # 准备 shard 的参数列表
    shard_params = []
    for shard_idx in range(total_shards):
        shard_params.append(
            (
                shard_idx,          # 分片序号
                samples_per_shard,  # 分片大小
                out_dir,
                min_scr,
                max_scr,
                0,                  # seed_offset
                flush_size
            )
        )

    # 多进程执行
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = [pool.apply_async(generate_shard, p) for p in shard_params]
        pool.close()
        pool.join()

    shard_files = [r.get() for r in results]
    return shard_files

if __name__ == "__main__":
    # 示例: 在本地生成100万条, 分成10个分片, 4进程并行
    # 每个分片(10万条)里， 每1万条就写一次磁盘，避免占用过多内存
    all_shard_files = generate_dataset_multiprocess(
        total_samples=1_000,
        samples_per_shard=100,
        num_processes=4,
        out_dir='rubik_shards',
        min_scr=8,
        max_scr=25,
        flush_size=10_000  # 每1万条写一次
    )
    print("All shard files:", all_shard_files)