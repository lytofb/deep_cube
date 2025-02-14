import random
import pickle
import pycuber as pc
from pycuber.solver import CFOPSolver

if __name__ == "__main__":
    # 尝试保存到本地 (pickle)
    filename = "rubik_shards/rubik_data.pkl"

    # 再尝试读取回来看是否成功
    with open(filename, "rb") as f:
        loaded_data = pickle.load(f)
        print("Loaded")
    print(f"从 {filename} 成功加载了 {len(loaded_data)} 条数据。")