import copy

def rotate_U(cube):
    """
    对魔方状态进行 U 转动。
    cube: list of 6 lists，每个子列表长度为 9，顺序为 [U, L, F, R, B, D]，
          每个面的元素按照行顺序排列。
    返回：转动后的魔方状态（新的 cube 副本）。
    """
    # 复制原状态（深复制）
    new_cube = [face[:] for face in cube]

    # 1. 转动 U 面（上面）本身，顺时针旋转
    old_U = cube[0]
    new_cube[0][0] = old_U[6]
    new_cube[0][1] = old_U[3]
    new_cube[0][2] = old_U[0]
    new_cube[0][3] = old_U[7]
    new_cube[0][4] = old_U[4]
    new_cube[0][5] = old_U[1]
    new_cube[0][6] = old_U[8]
    new_cube[0][7] = old_U[5]
    new_cube[0][8] = old_U[2]

    # 2. 处理相邻面的上排交换：
    # 注意：按照魔方展开图顺序，cube[1]是L, cube[2]是F, cube[3]是R, cube[4]是B
    # 保存原始上排（避免直接覆盖导致数据丢失）
    L_top = cube[1][0:3]
    F_top = cube[2][0:3]
    R_top = cube[3][0:3]
    B_top = cube[4][0:3]

    # 根据给定映射进行交换
    new_cube[1][0:3] = F_top      # F 上行 -> L 上行
    new_cube[2][0:3] = R_top      # R 上行 -> F 上行
    new_cube[3][0:3] = B_top      # B 上行 -> R 上行
    new_cube[4][0:3] = L_top      # L 上行 -> B 上行

    return new_cube


def rotate_L(cube):
    """
    对魔方状态进行 L 转动（左侧面顺时针转动）。

    参数:
      cube: list of 6 lists，每个子列表长度为 9，顺序为 [U, L, F, R, B, D]，
            每个面的元素按照行优先顺序排列。

    返回:
      转动后的魔方状态（新状态，不直接修改原始 cube）。

    转动规则:
      1. L 面自身顺时针旋转
      2. 更新相邻面的对应列：
         - B 面第三列（索引 [2, 5, 8]，上到下） → U 面第一列（索引 [0, 3, 6]），但顺序反转
         - U 面第一列 → F 面第一列（均为索引 [0, 3, 6]，顺序不变）
         - F 面第一列 → D 面第一列（索引 [0, 3, 6]）
         - D 面第一列 → B 面第三列，顺序反转
    """
    # 深复制魔方状态，避免直接修改原状态
    new_cube = [face[:] for face in cube]

    # 1. 对 L 面（左侧面，索引 1）进行顺时针旋转
    old_L = cube[1]
    new_cube[1][0] = old_L[6]
    new_cube[1][1] = old_L[3]
    new_cube[1][2] = old_L[0]
    new_cube[1][3] = old_L[7]
    new_cube[1][4] = old_L[4]
    new_cube[1][5] = old_L[1]
    new_cube[1][6] = old_L[8]
    new_cube[1][7] = old_L[5]
    new_cube[1][8] = old_L[2]

    # 2. 更新相邻四个面的对应列
    # U 面（上面）第一列，索引：0, 3, 6
    U_left = [cube[0][0], cube[0][3], cube[0][6]]
    # F 面（前面）第一列，索引：0, 3, 6
    F_left = [cube[2][0], cube[2][3], cube[2][6]]
    # D 面（下面）第一列，索引：0, 3, 6
    D_left = [cube[5][0], cube[5][3], cube[5][6]]
    # B 面（后面）第三列，索引：2, 5, 8
    B_right = [cube[4][2], cube[4][5], cube[4][8]]

    # B 面第三列 → U 面第一列（顺序反转）
    new_cube[0][0] = B_right[2]
    new_cube[0][3] = B_right[1]
    new_cube[0][6] = B_right[0]

    # U 面第一列 → F 面第一列（顺序不变）
    new_cube[2][0] = U_left[0]
    new_cube[2][3] = U_left[1]
    new_cube[2][6] = U_left[2]

    # F 面第一列 → D 面第一列（顺序不变）
    new_cube[5][0] = F_left[0]
    new_cube[5][3] = F_left[1]
    new_cube[5][6] = F_left[2]

    # D 面第一列 → B 面第三列（顺序反转）
    new_cube[4][2] = D_left[2]
    new_cube[4][5] = D_left[1]
    new_cube[4][8] = D_left[0]

    return new_cube


def rotate_F(cube):
    """
    对魔方状态进行 F 转动（前面顺时针转动）。

    参数:
      cube: list of 6 lists，每个子列表长度为 9，顺序为 [U, L, F, R, B, D]，
            每个面的元素按照行优先顺序排列。

    转动规则：
      1. F 面自身顺时针旋转。
      2. 相邻边的更新：
         - U 面的底行 (索引 6,7,8) -> R 面的左列 (索引 0,3,6)
         - R 面的左列 (索引 0,3,6) -> D 面的顶行，按照 D 面的索引 [2,1,0]（即 D210）
         - D 面的顶行（按 D210 顺序）-> L 面的右列 (索引 2,5,8)
         - L 面的右列 (索引 2,5,8) -> U 面的底行 (索引 6,7,8)
    """
    # 深复制魔方状态，避免直接修改原状态
    new_cube = [face[:] for face in cube]

    # 1. 对 F 面（索引 2）自身进行顺时针旋转
    old_F = cube[2]
    new_cube[2][0] = old_F[6]
    new_cube[2][1] = old_F[3]
    new_cube[2][2] = old_F[0]
    new_cube[2][3] = old_F[7]
    new_cube[2][4] = old_F[4]
    new_cube[2][5] = old_F[1]
    new_cube[2][6] = old_F[8]
    new_cube[2][7] = old_F[5]
    new_cube[2][8] = old_F[2]

    # 2. 保存相邻 4 个面的相关边数据
    # U 面底行（索引 6,7,8）
    U_bottom = cube[0][6:9]

    # R 面左列（索引 0,3,6）
    R_left = [cube[3][0], cube[3][3], cube[3][6]]

    # D 面顶行，按照 D210 的顺序，即 D 面原先的索引 0,1,2，但按 [2,1,0] 顺序保存
    D_top_reversed = [cube[5][2], cube[5][1], cube[5][0]]

    # L 面右列（索引 2,5,8）
    L_right = [cube[1][2], cube[1][5], cube[1][8]]

    # 3. 执行边的转换
    # U 面底行 -> R 面左列
    new_cube[3][0] = U_bottom[0]
    new_cube[3][3] = U_bottom[1]
    new_cube[3][6] = U_bottom[2]

    # R 面左列 -> D 面顶行（D210 顺序）
    new_cube[5][2] = R_left[0]
    new_cube[5][1] = R_left[1]
    new_cube[5][0] = R_left[2]

    # D 面顶行（D210 顺序） -> L 面右列
    new_cube[1][2] = D_top_reversed[2]
    new_cube[1][5] = D_top_reversed[1]
    new_cube[1][8] = D_top_reversed[0]

    # L 面右列 -> U 面底行
    new_cube[0][6] = L_right[0]
    new_cube[0][7] = L_right[1]
    new_cube[0][8] = L_right[2]

    return new_cube


def rotate_R(cube):
    """
    对魔方状态进行 R 转动（右侧面顺时针转动）。

    参数:
      cube: list of 6 lists，每个子列表长度为 9，顺序为 [U, L, F, R, B, D]，
            每个面的元素按照行优先顺序排列。

    转动规则：
      1. R 面自身顺时针旋转。
      2. 边的交换：
         - U 面右列 U258 (cube[0][2], cube[0][5], cube[0][8])
           -> 赋值到 B 面对应位置 B630 (cube[4][6], cube[4][3], cube[4][0])
         - B 面左列 B036 (cube[4][0], cube[4][3], cube[4][6])
           -> 赋值到 D 面右列 D852 (cube[5][8], cube[5][5], cube[5][2])
         - D 面右列 D852 (cube[5][8], cube[5][5], cube[5][2])
           -> 赋值到 F 面右列 F852 (cube[2][8], cube[2][5], cube[2][2])
         - F 面右列 F852 (cube[2][8], cube[2][5], cube[2][2])
           -> 赋值到 U 面右列 U852 (cube[0][8], cube[0][5], cube[0][2])
    """
    # 深复制魔方状态，防止修改原始数据
    new_cube = [face[:] for face in cube]

    # 1. 对 R 面（face index 3）进行顺时针旋转
    old_R = cube[3]
    new_cube[3][0] = old_R[6]
    new_cube[3][1] = old_R[3]
    new_cube[3][2] = old_R[0]
    new_cube[3][3] = old_R[7]
    new_cube[3][4] = old_R[4]
    new_cube[3][5] = old_R[1]
    new_cube[3][6] = old_R[8]
    new_cube[3][7] = old_R[5]
    new_cube[3][8] = old_R[2]

    # 2. 保存相邻边的数据（均从原始状态 cube 中提取，防止覆盖）
    # U 面右列 U258：索引 2,5,8
    U_edge = [cube[0][2], cube[0][5], cube[0][8]]
    # B 面左列 B036：索引 0,3,6
    B_edge = [cube[4][0], cube[4][3], cube[4][6]]
    # D 面右列 D852：索引 8,5,2
    D_edge = [cube[5][8], cube[5][5], cube[5][2]]
    # F 面右列 F852：索引 8,5,2
    F_edge = [cube[2][8], cube[2][5], cube[2][2]]

    # 3. 边的交换
    # U258 -> B630：将 U_edge 赋值到 B 面对应位置 B630
    new_cube[4][6] = U_edge[0]
    new_cube[4][3] = U_edge[1]
    new_cube[4][0] = U_edge[2]

    # B036 -> D852：将 B_edge 赋值到 D 面右列 D852
    new_cube[5][8] = B_edge[0]
    new_cube[5][5] = B_edge[1]
    new_cube[5][2] = B_edge[2]

    # D852 -> F852：将 D_edge 赋值到 F 面右列 F852
    new_cube[2][8] = D_edge[0]
    new_cube[2][5] = D_edge[1]
    new_cube[2][2] = D_edge[2]

    # F852 -> U852：将 F_edge 赋值到 U 面右列 U852
    new_cube[0][8] = F_edge[0]
    new_cube[0][5] = F_edge[1]
    new_cube[0][2] = F_edge[2]

    return new_cube

def rotate_B(cube):
    """
    对魔方状态进行 B 转动（后面顺时针转动）。

    参数:
      cube: list of 6 lists，每个子列表长度为 9，
            顺序为 [U, L, F, R, B, D]，每个面的元素按行顺序排列。

    转动规则：
      1. B 面自身顺时针旋转。
      2. 边的转换：
         - U 面上行 u012 (索引 [0,1,2]) -> L 面 l630 (索引 [6,3,0])
         - L 面 l036 (索引 [0,3,6]) -> D 面上行 d678 (索引 [6,7,8])
         - D 面 d678 (索引 [6,7,8]) -> R 面 r852 (索引 [8,5,2])
         - R 面 r852 (索引 [8,5,2]) -> U 面 u210 (索引 [2,1,0])
    """
    # 深复制魔方状态，防止修改原始数据
    new_cube = [face[:] for face in cube]

    # 1. 对 B 面自身（face index 4）进行顺时针旋转
    old_B = cube[4]
    new_cube[4][0] = old_B[6]
    new_cube[4][1] = old_B[3]
    new_cube[4][2] = old_B[0]
    new_cube[4][3] = old_B[7]
    new_cube[4][4] = old_B[4]
    new_cube[4][5] = old_B[1]
    new_cube[4][6] = old_B[8]
    new_cube[4][7] = old_B[5]
    new_cube[4][8] = old_B[2]

    # 2. 边的转换

    # (1) U 面上行 u012：face[0] 的索引 0,1,2
    U_top = cube[0][0:3]  # u012

    # (2) L 面两组边：
    # l630：L 面（face index 1）索引 [6,3,0]，将用于接收 U_top
    # l036：L 面索引 [0,3,6]，用于传递给 D 面
    L_edge = [cube[1][0], cube[1][3], cube[1][6]]  # l036

    # (3) D 面上边 d678：D 面（face index 5）索引 [6,7,8]
    D_top = cube[5][6:9]  # d678

    # (4) R 面对应边 r852：R 面（face index 3）索引 [8,5,2]
    R_edge = [cube[3][8], cube[3][5], cube[3][2]]  # r852

    # 按照转换规则执行赋值：
    # U u012 -> L l630
    new_cube[1][6] = U_top[0]
    new_cube[1][3] = U_top[1]
    new_cube[1][0] = U_top[2]

    # L l036 -> D d678
    new_cube[5][6] = L_edge[0]
    new_cube[5][7] = L_edge[1]
    new_cube[5][8] = L_edge[2]

    # D d678 -> R r852
    new_cube[3][8] = D_top[0]
    new_cube[3][5] = D_top[1]
    new_cube[3][2] = D_top[2]

    # R r852 -> U u210 (即 U 面索引 [2,1,0])
    new_cube[0][2] = R_edge[0]
    new_cube[0][1] = R_edge[1]
    new_cube[0][0] = R_edge[2]

    return new_cube

def rotate_D(cube):
    """
    对魔方状态进行 D 转动（下层顺时针转动）。

    参数:
      cube: list of 6 lists，每个子列表长度为 9，
            顺序为 [U, L, F, R, B, D]，每个面的元素按行顺序排列。

    转动规则：
      1. D 面自身顺时针旋转。
      2. 相邻面底行的交换：
         - F 面底行 (索引 6,7,8) -> R 面底行 (索引 6,7,8)
         - R 面底行 -> B 面底行
         - B 面底行 -> L 面底行
         - L 面底行 -> F 面底行
    """
    # 深复制魔方状态，避免修改原始数据
    new_cube = [face[:] for face in cube]

    # 1. 对 D 面（face index 5）自身进行顺时针旋转
    old_D = cube[5]
    new_cube[5][0] = old_D[6]
    new_cube[5][1] = old_D[3]
    new_cube[5][2] = old_D[0]
    new_cube[5][3] = old_D[7]
    new_cube[5][4] = old_D[4]
    new_cube[5][5] = old_D[1]
    new_cube[5][6] = old_D[8]
    new_cube[5][7] = old_D[5]
    new_cube[5][8] = old_D[2]

    # 2. 处理相邻面的底行交换（索引 6:9）
    # 保存原始底行数据
    F_bottom = cube[2][6:9]  # F 面底行
    R_bottom = cube[3][6:9]  # R 面底行
    B_bottom = cube[4][6:9]  # B 面底行
    L_bottom = cube[1][6:9]  # L 面底行

    # D 转动时，边的交换规则为：
    # F 底行 -> R 底行
    new_cube[3][6:9] = F_bottom
    # R 底行 -> B 底行
    new_cube[4][6:9] = R_bottom
    # B 底行 -> L 底行
    new_cube[1][6:9] = B_bottom
    # L 底行 -> F 底行
    new_cube[2][6:9] = L_bottom

    return new_cube

def rotate_D_prime(cube):
    """
    对魔方状态进行 D' 转动（下层逆时针转动）。

    参数:
      cube: list of 6 lists，每个子列表长度为 9，
            顺序为 [U, L, F, R, B, D]，每个面的元素按行顺序排列。

    转动规则：
      1. D 面（下层）自身逆时针旋转，使用如下映射：
         new[0] = old[2]
         new[1] = old[5]
         new[2] = old[8]
         new[3] = old[1]
         new[4] = old[4]
         new[5] = old[7]
         new[6] = old[0]
         new[7] = old[3]
         new[8] = old[6]
      2. 相邻面底行的交换（只影响 F、R、B、L 四个面）：
         新 F 底行 = 原 R 底行
         新 R 底行 = 原 B 底行
         新 B 底行 = 原 L 底行
         新 L 底行 = 原 F 底行
    """
    # 深复制魔方状态，避免直接修改原始数据
    new_cube = [face[:] for face in cube]

    # 1. 对 D 面（face index 5）自身进行逆时针旋转
    old_D = cube[5]
    new_cube[5][0] = old_D[2]
    new_cube[5][1] = old_D[5]
    new_cube[5][2] = old_D[8]
    new_cube[5][3] = old_D[1]
    new_cube[5][4] = old_D[4]
    new_cube[5][5] = old_D[7]
    new_cube[5][6] = old_D[0]
    new_cube[5][7] = old_D[3]
    new_cube[5][8] = old_D[6]

    # 2. 保存相邻面的底行数据
    # F 面底行（face index 2）索引 6,7,8
    F_bottom = cube[2][6:9]
    # R 面底行（face index 3）索引 6,7,8
    R_bottom = cube[3][6:9]
    # B 面底行（face index 4）索引 6,7,8
    B_bottom = cube[4][6:9]
    # L 面底行（face index 1）索引 6,7,8
    L_bottom = cube[1][6:9]

    # 3. 按 D' 的逆时针交换规则更新相邻面底行：
    # 新 F 底行 = 原 R 底行
    new_cube[2][6:9] = R_bottom
    # 新 R 底行 = 原 B 底行
    new_cube[3][6:9] = B_bottom
    # 新 B 底行 = 原 L 底行
    new_cube[4][6:9] = L_bottom
    # 新 L 底行 = 原 F 底行
    new_cube[1][6:9] = F_bottom

    return new_cube

def rotate_B_prime(cube):
    """
    对魔方状态进行 B′ 转动（后面逆时针转动）。

    参数:
      cube: list of 6 lists，每个子列表长度为 9，
            顺序为 [U, L, F, R, B, D]，每个面的元素按行顺序排列。

    转动规则：
      1. B 面自身逆时针旋转：
         new[0] = old[2]
         new[1] = old[5]
         new[2] = old[8]
         new[3] = old[1]
         new[4] = old[4]
         new[5] = old[7]
         new[6] = old[0]
         new[7] = old[3]
         new[8] = old[6]
      2. 边块交换（按逆时针循环）：
         - 新 U 面上行 u012 = 原 L 面 l630 (即 [L[6], L[3], L[0]])
         - 新 L 面 l036 = 原 D 面上行 d678 (即 [D[6], D[7], D[8]])
         - 新 D 面上行 d678 = 原 R 面 r852 (即 [R[8], R[5], R[2]])
         - 新 R 面 r852 = 原 U 面 u012 (即 [U[2], U[1], U[0]])
    """
    # 深复制魔方状态，避免直接修改原始数据
    new_cube = [face[:] for face in cube]

    # 1. 对 B 面（face index 4）进行逆时针旋转
    old_B = cube[4]
    new_cube[4][0] = old_B[2]
    new_cube[4][1] = old_B[5]
    new_cube[4][2] = old_B[8]
    new_cube[4][3] = old_B[1]
    new_cube[4][4] = old_B[4]
    new_cube[4][5] = old_B[7]
    new_cube[4][6] = old_B[0]
    new_cube[4][7] = old_B[3]
    new_cube[4][8] = old_B[6]

    # 2. 保存相关边数据（均从原始状态 cube 中提取）
    # U 面上行 u012（face index 0，索引 0,1,2）
    U_top = cube[0][0:3]
    # L 面 l630（face index 1，索引 6,3,0）
    L_l630 = [cube[1][6], cube[1][3], cube[1][0]]
    # D 面上行 d678（face index 5，索引 6,7,8）
    D_top = cube[5][6:9]
    # R 面 r852（face index 3，索引 8,5,2）
    R_r852 = [cube[3][8], cube[3][5], cube[3][2]]

    # 3. 边块的逆向交换
    # 新 U 面上行 u012 = 原 L 面 l630
    new_cube[0][0] = L_l630[0]  # U[0] = L[6]
    new_cube[0][1] = L_l630[1]  # U[1] = L[3]
    new_cube[0][2] = L_l630[2]  # U[2] = L[0]

    # 新 L 面 l036 = 原 D 面上行 d678
    # l036 指的是 L 面索引 [0,3,6]
    new_cube[1][0] = D_top[0]   # L[0] = D[6]
    new_cube[1][3] = D_top[1]   # L[3] = D[7]
    new_cube[1][6] = D_top[2]   # L[6] = D[8]

    # 新 D 面上行 d678 = 原 R 面 r852
    # D 面 d678 为索引 [6,7,8]
    new_cube[5][6] = R_r852[0]  # D[6] = R[8]
    new_cube[5][7] = R_r852[1]  # D[7] = R[5]
    new_cube[5][8] = R_r852[2]  # D[8] = R[2]

    # 新 R 面 r852 = 原 U 面 u012
    # r852 为 R 面索引 [8,5,2]，取自 U 面 u012但注意顺序：新 R[8] = U[2], 新 R[5] = U[1], 新 R[2] = U[0]
    new_cube[3][8] = U_top[2]
    new_cube[3][5] = U_top[1]
    new_cube[3][2] = U_top[0]

    return new_cube

def rotate_R_prime(cube):
    """
    对魔方状态进行 R' 转动（右侧面逆时针转动）。

    参数:
      cube: list of 6 lists，每个子列表长度为 9，
            顺序为 [U, L, F, R, B, D]，
            每个面的元素按照行优先顺序排列。

    转动规则（为 rotate_R 的逆操作）：
      1. R 面自身逆时针旋转：
         new[0] = old[2]
         new[1] = old[5]
         new[2] = old[8]
         new[3] = old[1]
         new[4] = old[4]
         new[5] = old[7]
         new[6] = old[0]
         new[7] = old[3]
         new[8] = old[6]
      2. 边块交换（逆转 rotate_R 的边交换循环）：
         - 新 U 面右列 (U[8], U[5], U[2]) = 原 F 面右列 (F[8], F[5], F[2])
         - 新 F 面右列 (F[8], F[5], F[2]) = 原 D 面右列 (D[8], D[5], D[2])
         - 新 D 面右列 (D[8], D[5], D[2]) = 原 B 面左列 (B[0], B[3], B[6])
         - 新 B 面左列 (B[0], B[3], B[6]) = 原 U 面右列 (U[8], U[5], U[2])
    """
    # 深复制魔方状态，避免修改原始数据
    new_cube = [face[:] for face in cube]

    # 1. R 面自身逆时针旋转
    old_R = cube[3]
    new_cube[3][0] = old_R[2]
    new_cube[3][1] = old_R[5]
    new_cube[3][2] = old_R[8]
    new_cube[3][3] = old_R[1]
    new_cube[3][4] = old_R[4]
    new_cube[3][5] = old_R[7]
    new_cube[3][6] = old_R[0]
    new_cube[3][7] = old_R[3]
    new_cube[3][8] = old_R[6]

    # 2. 保存相邻边数据
    # U 面右列：索引 2,5,8
    U_edge = [cube[0][2], cube[0][5], cube[0][8]]
    # F 面右列：按照 rotate_R 中定义 F_edge = [cube[2][8], cube[2][5], cube[2][2]]
    F_edge = [cube[2][8], cube[2][5], cube[2][2]]
    # D 面右列：定义为 D_edge = [cube[5][8], cube[5][5], cube[5][2]]
    D_edge = [cube[5][8], cube[5][5], cube[5][2]]
    # B 面左列：定义为 B_edge = [cube[4][0], cube[4][3], cube[4][6]]
    B_edge = [cube[4][0], cube[4][3], cube[4][6]]

    # 3. 边块交换（逆向循环）
    # 新 U 面右列 = 原 F 面右列
    new_cube[0][8] = B_edge[0]  # U[8] = F[8]
    new_cube[0][5] = B_edge[1]  # U[5] = F[5]
    new_cube[0][2] = B_edge[2]  # U[2] = F[2]

    # 新 F 面右列 = 原 D 面右列
    new_cube[2][2] = U_edge[0]  # F[8] = D[8]
    new_cube[2][5] = U_edge[1]  # F[5] = D[5]
    new_cube[2][8] = U_edge[2]  # F[2] = D[2]

    # 新 D 面右列 = 原 B 面左列
    new_cube[5][8] = F_edge[0]  # D[8] = B[0]
    new_cube[5][5] = F_edge[1]  # D[5] = B[3]
    new_cube[5][2] = F_edge[2]  # D[2] = B[6]

    # 新 B 面左列 = 原 U 面右列
    new_cube[4][0] = D_edge[0]  # B[0] = U[8]
    new_cube[4][3] = D_edge[1]  # B[3] = U[5]
    new_cube[4][6] = D_edge[2]  # B[6] = U[2]

    return new_cube

def rotate_F_prime(cube):
    """
    对魔方状态进行 F' 转动（前面逆时针转动）。

    参数:
      cube: list of 6 lists，每个子列表长度为 9，
            顺序为 [U, L, F, R, B, D]，每个面的元素按照行优先顺序排列。

    转动规则：
      1. F 面自身逆时针旋转，使用如下映射：
         new[0] = old[2]
         new[1] = old[5]
         new[2] = old[8]
         new[3] = old[1]
         new[4] = old[4]
         new[5] = old[7]
         new[6] = old[0]
         new[7] = old[3]
         new[8] = old[6]
      2. 相邻边的更新（逆转 rotate_F 的循环）：
         - 新 U 面底行 (索引 6,7,8) = 原 R 面左列 (索引 0,3,6)
         - 新 R 面左列 (索引 0,3,6) = 原 D 面顶行（按照 D210 顺序存储的那一份，即 D_top_reversed）
         - 新 D 面顶行（按 D210 顺序） = 原 L 面右列的逆序
         - 新 L 面右列 (索引 2,5,8) = 原 U 面底行 (索引 6,7,8)
    """
    # 深复制魔方状态，避免修改原始数据
    new_cube = [face[:] for face in cube]

    # 1. 对 F 面（face index 2）自身进行逆时针旋转
    old_F = cube[2]
    new_cube[2][0] = old_F[2]
    new_cube[2][1] = old_F[5]
    new_cube[2][2] = old_F[8]
    new_cube[2][3] = old_F[1]
    new_cube[2][4] = old_F[4]
    new_cube[2][5] = old_F[7]
    new_cube[2][6] = old_F[0]
    new_cube[2][7] = old_F[3]
    new_cube[2][8] = old_F[6]

    # 2. 保存相邻 4 个面的相关边数据（与 rotate_F 中一致）
    # U 面底行 (索引 6,7,8)
    U_bottom = cube[0][6:9]
    # R 面左列 (索引 0,3,6)
    R_left = [cube[3][0], cube[3][3], cube[3][6]]
    # D 面顶行，按照 D210 顺序，即 D 面原先的索引 0,1,2，但以 [2,1,0] 顺序保存
    D_top_reversed = [cube[5][2], cube[5][1], cube[5][0]]
    # L 面右列 (索引 2,5,8)
    L_right = [cube[1][2], cube[1][5], cube[1][8]]

    # 3. 按逆操作的循环进行边块赋值
    # F 顺时针时：L_right -> U_bottom
    # 因此 F' 时，新 U 底行 = 原 R_left
    new_cube[0][6] = R_left[0]
    new_cube[0][7] = R_left[1]
    new_cube[0][8] = R_left[2]

    # F 顺时针时：U_bottom -> R_left
    # 因此 F' 时，新 R 左列 = 原 D_top_reversed
    new_cube[3][0] = D_top_reversed[0]
    new_cube[3][3] = D_top_reversed[1]
    new_cube[3][6] = D_top_reversed[2]

    # F 顺时针时：R_left -> D_top_reversed
    # 因此 F' 时，新 D 顶行（D210 顺序） = 原 L_right 的逆序
    new_cube[5][2] = L_right[2]
    new_cube[5][1] = L_right[1]
    new_cube[5][0] = L_right[0]

    # F 顺时针时：D_top_reversed -> L_right（逆序操作）
    # 因此 F' 时，新 L 右列 = 原 U_bottom
    new_cube[1][2] = U_bottom[0]
    new_cube[1][5] = U_bottom[1]
    new_cube[1][8] = U_bottom[2]

    return new_cube

def rotate_L_prime(cube):
    """
    对魔方状态进行 L' 转动（左侧面逆时针转动）。

    参数:
      cube: list of 6 lists，每个子列表长度为 9，
            顺序为 [U, L, F, R, B, D]，
            每个面的元素按照行优先顺序排列。

    转动规则:
      1. L 面自身逆时针旋转（使用逆时针的 3×3 映射）。
      2. 更新相邻面的对应列（逆转 rotate_L 中的循环）：
         - 新 U 面第一列 (索引 [0,3,6]) = 原 F 面第一列 (索引 [0,3,6])
         - 新 F 面第一列 = 原 D 面第一列 (索引 [0,3,6])
         - 新 D 面第一列 = 原 B 面第三列（索引 [2,5,8]，逆序，即 [B[8], B[5], B[2]]）
         - 新 B 面第三列 = 原 U 面第一列（逆序，即 [U[6], U[3], U[0]]）
    """
    # 深复制魔方状态，避免直接修改原状态
    new_cube = [face[:] for face in cube]

    # 1. 对 L 面（face index 1）进行逆时针旋转
    old_L = cube[1]
    new_cube[1][0] = old_L[2]
    new_cube[1][1] = old_L[5]
    new_cube[1][2] = old_L[8]
    new_cube[1][3] = old_L[1]
    new_cube[1][4] = old_L[4]
    new_cube[1][5] = old_L[7]
    new_cube[1][6] = old_L[0]
    new_cube[1][7] = old_L[3]
    new_cube[1][8] = old_L[6]

    # 2. 保存相邻四个面的对应边数据
    # U 面第一列（索引 0, 3, 6）
    U_left = [cube[0][0], cube[0][3], cube[0][6]]
    # F 面第一列（索引 0, 3, 6）
    F_left = [cube[2][0], cube[2][3], cube[2][6]]
    # D 面第一列（索引 0, 3, 6）
    D_left = [cube[5][0], cube[5][3], cube[5][6]]
    # B 面第三列（索引 2, 5, 8，从上到下）
    B_right = [cube[4][2], cube[4][5], cube[4][8]]

    # 3. 执行边块更新（逆转 rotate_L 的顺时针赋值）
    # 新 U 面第一列 = 原 F 面第一列（顺序不变）
    new_cube[0][0] = F_left[0]
    new_cube[0][3] = F_left[1]
    new_cube[0][6] = F_left[2]

    # 新 F 面第一列 = 原 D 面第一列（顺序不变）
    new_cube[2][0] = D_left[0]
    new_cube[2][3] = D_left[1]
    new_cube[2][6] = D_left[2]

    # 新 D 面第一列 = 原 B 面第三列（顺序反转）
    new_cube[5][0] = B_right[2]
    new_cube[5][3] = B_right[1]
    new_cube[5][6] = B_right[0]

    # 新 B 面第三列 = 原 U 面第一列（顺序反转）
    new_cube[4][2] = U_left[2]
    new_cube[4][5] = U_left[1]
    new_cube[4][8] = U_left[0]

    return new_cube

def rotate_U_prime(cube):
    """
    对魔方状态进行 U' 转动（上层逆时针转动）。

    参数:
      cube: list of 6 lists，每个子列表长度为 9，
            顺序为 [U, L, F, R, B, D]，
            每个面的元素按照行优先顺序排列。

    转动规则：
      1. U 面自身逆时针旋转：
         new[0] = old[2]
         new[1] = old[5]
         new[2] = old[8]
         new[3] = old[1]
         new[4] = old[4]
         new[5] = old[7]
         new[6] = old[0]
         new[7] = old[3]
         new[8] = old[6]
      2. 邻面上排更新（逆时针循环）：
         - 新 L 面上行 (索引 0,1,2) = 原 B 面上行
         - 新 F 面上行 = 原 L 面上行
         - 新 R 面上行 = 原 F 面上行
         - 新 B 面上行 = 原 R 面上行
    """
    # 深复制魔方状态，避免直接修改原始数据
    new_cube = [face[:] for face in cube]

    # 1. U 面自身逆时针旋转
    old_U = cube[0]
    new_cube[0][0] = old_U[2]
    new_cube[0][1] = old_U[5]
    new_cube[0][2] = old_U[8]
    new_cube[0][3] = old_U[1]
    new_cube[0][4] = old_U[4]
    new_cube[0][5] = old_U[7]
    new_cube[0][6] = old_U[0]
    new_cube[0][7] = old_U[3]
    new_cube[0][8] = old_U[6]

    # 2. 邻面上排更新
    # 保存原始上排数据
    L_top = cube[1][0:3]
    F_top = cube[2][0:3]
    R_top = cube[3][0:3]
    B_top = cube[4][0:3]

    # 按逆时针循环更新：新 L 上行 = 原 B 上行，F = 原 L，上 R = 原 F，上 B = 原 R
    new_cube[1][0:3] = B_top
    new_cube[2][0:3] = L_top
    new_cube[3][0:3] = F_top
    new_cube[4][0:3] = R_top

    return new_cube


def move_cube(state_cube, move):
    """
    根据 move 指令，对魔方状态 state_cube 进行操作，返回新的状态。

    参数:
      state_cube: 魔方当前状态，格式为 6 个面，每个面为长度为 9 的列表，顺序为 [U, L, F, R, B, D]，
                  每个面的元素按行优先排列。
      move: 字符串，取值来自 MOVES_POOL，例如 'U', "U'", "U2", 'D', "D'", "D2", 等。

    返回:
      新的魔方状态（不修改原始 state_cube）。
    """
    # 定义正转和逆转函数的映射字典
    base_moves = {
        'U': rotate_U,
        'D': rotate_D,
        'L': rotate_L,
        'R': rotate_R,
        'F': rotate_F,
        'B': rotate_B
    }
    prime_moves = {
        'U': rotate_U_prime,
        'D': rotate_D_prime,
        'L': rotate_L_prime,
        'R': rotate_R_prime,
        'F': rotate_F_prime,
        'B': rotate_B_prime
    }

    # 初始状态，不直接修改原始状态（假设各转动函数内部已做深复制）
    new_state = state_cube

    # 根据 move 的格式调用相应函数
    if len(move) == 1:
        # 例如 'U'
        new_state = base_moves[move[0]](new_state)
    elif move.endswith("2"):
        # 例如 'D2'，调用正转函数两次
        new_state = base_moves[move[0]](new_state)
        new_state = base_moves[move[0]](new_state)
    elif move.endswith("'"):
        # 例如 "L'"，调用对应的逆转函数
        new_state = prime_moves[move[0]](new_state)
    else:
        raise ValueError("Invalid move: " + move)

    return new_state


if __name__ == '__main__':
    # 示例：假设初始魔方每个面用一个字符表示颜色，状态如下：
    cube = [
        ['y'] * 9,  # U面，全白
        ['r'] * 9,  # L面，全橙
        ['g'] * 9,  # F面，全绿
        ['o'] * 9,  # R面，全红
        ['b'] * 9,  # B面，全蓝
        ['w'] * 9   # D面，全黄
    ]

    new_cube = rotate_U(cube)
    original_cube = copy.deepcopy(new_cube)
    new_cube = rotate_L(new_cube)
    # original_cube = copy.deepcopy(new_cube)
    new_cube = rotate_F(new_cube)
    # original_cube = copy.deepcopy(new_cube)
    new_cube = rotate_R(new_cube)
    new_cube = rotate_B(new_cube)
    new_cube = rotate_D(new_cube)
    new_cube = rotate_D_prime(new_cube)
    new_cube = rotate_B_prime(new_cube)
    new_cube = rotate_R_prime(new_cube)
    new_cube = rotate_F_prime(new_cube)
    new_cube = rotate_L_prime(new_cube)
    new_cube = rotate_U_prime(new_cube)
    # print(new_cube)
    # assert original_cube==new_cube
    for i, face in enumerate(new_cube):
        print(f"Face {i}: {face}")
