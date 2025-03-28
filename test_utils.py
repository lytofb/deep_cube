# 文件名: test_utils.py

import unittest
import torch
import pycuber as pc

# 引入你的 utils.py 模块
import utils

class TestUtils(unittest.TestCase):

    def test_convert_state_to_tensor(self):
        # 构造 6x9 的颜色布局 (简化为全部 'w' 或混合)
        # 假设 row0..row5 共6行，每行9个字符
        state_6x9 = [
            ['w','w','w','w','w','w','w','w','w'],  # U
            ['g','g','g','g','g','g','g','g','g'],  # L
            ['r','r','r','r','r','r','r','r','r'],  # F
            ['b','b','b','b','b','b','b','b','b'],  # R
            ['o','o','o','o','o','o','o','o','o'],  # B
            ['y','y','y','y','y','y','y','y','y'],  # D
        ]
        tensor_54 = utils.convert_state_to_tensor(state_6x9)
        self.assertEqual(tensor_54.shape, (54,))
        self.assertTrue((tensor_54[:9] == 0).all())   # 'w' => 0 (默认COLOR_CHARS)
        self.assertTrue((tensor_54[9:18] == 1).all()) # 'g' => 1
        self.assertTrue((tensor_54[18:27] == 2).all())# 'r' => 2
        self.assertTrue((tensor_54[27:36] == 3).all())# 'b' => 3
        self.assertTrue((tensor_54[36:45] == 4).all())# 'o' => 4
        self.assertTrue((tensor_54[45:] == 5).all())  # 'y' => 5

        # 测试遇到未知颜色时抛异常
        with self.assertRaises(ValueError):
            bad_state = [
                ['X'] * 9, ['g']*9, ['r']*9, ['b']*9, ['o']*9, ['y']*9
            ]
            _ = utils.convert_state_to_tensor(bad_state)

    # def test_cube_to_6x9(self):
    #     # 创建一个 PyCuber Cube，并随意修改一些面颜色
    #     cube = pc.Cube()
    #     # 假设把 Up 面都改成 'r'
    #     up_face = cube.get_face('U')
    #     for row in range(3):
    #         for col in range(3):
    #             up_face[row][col].color = 'r'
    #     # 使用 cube_to_6x9 获取 6x9
    #     # 注意: cube_to_6x9 中 face_order = ['U','L','F','R','B','D']
    #     state_6x9 = utils.cube_to_6x9(cube)
    #
    #     self.assertEqual(len(state_6x9), 6)
    #     for row in state_6x9[0]:  # 0行对应 'U'
    #         self.assertEqual(row, 'r')  # 3x3 => 9格都应该是 'r'

    def test_move_str_to_idx(self):
        # 测试已知的动作
        self.assertEqual(utils.move_str_to_idx('U'), 0)
        self.assertEqual(utils.move_str_to_idx("D2"), 5)  # 需和 MOVES_POOL 索引匹配
        # 测试未知动作返回 MASK_OR_NOMOVE_TOKEN=21
        self.assertEqual(utils.move_str_to_idx("XYZ"), utils.MASK_OR_NOMOVE_TOKEN)

    def test_move_idx_to_str(self):
        # 测试已知的动作
        self.assertEqual(utils.move_idx_to_str(0), 'U')
        self.assertEqual(utils.move_idx_to_str(5), 'D2')  # 与上面相对应
        # 如果传入不存在的 index，会抛 KeyError
        with self.assertRaises(KeyError):
            _ = utils.move_idx_to_str(999)

    def test_convert_tensor_to_state_6x9(self):
        # 做一个 (54,) 的张量，依次 0..5 重复
        # 例如 0..5 => [w,g,r,b,o,y]
        tensor_54 = torch.tensor([0]*9 + [1]*9 + [2]*9 + [3]*9 + [4]*9 + [5]*9)
        # 转回 6x9
        state_6x9 = utils.convert_tensor_to_state_6x9(tensor_54)
        self.assertEqual(len(state_6x9), 6)
        self.assertEqual(len(state_6x9[0]), 9)

        # 第0行 (U 面) 应该全部是 'w'
        self.assertTrue(all(c == 'w' for c in state_6x9[0]))
        self.assertTrue(all(c == 'g' for c in state_6x9[1]))
        self.assertTrue(all(c == 'y' for c in state_6x9[5]))

        # 测试断言: 如果长度 != 54 会报 AssertionError
        bad_tensor = torch.tensor([0,1,2])  # 长度3
        with self.assertRaises(AssertionError):
            _ = utils.convert_tensor_to_state_6x9(bad_tensor)

    def test_create_cube_from_6x9(self):
        # 做一个 6x9 布局
        state_6x9 = [
            ['w']*9,  # U
            ['g']*9,  # L
            ['r']*9,  # F
            ['b']*9,  # R
            ['o']*9,  # B
            ['y']*9,  # D
        ]
        cube = utils.create_cube_from_6x9(state_6x9)
        # 验证该 cube 的各面颜色 (U, L, F, R, B, D)
        # 这里 face_order = ['U','L','F','R','B','D']
        # 可以取 U 面, 检查都为 'w'
        u_face = cube.get_face('U')
        for row in range(3):
            for col in range(3):
                self.assertEqual(u_face[row][col].color, 'w')

        # L 面 => 'g'
        l_face = cube.get_face('L')
        for row in range(3):
            for col in range(3):
                self.assertEqual(l_face[row][col].color, 'g')

        # 你也可以测试 R 面 => 'b' 等等


if __name__ == '__main__':
    unittest.main()
