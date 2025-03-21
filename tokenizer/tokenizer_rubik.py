from utils import convert_state_to_tensor, MASK_OR_NOMOVE_TOKEN, move_str_to_idx, move_idx_to_str


class RubikTokenizer:
    def __init__(self):
        # 如果需要可在此定义move和state的词表、特殊符号等
        pass

    def encode_state(self, s6x9):
        # 原先是 convert_state_to_tensor(s6x9)，可直接调用或包裹
        return convert_state_to_tensor(s6x9)

    def encode_move(self, mv):
        # 原先是 move_str_to_idx(mv)，也可加更多逻辑
        if mv is None:
            return MASK_OR_NOMOVE_TOKEN
        else:
            return move_str_to_idx(mv)

    def decode_move(self,action_idx):
        return move_idx_to_str(action_idx)
