from unittest import TestCase
from inference_history import update_state
import utils


class Test(TestCase):
    def test_update_state(self):
        state_6x9 = [
            ['w','w','w','w','w','w','w','w','w'],  # U
            ['g','g','g','g','g','g','g','g','g'],  # L
            ['r','r','r','r','r','r','r','r','r'],  # F
            ['b','b','b','b','b','b','b','b','b'],  # R
            ['o','o','o','o','o','o','o','o','o'],  # B
            ['y','y','y','y','y','y','y','y','y'],  # D
        ]
        tensor_54 = utils.convert_state_to_tensor(state_6x9)
        update_state(tensor_54,0)
        # self.fail()
