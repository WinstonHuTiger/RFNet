import numpy as np
import os
import torch
def numf(i, l=3, end='.jpg'):
    fname = '%d' % i
    return '0' * (l - len(fname)) + fname + end

src = 'data/fft_data_processed'

action_to_index = {
    'rip': 0, 'run': 1, 'sit': 2, 'squat': 3, 'walk': 4
}
tensory_to_save = np.zeros((1, 5, 4800, 5))
tensor_to_save = np.zeros((1, 5, 4800, 25600, 256))
index  = 0
for env in ["env1"]:
    for act in list(action_to_index.keys()):
        input_dir = os.path.join(src, env, act)
       
        for i in range(1, len(os.listdir(input_dir)) + 1):
            input_file = os.path.join(input_dir, '%d.npy' % i)
            arr = np.load(input_file)
            print(arr.shape)
            assert arr.shape == (25600, 256)

            # arr = standardization(arr, input_file)
            tensor_to_save[0, action_to_index[act], index, : , : ] = arr
            temp = np.zeros((6))
            temp[action_to_index[act]]  = 1
            tensor_to_save[0, action_to_index[act], index, : ] = temp
torch.save(torch.from_numpy(tensor_to_save), "data/x_train.pth")
torch.save(torch.from_numpy(tensory_to_save), "data/y_train.pht")