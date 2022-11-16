import numpy as np
import random
import os

def numf(i, l=3, end='.jpg'):
    fname = '%d' % i
    return '0' * (l - len(fname)) + fname + end


if __name__ == '__main__':

    src = r'data/fft_data_processed'
    dst = r'data/fft_data_sampled'
    seed = 2021
    step = 100

    for env in os.listdir(src):
        for act in os.listdir(os.path.join(src, env)):
            input_dir = os.path.join(src, env, act)
            output_dir = os.path.join(dst, env, act)
            print(output_dir)
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            for i, npy_file in enumerate(os.listdir(input_dir)):
                input_file = os.path.join(input_dir, npy_file)
                arr = np.load(input_file)
                # print(input_file)
                # assert arr.shape == (200, 769)

                # arr = standardization(arr, input_file)

                iter = 2
                random.seed(seed)
                start_point = random.randint(0, step)

                length = arr.shape[0] // iter
                for p in range(iter):
                    arr_temp = arr[length*p:length * (p + 1), :]
                    end_point = arr_temp.shape[0]
                    arr_save = arr_temp[start_point:end_point:step, :]
                    np.save(os.path.join(output_dir, numf((i+1) * iter - (iter - 1 - p), end='.npy')),arr_save)
