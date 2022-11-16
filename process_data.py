import numpy as np
import os

def numf(i, l=3, end='.jpg'):
    fname = '%d' % i
    return '0' * (l - len(fname)) + fname + end

src = 'data/fft_data'
dst = 'data/fft_data_processed'

for env in os.listdir(src):
    for act in os.listdir(os.path.join(src, env)):
        input_dir = os.path.join(src, env, act)
        output_dir = os.path.join(dst, env, act)
        print(output_dir)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        for i in range(1, len(os.listdir(input_dir)) + 1):
            input_file = os.path.join(input_dir, '%d.npy' % i)
            arr = np.load(input_file)
            # print(arr.shape)
            assert arr.shape == (76800, 256)

            # arr = standardization(arr, input_file)

            iter = 3
            length = arr.shape[0] // iter
            for p in range(iter):
                # print(os.path.join(output_dir, numf(i * iter - (iter - 1 - p))), i, length * p, length * (p + 1))
                np.save(os.path.join(output_dir, numf(i * iter - (iter - 1 - p), end='.npy')), arr[length * p: length * (p + 1),:])

            # np.save(os.path.join(output_dir, numf(i * 3 - 2, end='.npy')), arr[:, :200])
            # np.save(os.path.join(output_dir, numf(i * 3 - 1, end='.npy')), arr[:, 200:400])
            # np.save(os.path.join(output_dir, numf(i * 3 - 0, end='.npy')), arr[:, 400:600])
