import argparse
import os

import chainer
from net_mnist import Generator

import numpy as np
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description='Chainer: MNIST predicting CNN')
    parser.add_argument('--n_hidden', '-n', type=int, default=100,
                        help='Number of hidden units (z)')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--rows', '-r', type=int, default=10,
                        help='Number of rows in the image')
    parser.add_argument('--cols', '-c', type=int, default=10,
                        help='Number of cols in the image')
    parser.add_argument('--out', '-o', default='generate_image',
                        help='Directory to output the result')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed of z at visualization stage')
    args = parser.parse_args()

    print('# n_hidden: {}'.format(args.n_hidden))
    print('# epoch: {}'.format(args.epoch))
    print('# Number of rows in the image: {}'.format(args.rows))
    print('# Number of cols in the image: {}'.format(args.cols))
    print('')

    gen = Generator(n_hidden=args.n_hidden)
    chainer.serializers.load_npz('result/gen_epoch_100.npz', gen)

    np.random.seed(args.seed)
    n_images = args.rows * args.cols
    xp = gen.xp
    z = chainer.Variable(xp.asarray(gen.make_hidden(n_images)))

    x = gen(z)
    x = chainer.cuda.to_cpu(x.data)
    np.random.seed()

    # gen_output_activation_func is sigmoid (0 ~ 1)
    x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)
    # gen output_activation_func is tanh (-1 ~ 1)
    # x = np.asarray(np.clip((x+1) * 0.5 * 255, 0.0, 255.0), dtype=np.uint8)
    _, _, H, W = x.shape
    x = x.reshape((args.rows, args.cols, 1, H, W))
    x = x.transpose(0, 3, 1, 4, 2)
    x = x.reshape((args.rows * H, args.cols * W))

    preview_dir = '{}'.format(args.out)
    preview_path = preview_dir + '/generate_image_epoch_{:0>4}.png'.format(args.epoch)
    if not os.path.exists(preview_dir):
        os.makedirs(preview_dir)
    Image.fromarray(x).save(preview_path)

if __name__ == '__main__':
    main()
