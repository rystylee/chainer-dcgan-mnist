import os

import numpy as np
from PIL import Image

import chainer
import chainer.cuda
from chainer import Variable


def out_generated_image(gen, dis, rows, cols, seed, dst):
    @chainer.training.make_extension(trigger=(1, 'epoch'))
    def make_image(trainer):
        np.random.seed(seed)
        n_images = rows * cols
        xp = gen.xp
        z = Variable(xp.asarray(gen.make_hidden(n_images)))

        with chainer.using_config('train', False):
            x = gen(z)
        x = chainer.cuda.to_cpu(x.data)
        np.random.seed()

        # gen_output_activation_func is sigmoid (0 ~ 1)
        x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)
        # gen output_activation_func is tanh (-1 ~ 1)
        # x = np.asarray(np.clip((x+1) * 0.5 * 255, 0.0, 255.0), dtype=np.uint8)
        _, _, H, W = x.shape
        x = x.reshape((rows, cols, 1, H, W))
        x = x.transpose(0, 3, 1, 4, 2)
        x = x.reshape((rows * H, cols * W))

        preview_dir = '{}/preview'.format(dst)
        preview_path = preview_dir + '/image_epoch_{:0>4}.png'.format(trainer.updater.epoch)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        Image.fromarray(x).save(preview_path)
    return make_image
