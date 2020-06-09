import subprocess
from moviepy import editor as mpy
import tempfile
import shutil
import os
import json
import numpy as np
import torchvision
import torch


class ExpvizLogger:
    from expviz.logger import Logger
    
    def __init__(self, *args, **kwargs):
        self.expviz = self.Logger(*args, **kwargs)
    
    def write(self, scalar_dict, epoch):
        for key, value in scalar_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.expviz.add_scalar(key, value, epoch)
    
    def add_image(self, name, img, epoch):
        self.expviz.add_image(name, img, epoch)


def _prepare_video(V, n_rows=None):
    import numpy as np
    b, t, c, h, w = V.shape

    if V.dtype == np.uint8:
        V = np.float32(V) / 255.

    if n_rows is None:
        n_rows = 2**((b.bit_length() - 1) // 2)
    n_cols = b // n_rows

    r = b % n_rows
    if r != 0:
        V = np.concatenate(
            (V, np.zeros(shape=(n_rows - r, t, c, h, w))), axis=0)
        n_cols += 1 

    V = np.reshape(V, newshape=(n_rows, n_cols, t, c, h, w))
    V = np.transpose(V, axes=(2, 0, 4, 1, 5, 3))
    V = np.reshape(V, newshape=(t, n_rows * h, n_cols * w, c))

    return V


class Logger:
    def __init__(self, logdir):
        self.logdir = logdir

        if not os.path.exists(logdir):
            os.makedirs(logdir)

    def add_hparams(self, params):
        filename = os.path.join(self.logdir, "hparams.json")
        if not isinstance(params, dict):
            params = vars(params)
        
        with open(filename, 'w') as f:
            json.dump(params, f)

    def add_image(self, img_tensor, epoch, nrow=None):
        b = len(img_tensor)
        if nrow is None:
            nrow = 2**((b.bit_length() - 1) // 2)
        filename = os.path.join(self.logdir, "img/img_%04d.png"%epoch)
        torchvision.utils.save_image(img_tensor, filename, nrow=nrow)

    def add_video(self, vid_tensor, epoch, nrow=None, fps=5):
        vid_tensor = vid_tensor.cpu().numpy()

        vid_tensor = _prepare_video(vid_tensor, n_rows=nrow)
        
        if vid_tensor.dtype != np.uint8:
            vid_tensor = (vid_tensor * 255.0).astype(np.uint8)

        clip = mpy.ImageSequenceClip(list(vid_tensor), fps=fps)
        tmpdirname = tempfile.mkdtemp()
        list_files = os.path.join(tmpdirname, "frame_%04d.png")
        clip.write_images_sequence(list_files, verbose=False, logger=None)
        
        video_dir = os.path.join(self.logdir, "videos")
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        filename = os.path.join(video_dir, "video_%.5i.mp4" % epoch)
        #tmpfile = os.path.join(tmpdirname, "video.mp4")

        subprocess.run(["ffmpeg", "-r", str(fps), "-f", "image2", "-s", "1920x1080", "-i", list_files, "-vcodec", "libx264", "-crf", "25", filename],
                        stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        #subprocess.run(["ffmpeg", "-i", tmpfile, filename])

        shutil.rmtree(tmpdirname)

