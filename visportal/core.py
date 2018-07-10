import visdom
import numpy as np
import torch
# from torch.autograd import Variable

class VisdomPortal():
    def __init__(self, env_name='Portal_Default', image_norm=[(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]):
        # prepare visdom
        self.vis = visdom.Visdom(env=env_name)
        self.win_handles = {}
        self.image_norm = image_norm[0]
        self.image_std = image_norm[1]

    def draw_curve(self, value, step, title):
        '''

        :param value: list of torch.tensor, numpy, int or float.
        :param step: int, float
        :param title: string
        :return:
        '''
        assert isinstance(value, (int, float,
                                  np.ndarray)), 'Value type should be int, float or numpy.ndarray.'
        assert isinstance(step, (int, float, np.ndarray)), 'Step type should be int, float or numpy.ndarray.'

        if isinstance(value, (int, float)):
            value = np.array([value], dtype=np.float)

        if isinstance(step, np.ndarray):
            step = step.astype(int)
        else:
            step = np.array([step], dtype=int)

        if title in self.win_handles:
            win_name = self.win_handles[title]
            self.vis.line(value, step, win=win_name, update='append', opts={'title': title})
        else:
            self.win_handles[title] = self.vis.line(value, step, opts={'title': title})

    def draw_curves(self, values, names=None, step=None, title=None):
        '''
        :param values: list of torch.tensor, numpy, int or float.
        :param names: list of string
        :param step: int, float
        :param title: string
        :return: None
        '''
        assert isinstance(values, list) and isinstance(names, list)
        assert not isinstance(names, type(None)) and len(names) == len(values)

        for i, value in enumerate(values):

            if isinstance(value, torch.Tensor):
                if value.is_cuda:
                    value = value.cpu()
                value = value.numpy()
            if isinstance(value, np.ndarray):
                value = value[0]
            values[i] = value

        if title in self.win_handles:
            win_name = self.win_handles[title]
            self.vis.line(np.expand_dims(np.array(values), 0), np.expand_dims(np.repeat(step, len(values)), 0),
                          win=win_name, update='append', opts={'title': title, 'legend': names})
        else:
            self.win_handles[title] = self.vis.line(np.expand_dims(np.array(values), 0),
                                                    np.expand_dims(np.repeat(step, len(values)), 0),
                                                    opts={'title': title, 'legend': names})

    def draw_bars(self, value, title, legends=[]):
        if isinstance(value, dict):
            keys = []
            vals = []
            for key, val in value.items():
                keys.append(key)
                vals.append(val)
            legends = keys
            value = torch.from_numpy(np.array(vals))
        elif isinstance(value, list):
            value = torch.from_numpy(np.array(value))

        if title in self.win_handles:
            win_name = self.win_handles[title]
            self.vis.bar(value, win=win_name, opts={'legend': legends, 'title': title})
        else:
            self.win_handles[title] = self.vis.bar(value, opts={'legend': legends, 'title': title})

    def draw_images(self, value, title, unnormalize=True):
        assert isinstance(value, (torch.Tensor, np.ndarray)), 'Value type should be torch tensor or numpy.ndarray.'
        # variable
        # if isinstance(value, torch.autograd.variable.Variable):
        #     if value.is_cuda:
        #         value = value.cpu()
        #     value = value.data.numpy()
        # tensor
        if isinstance(value, torch.Tensor):
            if value.is_cuda:
                value = value.cpu()
            value = value.numpy()

        value = value.astype(np.float)
        assert len(value.shape) == 4, 'value must have 4 dimensions'
        if unnormalize:
            norm = np.array(self.image_norm[:value.shape[1]]).reshape((1, -1, 1, 1))
            std = np.array(self.image_std[:value.shape[1]]).reshape((1, -1, 1, 1))
            value = (value * std + norm) * 255

        if title in self.win_handles:
            win_name = self.win_handles[title]
            self.vis.images(value.astype(np.uint8), win=win_name,
                            opts={'caption': title})
        else:
            self.win_handles[title] = self.vis.images(value.astype(np.uint8),
                                                      opts={'caption': title})
