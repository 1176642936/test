import torch
from torch import nn
from lengyue_dl.backbone import resnet18


class BLSTMCTCNetwork(nn.Module):

    def __init__(self, image_shape, label_map_length):
        super(BLSTMCTCNetwork, self).__init__()
        self.image_shape = image_shape
        self.bone = resnet18()

        bone_output_shape = self._cal_shape()

        self.rnn = nn.LSTM(bone_output_shape, bone_output_shape, bidirectional=True)
        self.embedding = nn.Linear(bone_output_shape * 2, label_map_length)

    def _cal_shape(self):
        x = torch.zeros((1, 3) + self.image_shape)
        shape = self.bone.extract_features(x, run_to=3)[2].shape  # [1, 256, 4, 10] BATCH, DIM, HEIGHT, WIDTH
        return shape[1] * shape[2]

    def forward(self, x):
        x = self.bone.extract_features(x, run_to=3)[2]
        x = x.permute(3, 0, 1, 2)  # [10, 1, 256, 4]
        w, b, c, h = x.shape
        x = x.view(w, b, c * h)    # [10, 1, 256 * 4]

        # LSTM
        x, _ = self.rnn(x)
        time_step, batch_size, h = x.shape
        x = x.view(time_step * batch_size, h)
        x = self.embedding(x)    # [time_step * batch_size, label_map_length]
        x = x.view(time_step, batch_size, -1)

        return x
