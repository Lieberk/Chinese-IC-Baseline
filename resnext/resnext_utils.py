import paddle.nn as nn


class MyResnext(nn.Layer):
    def __init__(self, resnext):
        super(MyResnext, self).__init__()
        self.resnext = nn.Sequential(*list(resnext.children()))

    def forward(self, img):
        x = img.unsqueeze(0)
        x = self.resnext[:-2](x)
        fc = x.mean(3).mean(2).squeeze()
        att = x.squeeze().transpose([1, 2, 0])
        return fc, att

