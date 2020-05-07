import torch


class ColorDepthSqueezer():
    def __init__(self, s):
        self.s = int(s)

    def transform(self, x):
        return torch.floor(x*255/self.s)*self.s/255


if __name__ == "__main__":
    squeezer = ColorDepthSqueezer(128)
    x = torch.rand((3,3))
    print(x)
    print(squeezer.transform(x))
