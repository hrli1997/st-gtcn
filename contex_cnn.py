import torchvision.models as models
from torch import optim, nn
from readImg import img


class ContexConv(nn.Module):

    def __init__(self, out_shape):
        # input shape: shape as [batch, channel, h, w]
        # output shape: shape as [1, features]
        super(ContexConv, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(1000, out_shape[1]),
            nn.LeakyReLU(inplace=False),
        )
        self.Conv = models.resnet18(pretrained=False)

    def forward(self, pic):
        hidden = self.Conv(pic)
        output = self.fc(hidden)
        return output


if __name__ == "__main__":
    cnn = ContexConv([1, 50])
    res = cnn(img)
