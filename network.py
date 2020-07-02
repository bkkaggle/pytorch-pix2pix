import torch
import torch.nn as nn
import torch.nn.functional as F

# class generator(nn.Module):
#     # initializers
#     def __init__(self, d=64):
#         super(generator, self).__init__()
#         # Unet encoder
#         self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
#         self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
#         self.conv2_bn = nn.BatchNorm2d(d * 2)
#         self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
#         self.conv3_bn = nn.BatchNorm2d(d * 4)
#         self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
#         self.conv4_bn = nn.BatchNorm2d(d * 8)
#         self.conv5 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
#         self.conv5_bn = nn.BatchNorm2d(d * 8)
#         self.conv6 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
#         self.conv6_bn = nn.BatchNorm2d(d * 8)
#         self.conv7 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
#         self.conv7_bn = nn.BatchNorm2d(d * 8)
#         self.conv8 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
#         # self.conv8_bn = nn.BatchNorm2d(d * 8)

#         # Unet decoder
#         self.deconv1 = nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1)
#         self.deconv1_bn = nn.BatchNorm2d(d * 8)
#         self.deconv2 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
#         self.deconv2_bn = nn.BatchNorm2d(d * 8)
#         self.deconv3 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
#         self.deconv3_bn = nn.BatchNorm2d(d * 8)
#         self.deconv4 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
#         self.deconv4_bn = nn.BatchNorm2d(d * 8)
#         self.deconv5 = nn.ConvTranspose2d(d * 8 * 2, d * 4, 4, 2, 1)
#         self.deconv5_bn = nn.BatchNorm2d(d * 4)
#         self.deconv6 = nn.ConvTranspose2d(d * 4 * 2, d * 2, 4, 2, 1)
#         self.deconv6_bn = nn.BatchNorm2d(d * 2)
#         self.deconv7 = nn.ConvTranspose2d(d * 2 * 2, d, 4, 2, 1)
#         self.deconv7_bn = nn.BatchNorm2d(d)
#         self.deconv8 = nn.ConvTranspose2d(d * 2, 3, 4, 2, 1)

#     # weight_init
#     def weight_init(self, mean, std):
#         for m in self._modules:
#             normal_init(self._modules[m], mean, std)

#     # forward method
#     def forward(self, input):
#         e1 = self.conv1(input)
#         e2 = self.conv2_bn(self.conv2(F.leaky_relu(e1, 0.2)))
#         e3 = self.conv3_bn(self.conv3(F.leaky_relu(e2, 0.2)))
#         e4 = self.conv4_bn(self.conv4(F.leaky_relu(e3, 0.2)))
#         e5 = self.conv5_bn(self.conv5(F.leaky_relu(e4, 0.2)))
#         e6 = self.conv6_bn(self.conv6(F.leaky_relu(e5, 0.2)))
#         e7 = self.conv7_bn(self.conv7(F.leaky_relu(e6, 0.2)))
#         e8 = self.conv8(F.leaky_relu(e7, 0.2))
#         # e8 = self.conv8_bn(self.conv8(F.leaky_relu(e7, 0.2)))
#         d1 = F.dropout(self.deconv1_bn(
#             self.deconv1(F.relu(e8))), 0.5, training=True)
#         d1 = torch.cat([d1, e7], 1)
#         d2 = F.dropout(self.deconv2_bn(
#             self.deconv2(F.relu(d1))), 0.5, training=True)
#         d2 = torch.cat([d2, e6], 1)
#         d3 = F.dropout(self.deconv3_bn(
#             self.deconv3(F.relu(d2))), 0.5, training=True)
#         d3 = torch.cat([d3, e5], 1)
#         d4 = self.deconv4_bn(self.deconv4(F.relu(d3)))
#         # d4 = F.dropout(self.deconv4_bn(self.deconv4(F.relu(d3))), 0.5)
#         d4 = torch.cat([d4, e4], 1)
#         d5 = self.deconv5_bn(self.deconv5(F.relu(d4)))
#         d5 = torch.cat([d5, e3], 1)
#         d6 = self.deconv6_bn(self.deconv6(F.relu(d5)))
#         d6 = torch.cat([d6, e2], 1)
#         d7 = self.deconv7_bn(self.deconv7(F.relu(d6)))
#         d7 = torch.cat([d7, e1], 1)
#         d8 = self.deconv8(F.relu(d7))
#         o = F.tanh(d8)

#         return o


# class discriminator(nn.Module):
#     # initializers
#     def __init__(self, d=64):
#         super(discriminator, self).__init__()
#         self.conv1 = nn.Conv2d(6, d, 4, 2, 1)
#         self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
#         self.conv2_bn = nn.BatchNorm2d(d * 2)
#         self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
#         self.conv3_bn = nn.BatchNorm2d(d * 4)
#         self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 1, 1)
#         self.conv4_bn = nn.BatchNorm2d(d * 8)
#         self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 1)

#     # weight_init
#     def weight_init(self, mean, std):
#         for m in self._modules:
#             normal_init(self._modules[m], mean, std)

#     # forward method
#     def forward(self, input, label):
#         x = torch.cat([input, label], 1)
#         x = F.leaky_relu(self.conv1(x), 0.2)
#         x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
#         x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
#         x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
#         x = F.sigmoid(self.conv5(x))

#         return x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True):
        super(EncoderBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=4, stride=2, padding=1)

        if bn:
            self.bn = nn.BatchNorm2d(out_ch)
        else:
            self.bn = False

    def forward(self, x):
        x = self.conv(x)

        if self.bn:
            x = self.bn(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, dropout=True, relu=True):
        super(DecoderBlock, self).__init__()

        self.conv = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=4, stride=2, padding=1)

        if bn:
            self.bn = nn.BatchNorm2d(out_ch)
        else:
            self.bn = False

        if dropout:
            self.dropout = nn.Dropout(0.5, inplace=True)
        else:
            self.dropout = False

        if relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = False

    def forward(self, x, enc=None):
        if enc is not None:
            x = torch.cat([x, enc], dim=1)

        x = self.conv(x)

        if self.bn:
            x = self.bn(x)

        if self.dropout:
            x = self.dropout(x)

        if self.relu:
            x = self.relu(x)

        return x


class generator(nn.Module):
    def __init__(self, x=64):
        super(generator, self).__init__()

        self.enc1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)
        self.enc5 = EncoderBlock(512, 512)
        self.enc6 = EncoderBlock(512, 512)
        self.enc7 = EncoderBlock(512, 512)
        self.enc8 = EncoderBlock(512, 512, bn=False)

        self.dec1 = DecoderBlock(512, 512)
        self.dec2 = DecoderBlock(1024, 512)
        self.dec3 = DecoderBlock(1024, 512)
        self.dec4 = DecoderBlock(1024, 512, dropout=False)
        self.dec5 = DecoderBlock(1024, 256, dropout=False)
        self.dec6 = DecoderBlock(512, 128, dropout=False)
        self.dec7 = DecoderBlock(256, 64, dropout=False)
        self.dec8 = DecoderBlock(128, 3, dropout=False)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.leaky_relu(enc1, 0.2, inplace=True))
        enc3 = self.enc3(F.leaky_relu(enc2, 0.2, inplace=True))
        enc4 = self.enc4(F.leaky_relu(enc3, 0.2, inplace=True))
        enc5 = self.enc5(F.leaky_relu(enc4, 0.2, inplace=True))
        enc6 = self.enc6(F.leaky_relu(enc5, 0.2, inplace=True))
        enc7 = self.enc7(F.leaky_relu(enc6, 0.2, inplace=True))
        enc8 = self.enc8(F.leaky_relu(enc7, 0.2, inplace=True))

        dec1 = self.dec1(enc8)
        dec2 = self.dec2(dec1, enc7)
        dec3 = self.dec3(dec2, enc6)
        dec4 = self.dec4(dec3, enc5)
        dec5 = self.dec5(dec4, enc4)
        dec6 = self.dec6(dec5, enc3)
        dec7 = self.dec7(dec6, enc2)
        dec8 = self.dec8(dec7, enc1)

        out = torch.tanh(dec8)

        return out


class discriminator(nn.Module):
    def __init__(self, x=64):
        super(discriminator, self).__init__()

        self.conv1 = nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)

        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # def forward(self, x):
    def forward(self, x, y):

        x = torch.cat([x, y], 1)

        x = self.relu1(self.conv1(x))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))

        x = self.conv5(x)

        out = torch.sigmoid(x)

        return out
