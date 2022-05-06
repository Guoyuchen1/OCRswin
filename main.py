from PIL import Image
import Seqdecoder
import swinencoder
import torchvision.transforms as transforms
import torch
encoder = swinencoder.SwinTransformer(img_size=224,
                        patch_size=4,
                        in_chans=3,
                        num_classes=0,
                        embed_dim=96,
                        depths=[2, 2, 6, 2],
                        num_heads=[3, 6, 12, 24],
                        window_size=7,
                        mlp_ratio=4.,
                        qkv_bias=True,
                        qk_scale=None,
                        drop_rate=0.0,
                        drop_path_rate=0.1,
                        ape=False,
                        patch_norm=True,
                        use_checkpoint=False)

decoder = Seqdecoder.Decoder(256, 52 ,dropout_p=0.0, max_length=49)

img = Image.open("E:/testswin/train/1_pontifically_58805.jpg")

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
img = transform_test(img)
img.unsqueeze_(0)

encoder_out = encoder(img)
encoder_out =torch.squeeze(encoder_out)

encoder_out =encoder_out.unsqueeze(1)
print(encoder_out.shape)
decoder_input = torch.zeros(1).long()

decoder_hidden = decoder.initHidden(1)
decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_out)
print(decoder_output)
