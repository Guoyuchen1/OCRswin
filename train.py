import random
import tabnanny

from PIL import Image
import Seqdecoder
import swinencoder
import torchvision.transforms as transforms
import torch
import src.utils as utils
import torch.nn as nn
import src.dataset as dataset

with open('D:/pythonProject1/alphabet.txt') as f:
    data = f.readlines()
    alphabet = [x.rstrip() for x in data]
    alphabet = ''.join(alphabet)

# define convert bwteen string and label index
converter = utils.ConvertBetweenStringAndLabel(alphabet)

# len(alphabet) + SOS_TOKEN + EOS_TOKEN
num_classes = len(alphabet) + 2







transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#class ocr(nn.Module):
 #   def __init__(self,encoder,decoder):
 ##       super().__init__()
 #       self.encoder=encoder
 #       self.decoder=decoder
 #   def forward(self,x):
  #      encoder_out = encoder(x)
  #      encoder_out = torch.squeeze(encoder_out)
  #      encoder_out = encoder_out.unsqueeze(1)
  #      decoder_input = torch.zeros(1).long()

   #     decoder_hidden = decoder.initHidden(1)
    #    decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_out)
   #     x=decoder_output
   #     return x



#model=ocr(encoder=encoder,decoder=decoder)

def train(image, text, encoder, decoder, criterion, train_loader,test_loader, teach_forcing_prob=1):
    # optimizer
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.0001, betas=(0.5, 0.999))
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=0.0001, betas=(0.5, 0.999))

    # loss averager
    loss_avg = utils.Averager()

    for epoch in range(10):
        train_iter = iter(train_loader)
        for i in range(len(train_loader)):
            cpu_images, cpu_texts = train_iter.next()
            batch_size = cpu_images.size(0)

            for encoder_param, decoder_param in zip(encoder.parameters(), decoder.parameters()):
                encoder_param.requires_grad = True
                decoder_param.requires_grad = True
            encoder.train()
            decoder.train()

            target_variable = converter.encode(cpu_texts)
            utils.load_data(image, cpu_images)

            # CNN + BiLSTM
            encoder_outputs = encoder(image)
            encoder_outputs = torch.squeeze(encoder_outputs)
            encoder_outputs = encoder_outputs.unsqueeze(1)
            # start decoder for SOS_TOKEN
            decoder_input = target_variable[utils.SOS_TOKEN]
            decoder_hidden = decoder.initHidden(batch_size)

            loss = 0.0
            teach_forcing = True if random.random() > teach_forcing_prob else False
            if teach_forcing:
                for di in range(1, target_variable.shape[0]):
                    decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden,
                                                                                encoder_outputs)

                    loss += criterion(decoder_output, target_variable[di])
                    decoder_input = target_variable[di]

            else:
                for di in range(1, target_variable.shape[0]):

                    decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden,encoder_outputs)
                    loss += criterion(decoder_output, target_variable[di])
                    topv, topi = decoder_output.data.topk(1)

                    ni = topi.squeeze(0)

                    decoder_input = ni
            encoder.zero_grad()
            decoder.zero_grad()
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            loss_avg.add(loss)

            if i % 10 == 0:
                print('[Epoch {0}/{1}] [Batch {2}/{3}] Loss: {4}'.format(epoch, 10, i, len(train_loader),
                                                                         loss_avg.val()))
                loss_avg.reset()
        evaluate(image, text, encoder, decoder, test_loader, max_eval_iter=100)

def evaluate(image, text, encoder, decoder, data_loader, max_eval_iter=100):

    for e, d in zip(encoder.parameters(), decoder.parameters()):
        e.requires_grad = False
        d.requires_grad = False

    encoder.eval()
    decoder.eval()
    val_iter = iter(data_loader)

    n_correct = 0
    n_total = 0
    loss_avg = utils.Averager()

    for i in range(min(len(data_loader), max_eval_iter)):
        cpu_images, cpu_texts = val_iter.next()

        batch_size = cpu_images.size(0)
        utils.load_data(image, cpu_images)

        target_variable = converter.encode(cpu_texts)
        n_total += len(cpu_texts[0]) + 1

        decoded_words = []
        decoded_label = []
        encoder_outputs = encoder(image)
        encoder_outputs = torch.squeeze(encoder_outputs)
        encoder_outputs = encoder_outputs.unsqueeze(1)
        decoder_input = target_variable[0]
        decoder_hidden = decoder.initHidden(batch_size)
        for di in range(1, target_variable.shape[0]):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi.squeeze(1)
            decoder_input = ni
            if ni == utils.EOS_TOKEN:
                decoded_label.append(utils.EOS_TOKEN)
                break
            else:
                decoded_words.append(converter.decode(ni))
                decoded_label.append(ni)

        for pred, target in zip(decoded_label, target_variable[1:,:]):
            if pred == target:
                n_correct += 1

        if True:
            texts = cpu_texts[0]
            print('pred: {}, gt: {}'.format(''.join(decoded_words), texts))

    accuracy = n_correct / float(n_total)
    print('Test loss: {}, accuray: {}'.format(loss_avg.val(), accuracy))


def main():
    train_dataset = dataset.TextLineDataset(text_line_file="E:/testswin/train_list.txt", transform=None)
    sampler = dataset.RandomSequentialSampler(train_dataset, 1)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False, sampler=sampler, num_workers=1,
        collate_fn=dataset.AlignCollate(img_height=224, img_width=224))
    test_dataset = dataset.TextLineDataset(text_line_file="E:/testswin/test_list.txt", transform=dataset.ResizeNormalize(img_width=224, img_height=224))
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=int(2))
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
    num_classes = len(alphabet) + 2
    decoder = Seqdecoder.Decoder(256, output_size=num_classes, dropout_p=0.1, max_length=49)
    image = torch.FloatTensor(32, 3, 224, 224)
    text = torch.LongTensor(32)
    criterion = torch.nn.NLLLoss()

    train(image, text, encoder, decoder, criterion, train_loader, test_loader,teach_forcing_prob=0.5)
    #evaluate(image, text, encoder, decoder, test_loader, max_eval_iter=100)
if __name__ == "__main__":
    main()
