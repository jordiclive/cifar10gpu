import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.nn import Conv2d, MaxPool2d
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser
from collections import OrderedDict
device = 'cpu'
dtype = torch.float32


from Mish.mish import MISH
from Mish.functional import mish
from pytorch_lightning.metrics import functional as FM


# define resnet building blocks

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()

        self.left = nn.Sequential(Conv2d(inchannel, outchannel, kernel_size=3,
                                         stride=stride, padding=1, bias=False),
                                  nn.BatchNorm2d(outchannel),
                                  MISH(),
                                  Conv2d(outchannel, outchannel, kernel_size=3,
                                         stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(outchannel))

        self.shortcut = nn.Sequential()

        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(Conv2d(inchannel, outchannel,
                                                 kernel_size=1, stride=stride,
                                                 padding=0, bias=False),
                                          nn.BatchNorm2d(outchannel))

    def forward(self, x):
        out = self.left(x)

        out += self.shortcut(x)

        out = mish(out)

        return out

    # define resnet


class ResNet(pl.LightningModule):

    def __init__(self, ResidualBlock, config):
        super(ResNet, self).__init__()
        self.config = config
        self.inchannel = 64
        self.conv1 = nn.Sequential(Conv2d(3, 64, kernel_size=3, stride=1,
                                          padding=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())

        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.maxpool = MaxPool2d(4)
        self.num_classes = config.num_classes
        self.fc = nn.Linear(512, self.num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)

        layers = []

        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))

            self.inchannel = channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.maxpool(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


    def training_step(self,batch,batch_idx):
        x, y = batch
        x = x.to( dtype=dtype)
        y = y.to( dtype=torch.long)

        scores = self(x)
        loss = F.cross_entropy(scores, y)


        log = {'train_loss':loss}

        return OrderedDict({'loss':loss,'progress_bar': log,'log':log})


    def validation_step(self,batch,batch_idx):

        x, y = batch
        # put model to training mode
        x = x.to(dtype=dtype)  # move to device, e.g. GPU
        y = y.to( dtype=torch.long)

        scores = self(x)

        val_loss = F.cross_entropy(scores, y)

        acc = FM.accuracy(scores, y)

        return OrderedDict({'val_loss': val_loss,'val_acc':acc})

    def test_step(self, batch, batch_idx):
        result = self.validation_step(batch, batch_idx)
        result['test_acc'] = result.pop('val_acc')
        result['test_loss'] = result.pop('val_loss')
        return result

    def test_epoch_end(self, outputs):
        test_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        result = pl.EvalResult()
        result.log('final_metric', test_acc)
        return result

    def validation_epoch_end(self,outputs):
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        log = {'avg_val_loss': val_loss}
        self.logger.experiment.add_scalar('validation loss',
                            val_loss,
                            self.current_epoch)
        return {'log': log,'val_loss': val_loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(),lr = self.config.learning_rate)

def ResNet18(args):
    return ResNet(ResidualBlock,args)


def _parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument('--learning_rate', default = 1e-2, type = float)
    parser.add_argument('--num_classes', default = 10, type = int)
    parser.add_argument('--max_epochs',default=15, type= int)
    parser.add_argument('--gpus',default=0, type=int)
    parser.add_argument('--save_top_k', default = 1, type=int)
    parser.add_argument('--resume_from_checkpoint', default = None, type=str)
    #parser.add_argument('--ckpt_path', default='checkpoints/',type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    from data.data import CIFAR10

    from pathlib import Path



    DATA = CIFAR10()
    train_loader = DATA.train_dataloader()
    val_loader = DATA.val_dataloader()
    test_loader = DATA.test_dataloader()

    args = _parse_args()
    model = ResNet18(args)
    ckpt_path = str(Path(__file__).parents[0].resolve() / "checkpoints/")
    checkpoint_callback = ModelCheckpoint(save_top_k=args.save_top_k,verbose=True,monitor='val_loss',mode='min')


    trainer = pl.Trainer.from_argparse_args(args,default_root_dir=ckpt_path,checkpoint_callback=checkpoint_callback)
    trainer.fit(model,train_dataloader=train_loader,val_dataloaders=val_loader)
    trainer.test(test_dataloaders = test_loader)
    print(checkpoint_callback.best_model_path)

