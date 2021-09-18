import os
import torch

from transformers import ViTConfig, ViTModel

class ModViT(torch.nn.Module):
    head_file = "head.pt"

    def __init__(self, vit_cols, lin_cols, vit_lines, conv_lines,
                 pretrained_vit=True, vit_path="google/vit-base-patch16-224", img_size=224):
        super().__init__()

        if pretrained_vit:
            self.vit = ViTModel.from_pretrained(vit_path)
        else:
            config = ViTConfig(image_size=img_size)
            self.vit = ViTModel(config)

        self.head = torch.nn.Sequential(
            torch.nn.Linear(vit_cols, lin_cols),
            torch.nn.Conv1d(vit_lines, conv_lines, 1)
        )

        try:
            self.vit.cuda()
            self.head.cuda()
        except:
            print("[WARNING] Model is on CPU")


    def forward(self, x):
        out = self.vit(**x)
        last_hidden_state = out.last_hidden_state   # last_hidden_state.shape = torch.Size([1, vit_lines, vit_cols])

        # print(last_hidden_state.shape)
        # print(self.head)

        y = self.head(last_hidden_state)   # y.shape = torch.Size([1, conv_lines, lin_cols])
        return y[0]

    def train(self):
        self.vit.train()
        self.head.train()

    def eval(self):
        self.vit.eval()
        self.head.eval()

    def load(self, path, gpu=True):
        try:
            self.vit = ViTModel.from_pretrained(path)
            if gpu:
                checkpoint = torch.load(os.path.join(path, self.head_file)) # map_location="cuda"
            else:
                checkpoint = torch.load(os.path.join(path, self.head_file), map_location="cpu")
            self.head.load_state_dict(checkpoint['head_state_dict'])
            del checkpoint
            print("Checkpoint loaded")
        except:
            print("Could not load checkpoint")

        if gpu:
            self.vit.cuda()
            self.head.cuda()
        else:
            print("[WARNING] Model is on CPU")

    def save(self, path):
        self.vit.save_pretrained(path)
        torch.save({'head_state_dict': self.head.state_dict()}, os.path.join(path, self.head_file))
        print("Saved model")
