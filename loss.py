import torch
from bicubic import BicubicDownSample

class LossBuilder(torch.nn.Module):
    def __init__(self, ref_im, loss_str, eps):
        super(LossBuilder, self).__init__()
        assert ref_im.shape[2]==ref_im.shape[3]
        im_size = ref_im.shape[2]
        factor=1024//im_size
        assert im_size*factor==1024
        self.D = BicubicDownSample(factor=factor)
        self.ref_im = ref_im
        self.parsed_loss = [loss_term.split('*') for loss_term in loss_str.split('+')]
        self.eps = eps

    # Takes a list of tensors, flattens them, and concatenates them into a vector
    # Used to calculate euclidian distance between lists of tensors
    def flatcat(self, l):
        l = l if(isinstance(l, list)) else [l]
        return torch.cat([x.flatten() for x in l], dim=0)

    def _loss_l2(self, gen_im_lr, ref_im, **kwargs):
        return ((gen_im_lr - ref_im).pow(2).mean((1, 2, 3)).clamp(min=self.eps).sum())

    def _loss_l1(self, gen_im_lr, ref_im, **kwargs):
        return 10*((gen_im_lr - ref_im).abs().mean((1, 2, 3)).clamp(min=self.eps).sum())

    # Uses geodesic distance on sphere to sum pairwise distances of the 18 vectors
    def _loss_geocross(self, latent, **kwargs):
        if(latent.shape[1] == 1):
            return 0
        else:
            X = latent.view(-1, 1, 18, 512)
            Y = latent.view(-1, 18, 1, 512)
            A = ((X-Y).pow(2).sum(-1)+1e-9).sqrt()
            B = ((X+Y).pow(2).sum(-1)+1e-9).sqrt()
            D = 2*torch.atan2(A, B)
            D = ((D.pow(2)*512).mean((1, 2))/8.).sum()
            return D

    def forward(self, latent, gen_im):
        var_dict = {'latent': latent,
                    'gen_im_lr': self.D(gen_im),
                    'ref_im': self.ref_im,
                    }
        loss = 0
        loss_fun_dict = {
            'L2': self._loss_l2,
            'L1': self._loss_l1,
            'GEOCROSS': self._loss_geocross,
        }
        losses = {}
        for weight, loss_type in self.parsed_loss:
            tmp_loss = loss_fun_dict[loss_type](**var_dict)
            losses[loss_type] = tmp_loss
            loss += float(weight)*tmp_loss
        return loss, losses
