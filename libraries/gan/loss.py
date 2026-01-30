import torch.nn.functional as F
from torch import autograd

from libraries.custom_stylegan2.torch_utils.ops import upfirdn2d
from libraries.custom_stylegan2.dual_discriminator import filtered_resizing

def adv_loss_dis(real, fake, adv_loss_type, tmp=1.0):
    if adv_loss_type == "hinge":
        return F.relu(1 - real).mean() + F.relu(1 + fake).mean()
    elif adv_loss_type == "ce":
        return F.softplus(-real * tmp).mean() + F.softplus(fake * tmp).mean()
    else:
        assert False, f"{adv_loss_type} is not supported"


def adv_loss_gen(fake, adv_loss_type, tmp=1.0):
    if adv_loss_type == "hinge":
        return -fake.mean()
    elif adv_loss_type == "ce":
        return F.softplus(-fake * tmp).mean()
    else:
        assert False, f"{adv_loss_type} is not supported"


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

def d_r1_loss_dual(real_pred, real_img):
    
    # resample_filter = upfirdn2d.setup_filter([1,3,3,1])
    # image_raw = filtered_resizing(real_img['image_raw'], size=real_img['image'].shape[-1], f=resample_filter)
    # real_img = torch.cat([real_img['image'], image_raw], 1)
    # grad_real, = autograd.grad(
        # outputs=real_pred.sum(), inputs=real_img, create_graph=True
    # )
    # grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    r1_grads = autograd.grad(outputs=[real_pred.sum()], inputs=[real_img['image'], real_img['image_raw']], create_graph=True, only_inputs=True)
    r1_grads_image = r1_grads[0].pow(2).reshape(r1_grads[0].shape[0], -1).sum(1).mean()
    r1_grads_image_raw = r1_grads[1].pow(2).reshape(r1_grads[1].shape[0], -1).sum(1).mean()
    grad_penalty = r1_grads_image + r1_grads_image_raw


    return grad_penalty
    