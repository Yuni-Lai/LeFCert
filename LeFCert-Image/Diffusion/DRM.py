import torch
import torch.nn as nn 

# from Diffusion.improved_diffusion.script_util import (
#     NUM_CLASSES,
#     model_and_diffusion_defaults,
#     create_model_and_diffusion,
#     args_to_dict,
# )

# from Diffusion.guided_diffusion.script_util import (
#     NUM_CLASSES,
#     model_and_diffusion_defaults,
#     create_model_and_diffusion,
#     args_to_dict,
# )
# from transformers import AutoModelForImageClassification


class Args_ImageNet:
    image_size=256
    num_channels=256
    num_res_blocks=2
    num_heads=4
    num_heads_upsample=-1
    num_head_channels=64
    attention_resolutions="32,16,8"
    channel_mult=""
    dropout=0.0
    class_cond=False
    use_checkpoint=False
    use_scale_shift_norm=True
    resblock_updown=True
    use_fp16=False
    use_new_attention_order=False
    clip_denoised=True
    num_samples=10000
    batch_size=16
    use_ddim=False
    model_path=""
    classifier_path=""
    classifier_scale=1.0
    learn_sigma=True
    diffusion_steps=1000
    noise_schedule="linear"
    timestep_respacing=None
    use_kl=False
    predict_xstart=False
    rescale_timesteps=False
    rescale_learned_sigmas=False


class Args_CIFAR10:
    image_size=32
    num_channels=128
    num_res_blocks=3
    num_heads=4
    num_heads_upsample=-1
    attention_resolutions="16,8"
    dropout=0.3
    learn_sigma=True
    sigma_small=False
    class_cond=False
    diffusion_steps=4000
    noise_schedule="cosine"
    timestep_respacing=""
    use_kl=False
    predict_xstart=False
    rescale_timesteps=True
    rescale_learned_sigmas=True
    use_checkpoint=False
    use_scale_shift_norm=True


class DiffusionRobustModel(nn.Module):
    def __init__(self,dataset='cifar10'):
        super().__init__()
        if dataset in ['tiered_imagenet', 'cubirds200']:
            args = Args_ImageNet()
            from Diffusion.guided_diffusion.script_util import (
                NUM_CLASSES,
                model_and_diffusion_defaults,
                create_model_and_diffusion,
                args_to_dict,
            )
            model_path = "Diffusion/256x256_diffusion_uncond.pt"
        elif dataset == 'cifarfs':
            args = Args_CIFAR10()
            from Diffusion.improved_diffusion.script_util import (
                NUM_CLASSES,
                model_and_diffusion_defaults,
                create_model_and_diffusion,
                args_to_dict,
            )
            # model_path = "Diffusion/cifar10_uncond_50M_500K.pt"
            # model_path = "Diffusion/cifar100_500K.pt"
            # model_path = "Diffusion/cifar100_ema_0.9999_500K.pt"
            model_path = "Diffusion/cifarALLema750K.pt"
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        model.load_state_dict(
            torch.load(model_path)
        )
        model.eval().cuda()

        self.model = model 
        self.diffusion = diffusion

    def forward(self, x, t):
        x_in = x * 2 -1 # Rescale to [-1, 1] range
        imgs = self.denoise(x_in, t)
        imgs = torch.nn.functional.interpolate(imgs, (224, 224), mode='bicubic', antialias=True)
        return imgs

    def denoise(self, x_start, t, multistep=False):
        t_batch = torch.tensor([t] * len(x_start)).cuda()

        noise = torch.randn_like(x_start)

        x_t_start = self.diffusion.q_sample(x_start=x_start, t=t_batch, noise=noise)

        with torch.no_grad():
            if multistep:
                out = x_t_start
                for i in range(t)[::-1]:
                    t_batch = torch.tensor([i] * len(x_start)).cuda()
                    out = self.diffusion.p_sample(
                        self.model,
                        out,
                        t_batch,
                        clip_denoised=True
                    )['sample']
            else:
                out = self.diffusion.p_sample(
                    self.model,
                    x_t_start,
                    t_batch,
                    clip_denoised=True
                )['pred_xstart']

        return out