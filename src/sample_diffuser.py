import argparse
import sys
import os
import numpy as np
# import matplotlib.pyplot as plt
sys.path.append('./')
import torch
from diffusers import StableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler 
from diffusers import DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from pytorch_lightning import seed_everything


from src import diffuser_training 
import datetime


def sample(model_id, delta_ckpt,prompt, compress, freeze_model,cfg):
    # torch.cuda.empty_cache()
    # print(torch.cuda.memory_summary(device=None, abbreviated=False))

    generator = torch.Generator("cuda").manual_seed(1029)  
    seed_everything(1029)
    
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16,generator=generator).to("cuda")
    
    if is_xformers_available():
        pipe.enable_xformers_memory_efficient_attention()
    else:
        print('xformers not available')
    
    
    # print(pipe.scheduler.compatibles)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)  

    # outdir = 'outputs'
    # os.makedirs(outdir, exist_ok=True)
    if delta_ckpt is not None:
        
        diffuser_training.load_model(pipe.text_encoder, pipe.tokenizer, pipe.unet, delta_ckpt, compress, freeze_model)
        outdir = os.path.dirname(delta_ckpt) #samples will be created in same directory as delta_ckpt

        all_images = []

    prompt_addition = ""
    prompt_addition = ""
    full_prompt = prompt + prompt_addition
    negative_prompt = args.negative_prompt
    guidance_scale=cfg
    num_inference_steps = args.num_steps
    eta=0
    

    today = datetime.datetime.now().strftime("%b%d")
    dir_name = f'{outdir}/samples/{today}'
    os.makedirs(dir_name, exist_ok=True)
    print(f'made directory {dir_name}')
    
    steps = delta_ckpt.split('_')[1].split('.')[0]
    print(f'steps {steps}')
    
    if len(args.prompt)>10:
        args.prompt='long'    

    with torch.inference_mode():
        for i in range (args.num_images):
            image = pipe(prompt = full_prompt, negative_prompt = negative_prompt, num_images_per_prompt = 1,num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, eta=eta,generator=generator).images[0]
            print(f'saving image {i}')
            image.save(f'{dir_name}/{steps}_{cfg}_{i}_{args.comment}_{prompt}.jpg')
            
#     if len(args.negative_prompt)>10:
#         args.negative_prompt='long'
    

#                image.save(f'{dir_name}/{args.prompt}_{args.negative_prompt}_{i}_{num_inference_steps}_{guidance_scale}_{eta}_{args.comment}.jpg')
    



def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--ckpt', help='target string for query',
                        type=str)
    parser.add_argument('--delta_ckpt', help='target string for query', default=None,
                        type=str)
    # parser.add_argument('--from-file', help='path to prompt file', default='./',
    #                     type=str)
    parser.add_argument('--prompt', help='prompt to generate', default=None,
                        type=str)
    parser.add_argument('--negative_prompt', help='negative_prompt', default="",
                        type=str)
    parser.add_argument("--compress", action='store_true')
    parser.add_argument('--freeze_model', help='crossattn or crossattn_kv', default='crossattn_kv',
                        type=str)
    parser.add_argument("--comment", help='prevents images from being overwritten when saved',type=str)
    parser.add_argument("--cfg", help='cfg value',type=float)
    # parser.add_argument("--date", help='date for folder name',type=str)
    parser.add_argument("--num_steps", help='number of inference steps',type=int)
    parser.add_argument("--num_images", help='number of images generated',type=int)

    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sample(args.ckpt, args.delta_ckpt, args.prompt, args.compress, args.freeze_model,args.cfg)
