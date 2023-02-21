import argparse
import sys
import os
import numpy as np
# import matplotlib.pyplot as plt
sys.path.append('./')
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionUpscalePipeline
from diffusers import DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler 
from diffusers import DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from pytorch_lightning import seed_everything
from PIL import Image



from src import diffuser_training 
import datetime


def sample(model_id, delta_ckpt,prompt,negative_prompt='', num_steps = 30, freeze_model='crossattn_kv',cfg=7.5, num_images=5,comment=''):
    # torch.cuda.empty_cache()
#     # print(torch.cuda.memory_summary(device=None, abbreviated=False))

#     generator = torch.Generator("cuda").manual_seed(1029)  
    seed_everything(1029)
    
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    
    if is_xformers_available():
        pipe.enable_xformers_memory_efficient_attention()
    else:
        print('xformers not available')
    
    
    # print(pipe.scheduler.compatibles)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)  

    # outdir = 'outputs'
    # os.makedirs(outdir, exist_ok=True)
    if delta_ckpt is not None:
        
        diffuser_training.load_model(pipe.text_encoder, pipe.tokenizer, pipe.unet, delta_ckpt, False, freeze_model)
        outdir = os.path.dirname(delta_ckpt) #samples will be created in same directory as delta_ckpt

        all_images = []

    prompt_addition = ""
    prompt_addition = ""
    full_prompt = prompt + prompt_addition
    guidance_scale=cfg
    eta=0
    

    today = datetime.datetime.now().strftime("%b%d")
    dir_name = f'{outdir}/samples/{today}'
    os.makedirs(dir_name, exist_ok=True)
    print(f'made directory {dir_name}')
    
    # write number of training steps in saved name
    training_steps = delta_ckpt.split('_')[1].split('.')[0]
    print(f'steps {training_steps}')
    
    # takes only first 50 characters of prompt to name the image file
    prompt = prompt[:50]
    
    
    upscale = False
    
    if upscale:
    
        model_id = "stabilityai/stable-diffusion-x4-upscaler"
        pipeline = StableDiffusionUpscalePipeline.from_pretrained(
            model_id, revision="fp16", torch_dtype=torch.float16
        )
        pipeline = pipeline.to("cuda")

        if is_xformers_available():
            pipeline.enable_xformers_memory_efficient_attention()
            # pipeline.set_use_memory_efficient_attention_xformers(True)
        else:
            print('xformers not available')




    
    

    with torch.inference_mode():
        for i in range (num_images):
            image = pipe(prompt = full_prompt, negative_prompt = negative_prompt, num_images_per_prompt = 1,num_inference_steps=num_steps, guidance_scale=guidance_scale, eta=eta).images[0]
            print(f'saving image {i}')
            image.save(f'{dir_name}/{training_steps}_{cfg}_{i}_{comment}_{prompt}.jpg')
            
            if upscale:
                print(f'upscaling image {i}')
                
                # upscale from file
                image = Image.open(f"{dir_name}/test.png")
                upscaled_image = pipeline(prompt=prompt, image=image, noise_level=100).images[0]           
                upscaled_image.save(f'{dir_name}/test_UP.jpg')

                # upscale generated image
#                 upscaled_image = pipeline(prompt=prompt, image=image, noise_level=100).images[0]
#                 upscaled_image.save(f'{dir_name}/{training_steps}_{cfg}_{i}_{comment}_{prompt}_UP.jpg')



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
    sample(args.ckpt, args.delta_ckpt, args.prompt, args.negative_prompt, args.num_steps, args.freeze_model,args.cfg,args.num_images,args.comment)
