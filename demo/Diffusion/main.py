import argparse
from typing import Awaitable, Callable
from cuda import cudart
from endpoint import router
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from stable_diffusion_pipeline import StableDiffusionPipeline
from utilities import PIPELINE_TYPE, process_pipeline_args
import time

args = {'version': '1.5',
    'prompt': ['A pokemon with green eyes and red legs.'],
    'negative_prompt': [''],
    'batch_size': 1,
    'batch_count': 1,
    'height': 512,
    'width': 512,
    'denoising_steps': 50,
    'scheduler': 'DDIM',
    'guidance_scale': 7.5,
    'lora_path': None,
    'lora_scale': 1,
    'lora_weights': '',
    'onnx_opset': 18,
    'onnx_dir': 'onnx',
    'onnx_refit_dir': None,
    'force_onnx_export': False,
    'force_onnx_optimize': False,
    'framework_model_dir': 'pytorch_model',
    'engine_dir': 'engine',
    'force_engine_build': False,
    'build_static_batch': False,
    'build_dynamic_shape': False,
    'build_enable_refit': False,
    'build_all_tactics': False,
    'timing_cache': None,
    'num_warmup_runs': 0,
    'use_cuda_graph': False,
    'nvtx_profile': False,
    'torch_inference': '',
    'seed': None,
    'output_dir': 'output',
    'hf_token': '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',
    'verbose': False}

args = argparse.Namespace(**args)

app = FastAPI(docs_url="/docs")
app.include_router(router, prefix="/api/v1")
@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def initialize():
    # kwargs_init_pipeline, kwargs_load_engine, args_run_demo = process_pipeline_args(args)

    kwargs_init_pipeline = {'version': '1.5',
                            'max_batch_size': 4,
                            'denoising_steps': 40,
                            'scheduler': 'DDIM',
                            'guidance_scale': 7.5,
                            'output_dir': 'output',
                            'hf_token': None,
                            'verbose': False,
                            'nvtx_profile': False,
                            'use_cuda_graph': False,
                            'lora_scale': [],
                            'lora_path': [],
                            'framework_model_dir': 'pytorch_model',
                            'torch_inference': ''}

    kwargs_load_engine = {'onnx_opset': 18,
                        'opt_batch_size': 1,
                        'opt_image_height': 512,
                        'opt_image_width': 512,
                        'static_batch': False,
                        'static_shape': True,
                        'enable_all_tactics': False,
                        'enable_refit': True,
                        'timing_cache': None}

    args_run_demo :(['A pokemon with green eyes and red legs'], [''], 512, 512, 1, 1, 0, False)

    app.pipe = StableDiffusionPipeline(
            pipeline_type=PIPELINE_TYPE.TXT2IMG,
            **kwargs_init_pipeline)

    # Load TensorRT engines and pytorch modules
    app.pipe.loadEngines(
            args.engine_dir,
            args.framework_model_dir,
            args.onnx_dir,
            **kwargs_load_engine)

    # Load resources
    _, shared_device_memory = cudart.cudaMalloc(app.pipe.calculateMaxDeviceMemory())
    app.pipe.activateEngines(shared_device_memory)
    height, width, batch_size, seed = 512, 512, 1, 12
    app.pipe.loadResources(height, width, batch_size, seed)
    return