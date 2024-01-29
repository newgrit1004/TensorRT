from fastapi import APIRouter, Request
from models import LoraLoader
from hashlib import md5

router = APIRouter()

@router.post("/lora-test", status_code=200)
async def test(request:Request, body:dict) -> int:
    print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
    print(body)
    print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")

    request.app.pipe.lora_path = body["lora_path"]
    request.app.pipe.lora_scale = body["lora_scale"]
    request.app.pipe.lora_scales = dict()
    if request.app.pipe.lora_path:
        request.app.pipe.lora_loader = LoraLoader(request.app.pipe.lora_path)
        for i, path in enumerate(request.app.pipe.lora_path):
            request.app.pipe.lora_scales[path] = request.app.pipe.lora_scale[i]

    print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
    print(f"request.app.pipe.lora_path : {request.app.pipe.lora_path}")
    print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")

    lora_suffix = '-'+'-'.join([str(md5(path.encode('utf-8')).hexdigest())+'-'+('%.2f' % request.app.pipe.lora_scales[path]) for path in sorted(request.app.pipe.lora_loader.paths)]) if request.app.pipe.lora_loader else ''

    # request.app.pipe.loadEngines("./engine", "./pytorch_model", "./onnx", **kwargs_load_engine)
    # _, shared_device_memory = cudart.cudaMalloc(request.app.pipe.calculateMaxDeviceMemory())
    # request.app.pipe.activateEngines(shared_device_memory)
    # height, width, batch_size, seed = 512, 512, 1, 12
    # request.app.pipe.loadResources(height, width, batch_size, seed)

    print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
    print(f"lora_suffix : {lora_suffix}")
    print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
    request.app.pipe.refit_engine(lora_suffix)

    args_run_demo = (['A pokemon with green eyes and red legs'], [''], 512, 512, 1, 1, 0, False)
    request.app.pipe.run(*args_run_demo)

    return 1