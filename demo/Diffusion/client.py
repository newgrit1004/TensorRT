import requests
import json
HOST = "172.20.96.65" #PUT TensorRT engine host IP
PORT = "8000" #PUT TensorRT engine
url = f"http://{HOST}:{PORT}/api/v1/lora-test"

lora_path = ["sayakpaul/sd-model-finetuned-lora-t4"]
lora_scale = [1.0]
body = json.dumps({"lora_path":lora_path, "lora_scale":lora_scale})
response = requests.post(url=url, data=body)
print(f"first response :  {response}")

lora_path = ["WuLing/Genshin_Bennett_LoRA"]
lora_scale = [1.0]
body = json.dumps({"lora_path":lora_path, "lora_scale":lora_scale})
response = requests.post(url=url, data=body)
print(f"second response :  {response}")

lora_path = ["sayakpaul/sd-model-finetuned-lora-t4"]
lora_scale = [1.0]
body = json.dumps({"lora_path":lora_path, "lora_scale":lora_scale})
response = requests.post(url=url, data=body)
print(f"third response :  {response}")

