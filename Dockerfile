FROM nvcr.io/nvidia/pytorch:23.07-py3

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --pre --upgrade --extra-index-url https://pypi.nvidia.com tensorrt

ENV TRT_OSSPATH /workspace
WORKDIR $TRT_OSSPATH/demo/Diffusion

COPY demo/Diffusion/requirements.txt .
RUN pip3 install -r requirements.txt

RUN echo fs.inotify.max_user_watches=524288 | tee -a /etc/sysctl.conf && sysctl -p
RUN pip install fastapi uvicorn[standard]
