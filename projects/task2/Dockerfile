FROM mambaorg/micromamba:latest
COPY env.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN pip install pyarmor==6.7.4 torch==2.0.1
RUN pip install "ray[tune]"
# RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
WORKDIR /code
ADD * /code/
ADD pytransform /code/pytransform
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/code/pytransform
WORKDIR /code
CMD python -u checker_client.py
