"""The SD-VAE trainer's tests run on the CPU (fp32, tiny DiT), also on a machine with GPUs."""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
