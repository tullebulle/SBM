"""SBM text-to-image: the one-step guided CBM map in the RAE DINOv2-S latent, prompted by FLAN-T5 (TextDiT).

Importing the package registers the `${store:}` / `${repo:}` config resolvers (sbm_t2i.run).
"""
from sbm_t2i import run as _run  # noqa: F401
