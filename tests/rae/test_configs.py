"""configs/rae composes to the released recipes, command-line override sets apply, and every path goes through SBM_STORE."""
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

import sbm_rae.paths  # importing sbm_rae registers the ${sbm_store:} resolver

REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "configs" / "rae"


def compose_cfg(name="train", overrides=()):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(config_name=name, overrides=list(overrides))


def test_train_config_composes_the_rae_line():
    cfg = compose_cfg()
    assert cfg.model.name == "DiT-XL/1-DH" and cfg.model.input_size == 16 and cfg.model.in_channels == 384
    assert cfg.model.label_dim == 1000
    assert cfg.vae.enabled and cfg.vae.kind == "rae" and cfg.vae.rae_size == "small"
    assert cfg.loss.name == "guided_cbm" and cfg.dataset.name == "imagenet"
    assert cfg.repa.enabled is False and cfg.signed.enabled is False
    for removed in ("fixed_t", "imf", "feature_mean", "latent_stats", "cfg_distill", "time_sampling"):
        assert removed not in cfg.loss, removed
    assert "phi" not in cfg.loss.guided_cbm and "profiler" not in cfg


def test_record_experiment_holds_the_record_recipe():
    cfg = compose_cfg(overrides=["+experiment=rae_record"])
    g = cfg.loss.guided_cbm
    assert g.time_shift == 4.899 and g.transport.weight == 1 and g.balance.mode == "freeze"
    assert g.guide.start_step == 35000 and g.guide.kappa == 0.7273 and g.guide.autoguide.enabled and g.guide.autoguide.weight == 0.0909
    assert cfg.freeze.trainable == "head_last4" and cfg.lr.val == 1.0e-5 and cfg.muon.adam_lr == 1.0e-6
    assert cfg.trainer.num_train_steps == 1042500 and cfg.trainer.global_batch_size == 256


def test_override_sets_compose():
    """Full command-line override sets (the RAE latent with the guided objective, and the SD-VAE latent pack)."""
    rae = ("model=dit_xl_1_dh data_dir=/tmp/imagenet trainer.num_nodes=1 trainer.devices=auto trainer.strategy=ddp "
           "signed.alpha=0 signed.enabled=false trainer.per_device_batch_size=32 dataset=imagenet model.input_size=16 "
           "model.in_channels=384 vae.enabled=true vae.kind=rae vae.rae_size=small vae.latents_scale=1.0 vae.latents_bias=0.0 "
           "dataset.use_precomputed_latents=false trainer.global_batch_size=256 trainer.checkpoint_every_n_steps=2500 "
           "trainer.num_workers=14 lr.warmup_steps=0 trainer.gradient_clip_val=null model.dropout=0.0 sampling.enabled=false "
           "model.attn_func=default loss.name=guided_cbm loss.guided_cbm.time_shift=4.899 loss.guided_cbm.guide.start_step=35000 "
           "loss.guided_cbm.guide.kappa=0.7273 loss.guided_cbm.balance.mode=freeze loss.guided_cbm.transport.weight=1 "
           "trainer.compile.teacher=true trainer.compile.student_fwd=true trainer.compile.student_jvp=true muon.batched=true "
           "trainer.ema.foreach=true trainer.num_train_steps=1042500 lr.val=0.00001 muon.adam_lr=0.000001 lr.force_on_resume=true "
           "freeze.trainable=head_last4 trainer.checkpoint_sparse_after=25000 trainer.checkpoint_sparse_every=2500").split()
    cfg = compose_cfg(overrides=rae)
    assert cfg.loss.guided_cbm.time_shift == 4.899 and cfg.vae.kind == "rae" and cfg.freeze.trainable == "head_last4"
    sdvae = ("model=dit_xl_1_dh signed.alpha=0 signed.enabled=false model.input_size=32 model.in_channels=4 "
             "vae.enabled=true vae.kind=sdvae vae.latents_scale=0.18215 vae.local_files_only=true latent_data_dir=/tmp/a "
             "latent_pack_dir=/tmp/b dataset.use_precomputed_latents=true").split()
    cfg = compose_cfg(overrides=sdvae)
    assert cfg.latent_pack_dir == "/tmp/b" and cfg.vae.local_files_only is True
    bridge = compose_cfg(overrides=["dataset=bridge", "model=bridge_unet", "loss.name=bridge_cbm", "vae.kind=sdvae",
                                    "vae.latents_scale=0.18215", "model.in_channels=4"])
    assert bridge.model.num_steps_conditioning == 4 and bridge.model.num_actions == 7
    rt1 = compose_cfg(overrides=["dataset=rt1", "model=bridge_unet", "loss.name=bridge_cbm"])
    assert rt1.dataset.name == "bridge" and rt1.model.num_actions == 10


def test_store_paths_follow_sbm_store(monkeypatch):
    monkeypatch.setenv("SBM_STORE", "/some/store")
    fid = compose_cfg("fid")
    assert fid.fid.extra_refs.adm == "/some/store/ckpt/imagenet/fid_cache/adm_jit_in256_stats.npz"
    assert compose_cfg(overrides=["dataset=bridge"]).dataset.processed_dir == "/some/store/data/bridge/processed/bridge_v2"
    assert compose_cfg(overrides=["dataset=rt1"]).dataset.processed_dir == "/some/store/data/rt1/processed/rt_1"
    rec = compose_cfg(overrides=["+experiment=rae_record"])
    assert rec.loss.guided_cbm.guide.autoguide.ckpt.startswith("/some/store/rae/refs/")
    monkeypatch.delenv("SBM_STORE")
    assert compose_cfg("fid").fid.extra_refs.adm.startswith(str(Path(sbm_rae.paths.REPO) / "store" / "ckpt" / "imagenet") + "/")
    monkeypatch.delenv("SBM_IMAGENET_DIR", raising=False)
    assert compose_cfg().data_dir == str(Path(sbm_rae.paths.REPO) / "store" / "data" / "imagenet_1k")
    monkeypatch.setenv("SBM_IMAGENET_DIR", "/some/imagenet")
    assert compose_cfg().data_dir == "/some/imagenet"


def test_paths_helper(monkeypatch):
    from sbm_rae import paths

    monkeypatch.delenv("SBM_STORE", raising=False)
    for var in ("SBM_FID_CACHE", "SBM_RAE_ASSETS", "SBM_RAE_SRC"):
        monkeypatch.delenv(var, raising=False)
    assert paths.store_root() == REPO / "store"
    assert paths.rae_src_dir() == REPO / "third_party" / "RAE" / "src"
    assert (paths.rae_src_dir() / "stage1" / "rae.py").is_file(), "the vendored RAE source tree"
    monkeypatch.setenv("SBM_STORE", "/x")
    assert paths.store_root() == Path("/x")
    assert paths.fid_cache_dir() == Path("/x/ckpt/imagenet/fid_cache")
    assert paths.rae_assets_dir() == Path("/x/rae/assets/rae")
    monkeypatch.setenv("SBM_FID_CACHE", "/y")
    assert paths.fid_cache_dir() == Path("/y")
    monkeypatch.setenv("SBM_RAE_SRC", "/w")
    assert paths.rae_src_dir() == Path("/w")


def test_loss_factory_dispatches_to_the_kept_objectives():
    from sbm_rae.bridge_loss import BridgeCBMLoss
    from sbm_rae.guided_cbm import GuidedCBMLoss
    from sbm_rae.loss import get_loss_fn

    assert isinstance(get_loss_fn(compose_cfg()), GuidedCBMLoss)
    bridge = compose_cfg(overrides=["dataset=bridge", "model=bridge_unet", "loss.name=bridge_cbm"])
    assert isinstance(get_loss_fn(bridge), BridgeCBMLoss)
    for removed in ("beckmann_transport", "imf", "fixed_t_regression"):
        with pytest.raises(ValueError, match="removed"):
            get_loss_fn(compose_cfg(overrides=[f"loss.name={removed}"]))
    with pytest.raises(NotImplementedError, match="phi"):
        get_loss_fn(compose_cfg(overrides=["+loss.guided_cbm.phi.enabled=true"]))


def test_world_model_experiments_hold_the_paper_recipes(monkeypatch):
    """experiment/wm_bridge and wm_rt1: the training values of the released Bridge-V2 and RT-1 models (the constant-rate
    stage; the cool-down is a resume with the cosine overrides), on the SD3 VAE (vae.kind sdvae, not train.yaml's default)."""
    monkeypatch.setenv("SBM_STORE", "/some/store")
    b = compose_cfg(overrides=["+experiment=wm_bridge", "model=bridge_unet"])
    r = compose_cfg(overrides=["+experiment=wm_rt1", "model=bridge_unet"])
    for cfg in (b, r):
        g = cfg.loss.guided_cbm
        assert cfg.model.name == "Bridge_UNet" and cfg.model.in_channels == 16 and cfg.dataset.name == "bridge"
        assert cfg.vae.kind == "sdvae" and cfg.vae.subfolder == "vae" and cfg.vae.latents_scale == 1.5305
        assert cfg.vae.pretrained_model_name_or_path == "stabilityai/stable-diffusion-3-medium-diffusers"
        assert cfg.loss.name == "bridge_cbm" and g.time_shift == 1.0 and g.transport.weight == 30 and g.balance.mode == "none"
        assert g.guide.enabled is False and g.label_dropout == 0.1 and cfg.optimizer == "adamw" and cfg.weight_decay == 0.01
        assert cfg.trainer.per_device_batch_size == 64 and cfg.trainer.ema.active_decay == 0.999 and cfg.trainer.gradient_clip_val == 2.0
    assert b.dataset.action_dim == b.model.num_actions == 7 and b.lr.val == 2.0e-4 and b.trainer.global_batch_size == 128
    assert b.trainer.num_train_steps == 600000 and b.dataset.latent_cache_dir == "/some/store/data/bridge/latents_sd3"
    assert r.dataset.action_dim == r.model.num_actions == 10 and r.lr.val == 4.0e-4 and r.trainer.global_batch_size == 256
    assert r.trainer.num_train_steps == 800000 and r.dataset.latent_cache_dir == "/some/store/data/rt1/latents_sd3"
    assert r.dataset.processed_dir == "/some/store/data/rt1/processed/rt_1"
