# Official Upstream Baseline Sources (Vendored Verbatim)

These modules are copied verbatim (without altering network architectures or official losses) from the official upstream GitHub repositories for reproducible Queue-2 benchmarking and accurate paper citation:

1. **pytorch-CycleGAN-and-pix2pix** (`Pix2Pix`, `CycleGAN`)
   - Repository: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
   - Commit: `2a7afba`
   - Vendored files under `pytorch_CycleGAN_and_pix2pix/`:
     - `models/__init__.py`
     - `models/base_model.py`
     - `models/networks.py`
     - `models/pix2pix_model.py` (`Pix2PixModel`, official `unet_256`, `basic` PatchGAN `netD`, `vanilla` GAN + `100 * L1`)
     - `models/cycle_gan_model.py` (`CycleGANModel`, official `resnet_9blocks`, `basic` PatchGAN `netD`, `lsgan`, `lambda_A=10.0`, `lambda_B=10.0`, `lambda_identity=0.5`, `pool_size=50`)
     - `util/__init__.py`
     - `util/image_pool.py` (`ImagePool`)

2. **UNIStainNet** (`SPADEUNetGenerator`, `PatchDiscriminator`, `MultiScaleDiscriminator`)
   - Repository: https://github.com/facevoid/UNIStainNet
   - Commit: `bf3576b`
   - Vendored files under `UNIStainNet/`:
     - `src/__init__.py`
     - `src/models/__init__.py`
     - `src/models/blocks.py` (`SPADEBlock`, `ResBlock`, `SelfAttention`)
     - `src/models/discriminator.py` (`PatchDiscriminator`, `MultiScaleDiscriminator`, `hinge_loss_d`, `hinge_loss_g`, `r1_gradient_penalty`, `feature_matching_loss`)
     - `src/models/edge_encoder.py` (`EdgeEncoder`, `MultiScaleEdgeEncoder`)
     - `src/models/generator.py` (`SPADEUNetGenerator`)
     - `src/models/losses.py` (`VGGFeatureExtractor`, `gram_matrix`, `PatchNCELoss`)
     - `src/models/uni_processor.py` (`UNIFeatureProcessor`, `UNIFeatureProcessorHighRes`)
