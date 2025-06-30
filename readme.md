
# Virtual Try-On Telegram Bot

This project provides a simple Telegram bot that performs virtual try-on using
PyTorch. It automatically uses CUDA when available, so it will run on GPUs such
as the NVIDIA RTX A4000.

## Setup

1. Create a Python environment and install dependencies:

   ```bash
   conda env create -f environment.yml
   conda activate vton_bot
   ```

2. Set the required environment variables:

    - `BOT_TOKEN` – token for your Telegram bot.
    - `UNIFORMS` – JSON mapping uniform names to image paths.
    - `TARGET_DIR` – _(optional)_ directory used by `collect_checkpoints.py`.

   Example `.env` file:

   ```env
   BOT_TOKEN=123456:ABCDEF
   UNIFORMS={"Uniform 1": "static/uniforms/uniform1.png"}
```

3. Download the pretrained model checkpoints. The U^2-Net models are available on the
   [U^2-Net release page](https://github.com/xuebinqin/U-2-Net/releases), and CatVTON
   and VITON-HD checkpoints are provided in their respective repositories. The OpenPose
   BODY_25 and hand models can be downloaded directly from Google Drive:

   ```bash
   # BODY_25 (~200 MB)
    gdown https://drive.google.com/uc?id=1EULkcH_hhSU28qVc1jSJpCh2hGOrzpjK -O \
        openpose/models/body_pose_model.pth

   # Hand (~25 MB)
    gdown https://drive.google.com/uc?id=1yVyIsOD32Mq28EHrVVlZbISDN7Icgaxw -O \
        openpose/models/hand_pose_model.pth
   ```

   Create the `openpose/models/` directory if it does not already exist, and
   place the other model files under `models/` as listed in the table below. If you
   store the OpenPose weights elsewhere, set the `OPENPOSE_MODEL_DIR` environment
   variable to the directory containing `body_pose_model.pth` so that
   `VTONPipeline` can locate them.

4. *(Optional)* Build the OpenPose Python module. Ensure that the system package
   providing `google/protobuf/runtime_version.h` (`libprotobuf-dev` on Ubuntu) is
   installed. The compiler (`protoc`) must come from the same package version as
   the library; check with `which protoc` and `protoc --version` (Ubuntu 22.04
   ships 3.12.4). If a Conda environment shadows `/usr/bin/protoc`, override it
   when running CMake:

   ```bash
   cmake .. -DPROTOBUF_PROTOC_EXECUTABLE=/usr/bin/protoc
   ```

   Once the prerequisite headers are present, run `install_openpose_ubuntu.sh`
   from the repository root. After the build completes, run `sudo make install`
   and then `python3 -m pip install -e python` inside the `openpose`
   directory. This installs the Python bindings required for `VTONPipeline` to
   load OpenPose.


5. *(Optional)* Gather pretrained checkpoints into a single directory:

   ```bash
   python collect_checkpoints.py
   ```

   The script copies the various model files into `checkpoints_collected` by default.
   Set the `TARGET_DIR` environment variable or pass `--target-dir` to change the location.

The following table lists the expected locations and approximate sizes of the
required model files:

| Path | Description | Approx. size |
| --- | --- | --- |
| `openpose/models/body_pose_model.pth` | OpenPose BODY\_25 weights | ~200 MB |
| `openpose/models/hand_pose_model.pth` | OpenPose hand weights | ~25 MB |
| `models/u2net.pth` | U\^2-Net base model | ~170 MB |
| `models/cloth_segm_u2net_latest.pth` | U\^2-Net garment segmentation | ~170 MB |
| `models/CatVTON/SCHP/exp-schp-201908261155-lip.pth` | CatVTON human parsing (LIP) | ~250 MB |
| `models/CatVTON/SCHP/exp-schp-201908301523-atr.pth` | CatVTON human parsing (ATR) | ~250 MB |
| `models/u2net_portrait.pth` | U\^2-Net portrait model | ~170 MB |
| `models/cloth_seg.pth` | Cloth segmentation | ~100 MB |
| `models/gmm_final.pth` | Geometric matching module | ~110 MB |
| `models/seg_final.pth` | Segmentation refinement | ~320 MB |
| `pytorch3d/tests/pulsar/reference/nr0000-in.pth` | PyTorch3D reference data | <1 MB |
| `pytorch3d/tests/pulsar/reference/nr0000-out.pth` | PyTorch3D reference data | <1 MB |
| `pytorch3d/tests/data/icp_data.pth` | PyTorch3D ICP sample | <1 MB |
| `pytorch3d/docs/tutorials/data/camera_graph.pth` | PyTorch3D tutorial sample | <1 MB |
| `VITON-HD/checkpoints/seg_final.pth` | VITON-HD segmentation model | ~260 MB |
| `VITON-HD/checkpoints/gmm_final.pth` | VITON-HD GMM model | ~190 MB |
| `VITON-HD/checkpoints/alias_final.pth` | VITON-HD alias generator | ~1.1 GB |

Expect to allocate roughly **3 GB** of space for these files.

## Running

Launch the bot with:

```bash
python main.py
```

By default the pipeline uses DeepLabV3 for garment segmentation. To use a
pretrained U2Net model instead, instantiate `VTONPipeline` with
`segmentation_model="u2net"` and ensure the weights file
`models/cloth_segm_u2net_latest.pth` is available.

The result images will be saved next to the uploaded user photos.

### Sample image

The repository no longer ships with a default person photo. If you want to
experiment with the `vton.py` example script or quickly check the pipeline, put
any JPEG image of a person in the project root and pass its path with the
`--person` flag:

```bash
python vton.py --person path/to/your_photo.jpg --cloth static/uniforms/uniform1.png
```

You may also name the file `temp_person.jpg` and omit the flag so the example
uses that image automatically.

## Tests

Running the tests requires the same Conda environment used by the bot. Make
sure you create and activate it first:

```bash
conda env create -f environment.yml
conda activate vton_bot
pytest
```

If some of the heavy dependencies such as PyTorch are not available you can set
`SKIP_HEAVY_TESTS=1` to run only the lightweight tests. Heavy tests will also be
automatically skipped when the required packages are missing.

