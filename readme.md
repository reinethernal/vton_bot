
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

3. *(Optional)* Gather pretrained checkpoints into a single directory:

   ```bash
   python collect_checkpoints.py
   ```

   The script copies the various model files into `checkpoints_collected` by default.
   Set the `TARGET_DIR` environment variable or pass `--target-dir` to change the location.

## Running

Launch the bot with:

```bash
python main.py
```

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

Install `pytest` (already included in the conda environment) and run:

```bash
pytest
```

