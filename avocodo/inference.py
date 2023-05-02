import argparse
import os

import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from scipy.io.wavfile import write
from torch.utils.data import DataLoader

from avocodo.data_module import AvocodoData
from avocodo.lightning_module import Avocodo
from avocodo.meldataset import MAX_WAV_VALUE, MelDataset, mel_spectrogram

h = None
device = None


def get_mel(x):
    return mel_spectrogram(x, 1024, 80, 22050, 256, 1024, 0, 8000)


def inference(a, conf):
    avocodo = Avocodo.load_from_checkpoint(
        f"{a.checkpoint_path}/version_{a.version}/checkpoints/{a.checkpoint_file_id}",
        map_location="cpu",
    )
    avocodo_data = AvocodoData(conf.audio)
    avocodo_data.prepare_data()
    dls = []
    dls.append(avocodo_data.val_dataloader())

    if a.all is not None:
        ob = avocodo_data
        trainset = MelDataset(
            ob.training_filelist,
            ob.hparams.segment_size,
            ob.hparams.n_fft,
            ob.hparams.num_mels,
            ob.hparams.hop_size,
            ob.hparams.win_size,
            ob.hparams.sampling_rate,
            ob.hparams.fmin,
            ob.hparams.fmax,
            False,
            False,
            n_cache_reuse=0,
            fmax_loss=ob.hparams.fmax_for_loss,
            fine_tuning=ob.hparams.fine_tuning,
            base_mels_path=ob.hparams.input_mels_dir,
        )
        dls.append(
            DataLoader(
                trainset,
                num_workers=ob.hparams.num_workers,
                shuffle=False,
                sampler=None,
                batch_size=1,
                pin_memory=True,
                drop_last=True,
            )
        )

    output_path = f"{a.output_dir}/version_{a.version}/"
    os.makedirs(output_path, exist_ok=True)

    avocodo.generator.to(a.device)
    avocodo.generator.remove_weight_norm()

    m = torch.jit.script(avocodo.generator)
    torch.jit.save(m, os.path.join(output_path, "scripted.pt"))

    with torch.no_grad():
        for dataloader in dls:
            for i, batch in enumerate(dataloader):
                mels, _, file_ids, _ = batch

                y_g_hat = avocodo(mels.to(a.device))

                for _y_g_hat, file_id in zip(y_g_hat, file_ids):
                    audio = _y_g_hat.squeeze(0)
                    audio = audio * MAX_WAV_VALUE
                    audio = audio.cpu().numpy().astype("int16")

                    output_file = os.path.join(output_path, file_id.split("/")[-1])
                    print(file_id, flush=True)
                    write(output_file, conf.audio.sampling_rate, audio)
    print("Done inference")


def main():
    print("Initializing Inference Process..")

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", default="logs/Avocodo")
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--checkpoint_file_id", type=str, default="", required=True)
    parser.add_argument("--output_dir", type=str, default="generated_files")
    parser.add_argument("--script", type=bool, default=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--all", action="store_true")
    a = parser.parse_args()
    print(a)

    conf = OmegaConf.load(
        os.path.join(a.checkpoint_path, f"version_{a.version}", "hparams.yaml")
    )
    inference(a, conf)


if __name__ == "__main__":
    main()
