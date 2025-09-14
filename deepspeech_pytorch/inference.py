import json
from typing import List

import hydra
import torch
from torch.cuda.amp import autocast

from deepspeech_pytorch.configs.inference_config import TranscribeConfig
from deepspeech_pytorch.decoder import Decoder
from deepspeech_pytorch.loader.data_loader import ChunkSpectrogramParser
from deepspeech_pytorch.model import DeepSpeech
from deepspeech_pytorch.utils import load_decoder, load_model


def decode_results(decoded_output: List,
                   decoded_offsets: List,
                   cfg: TranscribeConfig):
    results = {
        "output": [],
        "_meta": {
            "acoustic_model": {
                "path": cfg.model.model_path
            },
            "language_model": {
                "path": cfg.lm.lm_path
            },
            "decoder": {
                "alpha": cfg.lm.alpha,
                "beta": cfg.lm.beta,
                "type": cfg.lm.decoder_type.value,
            }
        }
    }

    for b in range(len(decoded_output)):
        for pi in range(min(cfg.lm.top_paths, len(decoded_output[b]))):
            result = {'transcription': decoded_output[b][pi]}
            if cfg.offsets:
                result['offsets'] = decoded_offsets[b][pi].tolist()
            results['output'].append(result)
    return results


def transcribe(cfg: TranscribeConfig):
    device = torch.device("cuda" if cfg.model.cuda else "cpu")

    model = load_model(
        device=device,
        model_path=cfg.model.model_path
    )

    decoder = load_decoder(
        labels=model.labels,
        cfg=cfg.lm
    )

    spect_parser = ChunkSpectrogramParser(
        audio_conf=model.spect_cfg,
        normalize=True
    )

    decoded_output, decoded_offsets, _ = run_transcribe(
        audio_path=hydra.utils.to_absolute_path(cfg.audio_path),
        spect_parser=spect_parser,
        model=model,
        decoder=decoder,
        device=device,
        precision=cfg.model.precision,
        chunk_size_seconds=cfg.chunk_size_seconds
    )
    results = decode_results(
        decoded_output=decoded_output,
        decoded_offsets=decoded_offsets,
        cfg=cfg
    )
    print(json.dumps(results))


def run_transcribe(audio_path: str,
                   spect_parser: ChunkSpectrogramParser,
                   model: DeepSpeech,
                   decoder: Decoder,
                   device: torch.device,
                   precision: int,
                   chunk_size_seconds: float):
    hs = None # means that the initial RNN hidden states are set to zeros
    all_outs = []
    with torch.no_grad():
        for spect in spect_parser.parse_audio(audio_path, chunk_size_seconds):
            spect = spect.contiguous()
            spect = spect.view(1, 1, spect.size(0), spect.size(1))
            spect = spect.to(device)
            input_sizes = torch.IntTensor([spect.size(3)]).int()
            with autocast(enabled=precision == 16):
                out, output_sizes, hs = model(spect, input_sizes, hs)
            all_outs.append(out.cpu())
    all_outs = torch.cat(all_outs, axis=1) # combine outputs of chunks in one tensor
    


    '''
    import copy

    ori_input = copy.deepcopy(all_outs)
    ori_input = ori_input.transpose(0, 1)

    print(ori_input)

    print(f"ori_input.shape {ori_input.shape}")

    target_text = "idiot my name is jack"

    # 把target——text 转成对应的字符索引
    target = []
    for c in target_text:
        # c转大写
        # print(str.upper(c))
        target.append(model.labels.index(str.upper(c)))
    # print(f"target {target}")
    target_output = torch.tensor([target])

    # blank的标记索引是0
    ctcloss = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)


    print(ori_input.shape)
    print(target_output.shape)
    print(torch.tensor([ori_input.size(0)]))
    print(torch.tensor([len(target)]))

    loss = ctcloss(ori_input, target_output, torch.tensor([ori_input.size(0)]), torch.tensor([len(target)]))
    print(f"loss {loss}")
    '''
    



    # decoded_output就是解码出来的文本, decoded_offsets还没看懂是什么
    decoded_output, decoded_offsets = decoder.decode(all_outs)

    # all_outs shape是[batch_size, length, num_classes], 存的是每个位置上, 每个字符对应的概率
    return decoded_output, decoded_offsets, all_outs
