'''
How to run transcribe.py

python transcribe.py model.model_path=./weights/librispeech_pretrained_v3.ckpt model.cuda=True chunk_size_seconds=-1 audio_path=./audios/jack.wav
'''


'''
How to run attack.py

python attack.py -target "idiot my name is jack" \
-audio ./audios/jack.wav \
-lr 0.001 \
-steps 1000 \
-l2penalty 0 \
-output_path ./outputs/

'''


'''
How to run in background

python attack.py &
ps -ef | grep attack.py
'''

import os
import pytz
import logging
import datetime
import copy
import torch
import argparse
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

from torch import nn

from deepspeech_pytorch.utils import load_decoder, load_model
from deepspeech_pytorch.configs import inference_config
from deepspeech_pytorch.loader.data_loader import ChunkSpectrogramParser
from deepspeech_pytorch.inference import run_transcribe


# 命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("-target", type=str, help="target transcription", default="idiot my name is jack")
parser.add_argument("-audio", type=str, help="initial .wav file to use", default="./audios/jack.wav")
parser.add_argument("-model_path", type=str, help="model checkpoint to use", default="./weights/librispeech_pretrained_v3.ckpt")
parser.add_argument("-lr", type=float, help="learning rate", default=0.05)
parser.add_argument("-steps", type=int, help="Maximum number of steps of gradient calculation", default=10000)
parser.add_argument("-l2penalty", type=float, help="l2 penalty", default=0)
parser.add_argument("-output_path", type=str, help="adversarial example output path", default="./outputs/")
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# logger setup
now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
timezone = datetime.datetime.now().astimezone().tzinfo
logging.basicConfig(
    filename=args.output_path + f"{now}_{timezone}_attack.log",
    level=logging.INFO,
    format=f'%(asctime)s {timezone} - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_all_outs_with_grad(audio_tensor, model, spect_parser):
    """保持梯度的版本，用于攻击训练"""
    all_outs = []
    
    # 直接使用tensor进行谱图转换，保持梯度连接

    # 首先归一化音频到[-1, 1]范围
    if torch.max(torch.abs(audio_tensor)) > 1.0:
        audio_normalized = audio_tensor / 32768.0
    else:
        audio_normalized = audio_tensor
    
    # 使用torchaudio的谱图变换保持梯度
    n_fft = int(spect_parser.sample_rate * spect_parser.window_size)
    win_length = n_fft
    hop_length = int(spect_parser.sample_rate * spect_parser.window_stride)
    
    # 使用torch.stft代替librosa.stft以保持梯度
    stft = torch.stft(audio_normalized, 
                      n_fft=n_fft, 
                      hop_length=hop_length, 
                      win_length=win_length, 
                      window=torch.hann_window(win_length).to(device),
                      return_complex=True)
    
    # 计算幅度谱
    spect = torch.abs(stft)
    # log(S+1)
    spect = torch.log1p(spect + 1e-8)
    
    if spect_parser.normalize:
        mean = spect.mean()
        std = spect.std()
        std = torch.clamp(std, min=1e-8)
        spect = (spect - mean) / std
    
    # 调整维度以匹配模型输入格式
    spect = spect.contiguous()
    spect = spect.view(1, 1, spect.size(0), spect.size(1))
    spect = spect.to(device)
    input_sizes = torch.IntTensor([spect.size(3)]).int()
    
    out, _, _ = model(spect, input_sizes, None)
    all_outs.append(out)
    
    if len(all_outs) > 1:
        all_outs = torch.cat(all_outs, axis=1)
    else:
        all_outs = all_outs[0]
    
    return all_outs

def attack(audio, ori_length, target):

    audio_tensor = torch.tensor(audio).float().to(device)


    # 16bit音频取值为[-32768, 32767], 也就是[-2**15, 2**15-1], 原代码的adv.wav保存的时候就是这么处理的
    bit_high = 2**15 - 1
    bit_low = -2**15

    # delta = torch.randn(audio_tensor.shape).to(device)
    delta = torch.zeros_like(audio_tensor).to(device)
    delta.requires_grad = True
    
    optimizer = torch.optim.Adam([delta], lr=args.lr)
    
    # 1000steps没下降就把学习率减半
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1000, verbose=True)

    ctcloss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

    # 加载模型和相关组件
    model = load_model(
        device=device,
        model_path=args.model_path
    ).to(device).train()

    decoder = load_decoder(
        labels=model.labels,
        cfg=inference_config.LMConfig()
    )

    spect_parser = ChunkSpectrogramParser(
        audio_conf=model.spect_cfg,
        normalize=True
    )

    '''
    decoded_output 就是识别成功的文本
    decoded_offsets 还没看懂是什么
    outs_probs 是模型输出的每个位置上每个字符的概率
    '''
    # decoded_output, decoded_offsets, outs_probs = run_transcribe(
    #     audio_path=args.audio,
    #     spect_parser=spect_parser,
    #     model=model,
    #     decoder=decoder,
    #     device=device,
    #     precision=32,
    #     chunk_size_seconds=-1
    # )
    # print(f"transcribed result: {decoded_output[0]}")
    # return


    all_outs = get_all_outs_with_grad(audio_tensor, model, spect_parser)
    decoded_output, decoded_offsets = decoder.decode(all_outs)
    logger.info(f"Transcribed result from audio: {decoded_output[0]}")
    # print(decoded_offsets[0])


    # 把target 转成对应的字符索引
    targets = []
    for c in target:
        targets.append(model.labels.index(c))
    target_output = torch.tensor([targets])


    loss_draw = []
    step_draw = []

    logger.info("\n----- ----- attack start ----- -----")
    for step in range(args.steps):
        optimizer.zero_grad()

        adv_audio = audio_tensor + delta
        # adv_audio = audio_tensor
        adv_audio = torch.clamp(adv_audio, bit_low, bit_high)
        

        all_outs = get_all_outs_with_grad(adv_audio, model, spect_parser)

        all_outs_trans = all_outs.transpose(0, 1) # (time, batch, n_class)

        loss_ctc = ctcloss(all_outs_trans, target_output, torch.tensor([all_outs_trans.size(0)]), torch.tensor([len(targets)]))
        loss_l2 = torch.norm(delta, p=2)

        loss = loss_ctc + args.l2penalty * loss_l2
        loss.backward()


        # 监控梯度
        grad_norm = torch.norm(delta.grad).item()
        # torch.nn.utils.clip_grad_norm_(delta.grad, max_norm=1.0)
        

        optimizer.step()
        
        scheduler.step(loss)

        if step % 100 == 0:
            with torch.no_grad():
                test_outs = get_all_outs_with_grad(adv_audio, model, spect_parser)
                decoded_output, _ = decoder.decode(test_outs)

                # output_text = decoded_output[0][0]
                
                if args.l2penalty == 0:
                    # print(f"step {step:05d}/{args.steps}, lr {optimizer.param_groups[0]['lr']:.6f}, loss {loss:.4f} [ctc {loss_ctc:.4f}], grad_norm {grad_norm:.4f}, text {output_text}")
                    logger.info(f"step {step:05d}/{args.steps}, lr {optimizer.param_groups[0]['lr']}, loss {loss:.4f} [ctc {loss_ctc:.4f}], grad_norm {grad_norm:.4f}, text {decoded_output[0]}")
                else:
                    # print(f"step {step:05d}/{args.steps}, lr {optimizer.param_groups[0]['lr']:.6f}, loss {loss:.4f} [ctc {loss_ctc:.4f}, l2 {loss_l2:.4f}], grad_norm {grad_norm:.4f}, text {output_text}")
                    logger.info(f"step {step:05d}/{args.steps}, lr {optimizer.param_groups[0]['lr']}, loss {loss:.4f} [ctc {loss_ctc:.4f}, l2 {loss_l2:.4f}], grad_norm {grad_norm:.4f}, text {decoded_output[0]}")
                
                # 绘制loss曲线
                loss_draw.append(loss.item())
                step_draw.append(step)
                
                plt.plot(step_draw, loss_draw)
                plt.xlabel('Step')
                plt.ylabel('Loss')
                plt.title('Loss Curve During Attack')
                plt.savefig(args.output_path + f"loss_curve.png")
                plt.close()

                # 判断一下成功没
                if decoded_output[0][0] == target:
                    # print("Attack success!")
                    logger.info("----- ----- Attack success! ----- -----")
                    adv_audio_np = adv_audio.detach().cpu().numpy().astype(np.int16)
                    wav.write(args.output_path + "adv.wav", 16000, adv_audio_np)
                    # print(f"Adversarial example saved to {args.output_path + 'adv.wav'}")
                    logger.info(f"Adversarial example saved to {args.output_path + 'adv.wav'}")
                    return
    
    logger.info("----- ----- Attack finished ----- -----")


                


if __name__ == "__main__":
    logger.info(f"Timezone: {datetime.datetime.now(pytz.utc).astimezone().tzinfo}")


    # audio是读取后转的numpy数组 
    fs, audio = wav.read(args.audio)

    assert fs == 16000  # Check the sampling rate is 16kHz
    assert audio.dtype == np.int16  # Check the audio is in 16-bit format
    
    # 如果为双声道, 只保留单声道
    if audio.shape[-1] == 2:
        audio = np.squeeze(audio[:,1])
        # print(audio.shape)

    # 分贝数
    source_dB = 20 * np.log10(np.max(np.abs(audio)))
    # print(f'source dB: {source_dB}')

    logger.info(f'source dB: {source_dB}')

    # print(f'Taget length: {len(args.target)} Target transcription: {str.upper(args.target)}')
    logger.info(f'Target text: {str.upper(args.target)}')

    # print out all cli commands
    logger.info(vars(args))

    # 因为用的是model.labels, 也就是labels.json, 所以里面都是大写字母, 所以要c转大写
    attack(audio, len(audio), str.upper(args.target))



