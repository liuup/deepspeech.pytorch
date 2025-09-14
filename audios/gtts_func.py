import argparse
import io
from pydub import AudioSegment
from gtts import gTTS


parser = argparse.ArgumentParser()
parser.add_argument("-text", type=str, help="original text")
parser.add_argument("-out", type=str, help="the name of .wav file", default="output.wav")
args = parser.parse_args()


res = gTTS(text=args.text, lang='en')
mp3_fp = io.BytesIO()
res.write_to_fp(mp3_fp)
mp3_fp.seek(0)

sound = AudioSegment.from_file(mp3_fp, format="mp3")
sound.export(args.out, format="wav")