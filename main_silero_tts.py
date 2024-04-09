#@title Install dependencies

# !pip install -q torchaudio omegaconf

import torch
from pprint import pprint
from omegaconf import OmegaConf
from IPython.display import Audio, display
import wave

import contextlib
import playsound
from datetime import datetime


SAMPLE_RATE = 48000
SAMPLE_WIDTH = 2
CHANNELS = 1

def write_wave(path, audio, sample_rate):
    """Writes a .wav file.
    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


def get_filename_from_silero():
    now = datetime.now()

    str_now = now.strftime('%Y%m%d_%H%M%S')
    str_path = "./voice/"
    filename = str_path + str_now + '_silero_answer.wav'

    return filename

# Can Read avaiable models and available languages

# torch.hub.download_url_to_file('https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml',
#                                'latest_silero_models.yml',
#                                progress=False)
# models = OmegaConf.load('latest_silero_models.yml')

# # see latest avaiable models
# available_languages = list(models.tts_models.keys())
# print(f'Available languages {available_languages}')

# for lang in available_languages:
#     _models = list(models.tts_models.get(lang).keys())
#     print(f'Available models for {lang}: {_models}')

language = 'en'
model_id = 'v3_en'
# model_id = 'v3_en_indic'
# model_id = 'lj_v2'
# model_id = 'lj_16khz'

device = torch.device('cpu')

model, example_text = torch.hub.load(repo_or_dir='',
                                     model='silero_tts',
                                     language=language,
                                     speaker=model_id,
                                     source='local')
model.to(device)  # gpu or cpu
model.speakers

speaker = 'en_0'                # woman  
# speaker = 'xenia'             # russia woman?
# speaker = 'en_0'              # young woman 
# speaker = 'tamil_female'      # india woman
# speaker = 'en_3'                # old woman 
# speaker = 'en_5'                # young woman and mechanical
# speaker = 'en_6'                # woman  


put_accent=True
put_yo=True
example_text = "Hello! How can I help you? I am a Text To Speech Service"
# example_text = 'В недрах тундры выдры в г+етрах т+ырят в вёдра ядра к+едров.'

audio = model.apply_tts(text=example_text,
                        speaker=speaker,
                        sample_rate=SAMPLE_RATE,
                        put_accent=put_accent,
                        put_yo=put_yo)

# Not working. why?
# display(Audio(audio, rate=sample_rate))

# save test.wav file 
# audio_paths = model.save_wav(text=example_text,
#                              speaker=speaker,
#                              sample_rate=sample_rate,
#                              put_accent=put_accent,
#                              put_yo=put_yo)

f_name = get_filename_from_silero()

# Why do we use 32767?
# Assuming that the wav-file is 16 bit integer, the range is [-32768, 32767], thus dividing by 32768 (2^15) will give the proper twos-complement range of [-1, 1]
# Another answer is that -32767 to +32767 is proper audio (to be symmetrical) and 32768 means that the audio clipped at that point
write_wave(path=f_name, audio=(audio * 32767).numpy().astype('int16'), sample_rate=SAMPLE_RATE)
print(f_name)

playsound.playsound(f_name)