import tts_edge as edge
from pydub import AudioSegment
import playsound
# import simpleaudio as sa
from datetime import datetime
from scipy.io import wavfile


# SAMPLE_RATE = 44100
# SAMPLE_WIDTH = 2
# CHANNELS = 2

# SUPPORTED_VOICES = {
#     'en-US-AnaNeural': 'en-US',
#     'en-US-AriaNeural': 'en-US',
#     'en-US-ChristopherNeural': 'en-US',
#     'en-US-EricNeural': 'en-US',
#     'en-US-GuyNeural': 'en-US',
#     'en-US-JennyNeural': 'en-US',
#     'en-US-MichelleNeural': 'en-US',
#     'ko-KR-InJoonNeural': 'ko-KR',
#     'ko-KR-SunHiNeural': 'ko-KR'
# }

SUPPORTED_VOICES = [
    'en-US-AnaNeural',
    'en-US-AriaNeural',
    'en-US-ChristopherNeural',
    'en-US-EricNeural',
    'en-US-GuyNeural',
    'en-US-JennyNeural',
    'en-US-MichelleNeural',
    'ko-KR-InJoonNeural',
    'ko-KR-SunHiNeural'
]



def get_filename_from_edge():
    now = datetime.now()

    str_now = now.strftime('%Y%m%d_%H%M%S')
    str_path = "./voice/"
    filename = str_path + str_now + '_edge_answer.mp3'

    return filename


def save_mp3_file(raw, f_name):
    raw_size = len(raw)
    with open(f_name, "wb") as binary_file:
        write_size = binary_file.write(raw)
    
    if(raw_size == write_size):
        # print(f'raw: {raw_size}')
        # print(f'write: {write_size}')
        # print('file creation is successful')
        return True
    else:
        return False



def edge_tts_generate(input: str, type: str, rate: int):
    data = {
        'text': input,
        'voice': type,
        'rate': rate
    }

    try:
        audio_raw_data = edge.generate_audio(text=data["text"], voice=data["voice"], rate=data["rate"])
        # print(audio_raw_data)
        
        # return Response(audio, mimetype="audio/mpeg")

        # audio_segment = AudioSegment(
        #     data=audio_raw_data,
        #     sample_width=SAMPLE_WIDTH,    # Sample width in bytes
        #     frame_rate=SAMPLE_RATE,       # Frame rate
        #     channels=CHANNELS             # Mono
        # )

        #audio_segment.export(filename, format='wav')

        filename = get_filename_from_edge()
        result = save_mp3_file(audio_raw_data, filename)
        print(f'tts-result: {result}')

        return filename

    except Exception as e:
        print(e)
        return None
        #abort(500, data["voice"])


input_text = "Hello! How can I help you? I am a Text To Speech Service"
# input_text = 'this is test text and welcome! thank you! English is good language!'
# input_text = 'thank you! there is my world!'

voice_type = SUPPORTED_VOICES[5]
speed = 10

f_name = edge_tts_generate(input_text, voice_type, speed)
print(f_name)        

playsound.playsound(f_name)

# fs, data = wavfile.read(f_name)
# print(fs, data.shape)







