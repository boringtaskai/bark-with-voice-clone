from IPython.display import Audio
from scipy.io.wavfile import write as write_wav

from bark.api import generate_audio
from bark.generation import SAMPLE_RATE, preload_models, codec_decode, generate_coarse, generate_fine, generate_text_semantic

semantic_path = None # "semantic_output/pytorch_model.bin" # set to None if you don't want to use finetuned semantic
coarse_path = None # "coarse_output/pytorch_model.bin" # set to None if you don't want to use finetuned coarse
fine_path = None # "fine_output/pytorch_model.bin" # set to None if you don't want to use finetuned fine
use_rvc = False # Set to False to use bark without RVC
rvc_name = 'mi-test'
rvc_path = f"Retrieval-based-Voice-Conversion-WebUI/weights/rvc_name.pth"
index_path = f"Retrieval-based-Voice-Conversion-WebUI/logs/rvc_name/added_IVF256_Flat_nprobe_1_rvc_name_v2.index"
device="cuda:0"
is_half=True

# download and load all models
preload_models(
    text_use_gpu=True,
    text_use_small=False,
    text_model_path=semantic_path,
    coarse_use_gpu=True,
    coarse_use_small=False,
    coarse_model_path=coarse_path,
    fine_use_gpu=True,
    fine_use_small=False,
    fine_model_path=fine_path,
    codec_use_gpu=True,
    force_reload=False,
    path="models"
)

# if use_rvc:
#     from rvc_infer import get_vc, vc_single
#     get_vc(rvc_path, device, is_half)

# simple generation
text_prompt = "Halo aku Wawan Budi Santoso, aku suka pergi jalan-jalan."
voice_name = "speaker_1" # use your custom voice name here if you have on

filepath = "output/audio.wav"
audio_array = generate_audio(text_prompt, history_prompt=voice_name, text_temp=0.7, waveform_temp=0.7)
write_wav(filepath, SAMPLE_RATE, audio_array)

# if use_rvc:
#     index_rate = 0.75
#     f0up_key = -6
#     filter_radius = 3
#     rms_mix_rate = 0.25
#     protect = 0.33
#     resample_sr = SAMPLE_RATE
#     f0method = "harvest" #harvest or pm
#     try:
#         audio_array = vc_single(0,filepath,f0up_key,None,f0method,index_path,index_rate, filter_radius=filter_radius, resample_sr=resample_sr, rms_mix_rate=rms_mix_rate, protect=protect)
#     except:
#         audio_array = vc_single(0,filepath,f0up_key,None,'pm',index_path,index_rate, filter_radius=filter_radius, resample_sr=resample_sr, rms_mix_rate=rms_mix_rate, protect=protect)
#     write_wav(filepath, SAMPLE_RATE, audio_array)

# Audio(audio_array, rate=SAMPLE_RATE)