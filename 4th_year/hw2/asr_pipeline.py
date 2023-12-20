from abc import abstractmethod
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Union, Iterable
import os
from urllib.parse import urlparse, parse_qs

from pydub import AudioSegment
from pydub.silence import split_on_silence
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi

import numpy as np
from seamless_communication.inference import Translator
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import pipeline


class BaseASR:
    # dictionary with loaded models in order to not load model twice
    loaded_models = defaultdict(dict)

    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device

    @staticmethod
    def load_full_audio(audio_path, sample_rate=16000) -> torch.Tensor:
        """Load the audio file & convert to 16,000 sampling rate"""
        speech, sr = torchaudio.load(audio_path)
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        speech = resampler(speech)
        return speech.squeeze()

    @staticmethod
    def split_audio(audio: AudioSegment,
                    max_segment_duration=30,
                    min_silence_len=1000,
                    silence_thresh=-30,
                    target_sr=16000) -> Iterable[AudioSegment]:

        audio = audio.set_frame_rate(target_sr).set_channels(1)
        segments = split_on_silence(audio,
                                    min_silence_len=min_silence_len,
                                    silence_thresh=silence_thresh)
        result_segments = []
        for i, segment in enumerate(segments):
            if len(segment) > max_segment_duration * 1000:
                segments.pop(i)
                sub_segments = BaseASR.split_audio(segment,
                                                   max_segment_duration,
                                                   min_silence_len=min_silence_len - 100,
                                                   silence_thresh=silence_thresh,
                                                   target_sr=target_sr)
                result_segments.extend(sub_segments)
            else:
                result_segments.append(segment)
        return result_segments

    @staticmethod
    def audiosegment2rawdata(audio: AudioSegment) -> np.ndarray:
        audio = np.array(audio.get_array_of_samples())
        return audio / np.max(np.abs(audio))

    @abstractmethod
    def recognize(self,
                  audio_file: Union[str, os.PathLike],
                  file_format: str,
                  language: str) -> str:
        ...

    @abstractmethod
    def register_model(self, *args, **kwargs):
        """
        Adds model to loaded_models

        """
        ...


class WhisperASR(BaseASR):
    def __init__(self, model_name, device):
        super().__init__(model_name, device)
        self.register_model()

    def register_model(self):
        if self.model_name not in WhisperASR.loaded_models:
            whisper_processor = WhisperProcessor.from_pretrained(self.model_name)
            whisper_model = WhisperForConditionalGeneration.from_pretrained(self.model_name).to(self.device)

            WhisperASR.loaded_models[self.model_name] = {
                'processor': whisper_processor,
                'model': whisper_model,
                'ref_count': 1
            }
        else:
            WhisperASR.loaded_models[self.model_name]['ref_count'] += 1

    @property
    def model(self):
        return self.loaded_models[self.model_name]['model']

    @property
    def processor(self):
        return self.loaded_models[self.model_name]['processor']

    def recognize(self,
                  audio_file: Union[str, os.PathLike],
                  language: str = 'ko',
                  file_format: str = 'mp3') -> str:
        audio = AudioSegment.from_file(audio_file, format=file_format).set_frame_rate(16000).set_channels(1)
        segments = self.split_audio(audio, target_sr=16000)  # так как он может делать inference только для 30 секунд
        transcriptions = []
        for seg in segments:
            audio = self.audiosegment2rawdata(seg)
            input_features = self.processor(
                audio=audio, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(self.device)
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language, task="transcribe")
            predicted_ids = self.model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
            transcriptions.append(transcription)
        return ' '.join([part[0] for part in transcriptions])


class SeamlessM4TASR(BaseASR):
    def __init__(self, model_name, vocoder, device):
        super().__init__(model_name, device)
        self.register_model(vocoder)

    def register_model(self, vocoder: str):
        if self.model_name not in SeamlessM4TASR.loaded_models:
            translator = Translator(
                self.model_name,
                vocoder,
                device=torch.device(self.device),
                dtype=torch.float16,
            )
            SeamlessM4TASR.loaded_models[self.model_name] = {
                'translator': translator,
                'ref_count': 1
            }
        else:
            SeamlessM4TASR.loaded_models[self.model_name]['ref_count'] += 1

    @property
    def translator(self):
        return self.loaded_models[self.model_name]['translator']

    def recognize(self,
                  audio_file: Union[str, os.PathLike],
                  language: str = 'kor',
                  audio_format: str = 'mp3') -> str:
        audio = AudioSegment.from_file(audio_file, format=audio_format).set_frame_rate(16000).set_channels(1)
        segments = self.split_audio(audio)
        m4t_recognized = []
        for seg in segments:
            audio = torch.Tensor(self.audiosegment2rawdata(seg))
            text_output, _ = self.translator.predict(
                input=audio,
                task_str="asr",
                tgt_lang=language,
            )
            m4t_recognized.append(text_output)
        return ' '.join([str(s[0]) for s in m4t_recognized])


class Wav2VecASR(BaseASR):
    def __init__(self, model_name, device):
        super().__init__(model_name, device)
        self.register_model()

    def register_model(self, *args, **kwargs):
        if self.model_name not in Wav2VecASR.loaded_models:
            pipe = pipeline("automatic-speech-recognition", model=self.model_name)
            Wav2VecASR.loaded_models[self.model_name] = {
                'pipe': pipe,
                'ref_count': 1
            }
        else:
            Wav2VecASR.loaded_models[self.model_name]['ref_count'] += 1

    @property
    def pipe(self):
        return self.loaded_models[self.model_name]['pipe']

    def recognize(self,
                  audio_file: Union[str, os.PathLike],
                  language: str = 'ko',
                  audio_format: str = 'mp3') -> str:
        audio = self.load_full_audio(audio_file)
        return self.pipe(audio[0].cpu().numpy())


class YoutubeASR:
    default_params = {
        'whisper': {
            'model_name': 'openai/whisper-tiny',
        },
        'm4t': {
            'model_name': 'seamlessM4T_medium',
            'vocoder': 'vocoder_36langs'
        },
        'wav2vec': {
            'model_name': 'anantoj/wav2vec2-xls-r-1b-korean'
        }
    }

    implementations = {
        'whisper': WhisperASR,
        'm4t': SeamlessM4TASR,
        'wav2vec': Wav2VecASR
    }

    def __init__(self,
                 models_params: Dict[str, dict] = None):
        if not models_params:
            models_params = YoutubeASR.default_params

        self.models_params = models_params

    @staticmethod
    def download_video(youtube_link: str,
                       dest_path: Union[str, os.PathLike],
                       only_audio: bool = True) -> Path:
        dest_path = Path(dest_path).resolve()
        youtube = YouTube(youtube_link)
        video = youtube.streams.filter(only_audio=only_audio).first()
        video.download(dest_path.parent, filename=dest_path.name)
        return dest_path

    @staticmethod
    def get_generated_captions(youtube_link: str,
                               lang: str) -> List[Dict[str, Any]]:
        video_id = parse_qs(
            urlparse(youtube_link).query
        )['v'][0]
        tr = YouTubeTranscriptApi.get_transcript(video_id, (lang,))
        return tr

    @staticmethod
    def crop_captions(subtitles: List[Dict[str, Any]],
                      start: str,
                      end: str) -> str:
        starts = [s['start'] for s in subtitles]
        start = bisect_left(starts, YoutubeASR.timecode_to_ms(start) / 1000)
        end = bisect_left(starts, YoutubeASR.timecode_to_ms(end) / 1000)
        return ' '.join([s['text'] for s in subtitles[start: end]])

    @staticmethod
    def timecode_to_ms(timecode: str) -> int:
        minutes, seconds = map(int, timecode.split(":"))
        return (minutes * 60 + seconds) * 1000

    @staticmethod
    def trim_audio(audio_path: Union[str, os.PathLike],
                   trimmed_path: Union[str, os.PathLike],
                   timecode_start: str = None,
                   timecode_end: str = None):
        audio_path = Path(audio_path).resolve()
        trimmed_path = Path(trimmed_path).resolve()
        start_ms = YoutubeASR.timecode_to_ms(timecode_start) if timecode_start else 0
        end_ms = YoutubeASR.timecode_to_ms(timecode_end) if timecode_end else None
        audio = AudioSegment.from_file(audio_path, format="mp4")
        trimmed_audio = audio[start_ms:end_ms]
        trimmed_audio.export(trimmed_path, format="mp3")

    def recognize(self,
                  yb_video: Union[str, os.PathLike],
                  asr_model: str,
                  lang: str,
                  device: str = 'cuda:0',
                  start: str = None,
                  end: str = None) -> str:
        if not Path(yb_video).is_file():
            try:
                video_id = parse_qs(
                    urlparse(yb_video).query
                )['v'][0]
                video_path = f'{video_id}.mp3'

                print(f'Downloading audio to {video_path}')
                self.download_video(yb_video, video_path)
                yb_video = video_path
            except KeyError:
                raise ValueError('yb_video should be either path to file or youtube link')

        yb_video = Path(yb_video).resolve()

        if start or end:
            trimmed_video = f'{yb_video.stem}_trimmed{yb_video.suffix}'
            print(f'Trimming audio to {trimmed_video}')

            self.trim_audio(yb_video, trimmed_video, start, end)
            yb_video = trimmed_video

        model = self.implementations[asr_model](device=device, **self.models_params[asr_model])

        return model.recognize(yb_video, language=lang)