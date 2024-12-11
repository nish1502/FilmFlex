import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
import pyaudio
import wave
from faster_whisper import WhisperModel
import time
import random
from collections import deque

NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    from main import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_custom_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_custom_nodes()


from nodes import (
    SaveImage,
    VAEDecode,
    CheckpointLoaderSimple,
    EmptyLatentImage,
    NODE_CLASS_MAPPINGS,
    CLIPTextEncode,
)


def transcribe_chunk(model, file_path):
    segments, info = model.transcribe(file_path, beam_size=7)
    transcription = ' '.join(segment.text for segment in segments)
    # Remove "thank you very much" from the transcription
    transcription = transcription.replace("thank you very much", "")
    return transcription


def record_chunk(p, stream, file_path, chunk_length=2):
    frames = []
    for _ in range(0, int(16000 / 1024 * chunk_length)):
        data = stream.read(1024)
        frames.append(data)

    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()


# Initialize a deque with a maximum length of 60 characters
transcription_buffer = deque(maxlen=100)


def main_combined():
    # Initialize image generation components
    import_custom_nodes()

    with torch.inference_mode():
        emptylatentimage = EmptyLatentImage()
        emptylatentimage_5 = emptylatentimage.generate(width=512, height=512, batch_size=1)

        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_20 = checkpointloadersimple.load_checkpoint(ckpt_name="re.safetensors")

        cliptextencode = CLIPTextEncode()
        ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
        ksamplerselect_14 = ksamplerselect.get_sampler(sampler_name="dpmpp_2m")

        sdturboscheduler = NODE_CLASS_MAPPINGS["SDTurboScheduler"]()
        samplercustom = NODE_CLASS_MAPPINGS["SamplerCustom"]()
        vaedecode = VAEDecode()
        saveimage = SaveImage()

        # Initialize audio transcription components
        model_size = "medium.en"
        model = WhisperModel(model_size, device="cuda", compute_type="float16")

        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)

        try:
            while True:
                # Audio transcription
                chunk_file = "temp_chunk.wav"
                record_chunk(p, stream, chunk_file)
                transcription = transcribe_chunk(model, chunk_file)
                print(transcription)
                os.remove(chunk_file)

                # Update the transcription buffer
                for char in transcription:
                    transcription_buffer.append(char)
                    # Convert the buffer back to a string
                transcription_window = ''.join(transcription_buffer)
                print(transcription_window)

                # Image generation based on transcription
                if transcription_window.strip():  # Check if transcription is not empty
                    cliptextencode_6 = cliptextencode.encode(
                        text="illustration of" + transcription_window,
                        clip=get_value_at_index(checkpointloadersimple_20, 1)
                    )

                    for q in range(1):
                        sdturboscheduler_22 = sdturboscheduler.get_sigmas(
                            steps=20,
                            denoise=1,
                            model=get_value_at_index(checkpointloadersimple_20, 0),
