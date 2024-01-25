import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.pipelines.audio_utils import ffmpeg_microphone_live
import sys
from diffusers import DiffusionPipeline
import os
import threading

# Determine the device acceleration type
device = "cpu"
torch_dtype = torch.float32
if torch.cuda.is_available():
    device = "cuda:0"
    torch_dtype = torch.float16
elif torch.backends.mps.is_available():
    device = "mps"
    torch_dtype = torch.float16

print("Using device", device)

model_id = "distil-whisper/distil-medium.en"

theme = " 3d, low-poly game art, polygon mesh, jagged, blocky."

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=False, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

transcriber = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    torch_dtype=torch_dtype,
    device=device,
)

if not os.path.exists(".img"):
    os.mkdir(".img")

diffusionPipeline = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo").to("mps")

def remove_last_char(s):
    if len(s) > 0:
        return s[:-1]
    return s

def generate_image(prompt: str):
    print("Generating image for prompt:", prompt)

    results = diffusionPipeline(
        prompt=prompt,
        num_inference_steps=3,
        guidance_scale=0.0,
        num_images = 1
    )
    image = results.images[0]
    image.save(f".img/image.png")

def transcribe(chunk_length_s=5.0, stream_chunk_s=1.0):
    sampling_rate = transcriber.feature_extractor.sampling_rate

    # prime the pipeline with a white background
    generate_image("white background")

    # start the main loop
    print("Start speaking...")

    # select a theme
    # theme = " renaissance oil painting style."
    # theme = " anime style."
    # theme = " line art style."
    # theme = " bright watercolor style."
    # theme = " closeup portrait photo, dramatic."
    # theme = " minecraft block style."
    # theme = " 3d, low-poly game art, polygon mesh, jagged, blocky."

    while True:
        mic = ffmpeg_microphone_live(
            sampling_rate=sampling_rate,
            chunk_length_s=chunk_length_s,
            stream_chunk_s=stream_chunk_s,
        )

        for item in transcriber(mic, generate_kwargs={"max_new_tokens": 128}):
            sys.stdout.write("\033[K")
            print(item["text"], end="\r")
            if not item["partial"][0]:
                break

        # Generate the image in a thread
        image_thread = threading.Thread(target=generate_image, args=(remove_last_char(item["text"]) + "," + theme,))
        image_thread.start()

import gradio as gr

themes = [
    " renaissance oil painting style.",
    " anime style.",
    " line art style.",
    " bright watercolor style.",
    " closeup portrait photo, dramatic.",
    " minecraft block style.",
    " 3d, low-poly game art, polygon mesh, jagged, blocky.",
]

def theme_setter(new_theme):
    global theme
    theme = new_theme

demo = gr.Interface(
    theme_setter,
    [
        gr.Radio(themes, label="Theme", info="Prompt theme"),
    ],
    "text",
    examples=[
        ["minecraft block style"],
    ]
)

image_thread = threading.Thread(target=transcribe, args=())
image_thread.start()

demo.launch()


