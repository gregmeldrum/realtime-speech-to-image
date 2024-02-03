#!/usr/bin/env python3

import html
import os
import queue
import sys
import threading

import gradio as gr
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.pipelines.audio_utils import ffmpeg_microphone_live
from diffusers import DiffusionPipeline


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

themes = (
    "",
    "renaissance oil painting style.",
    "anime style.",
    "line art style.",
    "bright watercolor style.",
    "closeup portrait photo, dramatic.",
    "soviet propaganda poster style.",
    "minecraft block style.",
    "3d, low-poly game art, polygon mesh, jagged, blocky.",
    "retro comic book style.",
)
prompt_queue = queue.Queue()
generated_queue = queue.Queue()

# That's what you get when you don't have access to Go channels.
_lock = threading.Lock()
_theme = themes[-1]
_prompt = ""

def get_theme():
    with _lock:
        return _theme

def set_theme(theme):
    global _theme
    with _lock:
        _theme = theme

def get_prompt():
    with _lock:
        return _prompt

def set_prompt(prompt):
    global _prompt
    with _lock:
        _prompt = prompt


def generate_image(i, diffusionPipeline, prompt: str):
    name = f".img/image{i}"
    sys.stdout.write(f"\nGenerating {name}.png for prompt: {repr(prompt)}\n")
    results = diffusionPipeline(
        prompt=prompt,
        num_inference_steps=3,
        guidance_scale=0.0,
        num_images = 1
    )
    img = results.images[0]
    img.save(name + ".png")
    with open(name + ".txt", "w") as f:
        f.write(prompt)
    generated_queue.put((prompt, name + ".png"))
    return img


def thread_generate_image(diffusionPipeline):
    i = 0
    while True:
        prompt = prompt_queue.get()
        theme = get_theme()
        if theme:
          prompt += ", " + theme
        img = generate_image(i, diffusionPipeline, prompt)
        i += 1


def thread_transcribe(transcriber, chunk_length_s=5.0, stream_chunk_s=1.0):
    sys.stdout.write("Start talking....\n")
    while True:
        mic = ffmpeg_microphone_live(
            sampling_rate=transcriber.feature_extractor.sampling_rate,
            chunk_length_s=chunk_length_s,
            stream_chunk_s=stream_chunk_s,
        )
        text = ""
        for item in transcriber(mic, generate_kwargs={"max_new_tokens": 128}):
            sys.stdout.write("\r\033[K" + item["text"])
            t = item["text"].strip()
            set_prompt(t)
            if not item["partial"][0] and t.lower() not in ("you", "you're"):
                text = t
                break
        if text:
            text = text[:-1]
            prompt_queue.put(text)


_html = ""

def regen_ui():
    # Should use html as input but it doesn't work, not sure why.
    global _html
    try:
        txt, img = generated_queue.get_nowait()
        if _html:
            _html += "<br>"
        _html += html.escape(txt) + "<img src=\"/file=" + html.escape(img) + "\" />"
    except queue.Empty:
        pass
    return get_prompt(), _html


def main():
    os.chdir(BASE_DIR)
    if not os.path.exists(".img"):
        os.mkdir(".img")

    # Determine the device acceleration type.
    device = "cpu"
    torch_dtype = torch.float32
    if torch.cuda.is_available():
        device = "cuda:0"
        torch_dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        torch_dtype = torch.float16
    print(f"- Using device {device}")

    print("- Loading whisper")
    model_id = "distil-whisper/distil-medium.en"
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

    print("- Loading sdxl-turbo")
    diffusionPipeline = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo").to("mps")
    diffusionPipeline.set_progress_bar_config(disable=True)

    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    r = gr.Radio(choices=themes, value=themes[-1], label="Theme", info="Prompt theme")
                    r.select(fn=set_theme, inputs=r, outputs=None)
                with gr.Row():
                    with gr.Column():
                        txt = gr.Textbox( label="Prompt", interactive=False)
            with gr.Column():
                html = gr.HTML(value="")
        ui.load(fn=regen_ui, inputs=None, outputs=[txt, html], every=0.5)
    print("- Finished loading!")

    # Start the threads.
    threading.Thread(target=thread_generate_image, args=(diffusionPipeline,), daemon=True).start()
    threading.Thread(target=thread_transcribe, args=(transcriber,), daemon=True).start()
    ui.launch(quiet=True, share=False, allowed_paths=[os.path.join(BASE_DIR, '.img')])


if __name__ == "__main__":
    sys.exit(main())
