from big_sleep import big_sleep
from tqdm import tqdm, trange
import torchvision.io
import torchvision.utils
 
import torch
import torch.linalg
import threading
import os
import json
import fire
import sys
import time
import signal
import queue
import random

# graceful keyboard interrupt

terminate = False

def signal_handling(signum,frame):
    global terminate
    print("Detected keyboard interrupt; exiting...")
    terminate = True

signal.signal(signal.SIGINT,signal_handling)

FRAME_QUEUE = queue.Queue(maxsize=100)

def permute_guidance(guidance):
    prompt = set(s for s in guidance.get('prompt', '').split('\\') if s)
    avoid = set(s for s in guidance.get('avoid', '').split('\\') if s)

    keywords = set.union(prompt, avoid)
    if not keywords:
        return False

    new_prompt = prompt
    new_avoid = avoid
    while prompt == new_prompt and avoid == new_avoid:
        new_prompt = set(s for s in keywords if random.randrange(2))
        new_avoid = set(s for s in keywords if s not in new_prompt)
        print(f'Generated {new_prompt} - {new_avoid}')

    guidance['prompt'] = '\\'.join(new_prompt)
    guidance['avoid'] = '\\'.join(new_avoid)
    return True


class Slippery(big_sleep.Imagine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_target = None
        self.image_weight = 0

    def loss(self):
        text_loss = super().loss()

        if self.image_target is not None:
            image = self.model.model()[0]
            #print("IMAGE: ", image)
            #print("TARGET: ", self.image_target)
            image_loss = self.image_weight * torch.linalg.norm(image - self.image_target)
        else:
            image_loss = 0
        return text_loss + image_loss

    def forward(self):
        global terminate
        print(f'Generating')

        self.model(self.encoded_texts["max"][0])  # warmup

        if self.open_folder:
            big_sleep.open_folder('./')
            self.open_folder = False

        step = 0
        loss_record = float('inf')
        frames_since_record = 0
        last_guidance = {}
        guidance = {}
        swap_prompt_avoid = False
        while True:
            if terminate:
                print("Saving to model.pickle")
                torch.save(self, "model.pickle")
                return
            try:
                fh = open('PROMPT.json', 'r')
                guidance = json.load(fh)
                fh.close()
            except:
                print(sys.exc_info()[0])
                
            if guidance is None:
                guidance = {}

            prompt = guidance.get('prompt', '')
            avoid = guidance.get('avoid', '')
            learning_rate = guidance.get('learning_rate', 0.07)
            image_prompt = guidance.get('image_prompt')
            patience = guidance.get('patience', 60)

            if last_guidance != guidance:
                print(f'prompt: {prompt} - {avoid} | rate: {learning_rate}')
                self.encode_max_and_min(prompt, avoid)
                for g in self.optimizer.param_groups:
                    g['lr'] = learning_rate

                if image_prompt and len(image_prompt) == 2:
                    self.image_target = torchvision.io.read_image(image_prompt[0]).cuda() / 256.0
                    self.image_weight = image_prompt[1]
                else:
                    self.image_target = None
                    self.image_weight = 0

                loss_record = float('inf')
                frames_since_record = 0
                last_guidance = guidance

            def on_image(loss, image):
                if loss < loss_record:
                    torchvision.utils.save_image(image.cpu(), f'./out.{step}.png')
                    FRAME_QUEUE.put(step)

            loss = self.train_step(0, step, on_image)
            if loss < loss_record:
                print(f"frame {step}; loss: {loss} (record)")
                loss_record = loss
                frames_since_record = 0
            else:
                print(f"frame {step}; loss: {loss} ({frames_since_record})")
                frames_since_record += 1

            if frames_since_record > patience:
                print("Patience exceeded -- permuting prompts")
                if permute_guidance(guidance):
                    with open('PROMPT.json', 'w') as fh:
                        json.dump(guidance, fh, sort_keys=True, indent=2)
                    last_guidance = {}  # force reload
                else:
                    print("... but nothing to permute")

            step += 1

def image_process_thread():
    frame_index = 0
    while not terminate:
        try:
            step = FRAME_QUEUE.get(timeout=2)
        except queue.Empty:
            continue

        print(f"process {step}")

        os.system(f"""
            (
                pngquant -o frame.{frame_index}.png --force out.{step}.png
                ln -sf frame.{frame_index}.png progress.png
                rm -f out.{step}.png
            ) &
        """)

        if frame_index != 0 and frame_index % 60 == 0:
            time.sleep(5)  # hopefully avoid race conditions with the above... but hacky ofc
            os.system(f"""
                ffmpeg -r 30 -f image2 -s 512x512 -i frame.%d.png -vcodec libx264 -pix_fmt yuv420p -bf 0 -r 30 _next.mp4
                if [ -e progress.mp4 ]; then
                    echo 'file progress.mp4' > list.txt
                    echo 'file _next.mp4' >> list.txt
                    ffmpeg -f concat -i list.txt -auto_convert 1 -c copy _progress.next.mp4
                    mv _progress.next.mp4 progress.mp4
                    rm -f list.txt _next.mp4
                else
                    mv _next.mp4 progress.mp4
                fi

                cp -f frame.{frame_index}.png tmp.png
                ln -sf tmp.png progress.png
                rm -f frame.*.png
            """)
            frame_index = 0
        else:
            frame_index += 1

def load():
    if os.path.exists('model.pickle'):
        return torch.load('model.pickle')
    else:
        return Slippery('out', epochs=1, iterations=10000, save_every=1, save_progress=True)

def train():
    load()()

image_process = threading.Thread(target=image_process_thread)
image_process.start()

fire.Fire(train)

image_process.join()
