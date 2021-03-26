from big_sleep import big_sleep
from tqdm import tqdm, trange
import os
import json
import fire
import sys
import time

class Slippery(big_sleep.Imagine):
    def forward(self):
        print(f'Generating')

        self.model(self.encoded_texts["max"][0])  # warmup

        if self.open_folder:
            big_sleep.open_folder('./')
            self.open_folder = False

        step = 0
        image_pbar = tqdm(desc='image update', position=2, leave=True)
        last_guidance = {}
        guidance = {}
        while True:
            try:
                fh = open('PROMPT', 'r')
                guidance = json.load(fh)
                fh.close()
            except:
                print(sys.exc_info()[0])
                
            prompt = guidance.get('prompt', '')
            avoid = guidance.get('avoid', '')
            learning_rate = guidance.get('learning_rate', 0.07)

            if last_guidance != guidance:
                print(f'prompt: {prompt} - {avoid} | rate: {learning_rate}')
                self.encode_max_and_min(prompt, avoid)
                last_guidance = guidance
                for g in self.optimizer.param_groups:
                    g['lr'] = learning_rate
            loss = self.train_step(0, step, image_pbar)
            image_pbar.set_description(f'loss: {loss}')
            os.system(f'( pngquant -o out.{step}.png --force out.{step}.png; ln -sf out.{step}.png progress.png ) &')

            step += 1


def train():
    imagine = Slippery('out', epochs=1, iterations=10000, save_every=1, save_progress=True)

    imagine()

fire.Fire(train)
