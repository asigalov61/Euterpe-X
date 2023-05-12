# -*- coding: utf-8 -*-
"""Euterpe_X_TTM_Edition.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/asigalov61/Euterpe-X/blob/main/Euterpe_X_TTM_Edition.ipynb

# Euterpe X TTM Edition (ver. 2.1)

***

Powered by tegridy-tools: https://github.com/asigalov61/tegridy-tools

***

WARNING: This complete implementation is a functioning model of the Artificial Intelligence. Please excercise great humility, care, and respect. https://www.nscai.gov/

***

#### Project Los Angeles

#### Tegridy Code 2023

***

# (GPU CHECK)
"""

#@title NVIDIA GPU check
!nvidia-smi

"""# (SETUP ENVIRONMENT)"""

#@title Install dependencies
!git clone https://github.com/asigalov61/Euterpe-X
!pip install torch
!pip install einops
!pip install fuzzywuzzy[speedup]
!pip install torch-summary
!pip install sklearn
!pip install tqdm
!pip install matplotlib
!apt install fluidsynth #Pip does not work for some reason. Only apt works
!pip install midi2audio

# Commented out IPython magic to ensure Python compatibility.
#@title Import modules

print('=' * 70)
print('Loading core Euterpe X modules...')

import os
import pickle
import random
import secrets
import statistics
from time import time
import tqdm

print('=' * 70)
print('Loading main Euterpe X modules...')
import torch

# %cd /content/Euterpe-X

import TMIDIX

from x_transformer import TransformerWrapper, Decoder, AutoregressiveWrapper

# %cd /content/

from fuzzywuzzy import process

print('=' * 70)
print('Loading aux Euterpe X modules...')

import matplotlib.pyplot as plt

from torchsummary import summary
from sklearn import metrics

from midi2audio import FluidSynth
from IPython.display import Audio, display

print('=' * 70)
print('Done!')
print('Enjoy! :)')
print('=' * 70)

"""# (LOAD MODEL)"""

# Commented out IPython magic to ensure Python compatibility.
#@title Unzip Pre-Trained Euterpe X Model
print('=' * 70)
# %cd /content/Euterpe-X/Model

print('=' * 70)
print('Unzipping pre-trained Euterpe X model...Please wait...')

!cat /content/Euterpe-X/Model/Euterpe_X_Trained_Model.zip* > /content/Euterpe-X/Model/Euterpe_X_Trained_Model.zip
print('=' * 70)

!unzip -j /content/Euterpe-X/Model/Euterpe_X_Trained_Model.zip
print('=' * 70)

print('Done! Enjoy! :)')
print('=' * 70)
# %cd /content/
print('=' * 70)

#@title Load Euterpe X Model
full_path_to_model_checkpoint = "/content/Euterpe-X/Model/Euterpe_X_Trained_Model_58000_steps_0.6865_loss.pth" #@param {type:"string"}

print('=' * 70)
print('Loading Euterpe X Pre-Trained Model...')
print('Please wait...')
print('=' * 70)
print('Instantiating model...')

SEQ_LEN = 2048

# instantiate the model

model = TransformerWrapper(
    num_tokens = 3344,
    max_seq_len = SEQ_LEN,
    attn_layers = Decoder(dim = 1024, depth = 32, heads = 8)
)

model = AutoregressiveWrapper(model)

model = torch.nn.DataParallel(model)

model.cuda()
print('=' * 70)

print('Loading model checkpoint...')

model.load_state_dict(torch.load(full_path_to_model_checkpoint))
print('=' * 70)

model.eval()

print('Done!')
print('=' * 70)

# Model stats
print('Model summary...')
summary(model)

# Plot Token Embeddings
tok_emb = model.module.net.token_emb.emb.weight.detach().cpu().tolist()

tok_emb1 = []

for t in tok_emb:
    tok_emb1.append([abs(statistics.median(t))])

cos_sim = metrics.pairwise_distances(
   tok_emb1, metric='euclidean'
)
plt.figure(figsize=(7, 7))
plt.imshow(cos_sim, cmap="inferno", interpolation="nearest")
im_ratio = cos_sim.shape[0] / cos_sim.shape[1]
plt.colorbar(fraction=0.046 * im_ratio, pad=0.04)
plt.xlabel("Position")
plt.ylabel("Position")
plt.tight_layout()
plt.plot()
plt.savefig("/content/Euterpe-X-Tokens-Embeddings-Plot.png", bbox_inches="tight")

"""# (LOAD AUX DATA)"""

# Commented out IPython magic to ensure Python compatibility.
#@title Unzip Euterpe X Aux Data
print('=' * 70)
# %cd /content/Euterpe-X/Aux-Data

print('=' * 70)
print('Unzipping Euterpe X Aux Data...Please wait...')

!cat /content/Euterpe-X/Aux-Data/Euterpe_X_Aux_Data.zip* > /content/Euterpe-X/Aux-Data/Euterpe_X_Aux_Data.zip
print('=' * 70)

!unzip -j /content/Euterpe-X/Aux-Data/Euterpe_X_Aux_Data.zip
print('=' * 70)

print('Done! Enjoy! :)')
print('=' * 70)
# %cd /content/
print('=' * 70)

#@title Load Euterpe X Aux Data

AUX_DATA = TMIDIX.Tegridy_Any_Pickle_File_Reader('/content/Euterpe-X/Aux-Data/Euterpe_X_Aux_Data')

print('Done!')

"""# (GENERATE)"""

#@title Standard/Simple Continuation

#@markdown Text-To-Music Settings

#@markdown NOTE: You can enter any desired title or artist, or both

enter_desired_song_title = "Family Guy" #@param {type:"string"}
enter_desired_artist = "TV Themes" #@param {type:"string"}

#@markdown Generation Settings

number_of_tokens_to_generate = 525 #@param {type:"slider", min:30, max:2046, step:33}
number_of_batches_to_generate = 4 #@param {type:"slider", min:1, max:16, step:1}
temperature = 0.9 #@param {type:"slider", min:0.1, max:1, step:0.1}
allow_model_to_stop_generation_if_needed = False #@param {type:"boolean"}

print('=' * 70)
print('Euterpe X TTM Model Generator')
print('=' * 70)

print('Searching titles...Please wait...')
random.shuffle(AUX_DATA)

titles_index = []

for A in AUX_DATA:
  titles_index.append(A[0])

search_string = ''

if enter_desired_song_title != '' and enter_desired_artist != '':
  search_string = enter_desired_song_title + ' --- ' + enter_desired_artist

else:
  search_string = enter_desired_song_title + enter_desired_artist

search_match = process.extract(query=search_string, choices=titles_index, limit=1)
search_index = titles_index.index(search_match[0][0])

print('Done!')
print('=' * 70)
print('Selected title:', AUX_DATA[search_index][0])
print('=' * 70)

if allow_model_to_stop_generation_if_needed:
  min_stop_token = 3343
else:
  min_stop_token = None

# Velocities
velocities_map = [80, 80, 70, 100, 90, 80, 100, 100, 100, 90, 110, 100]
vel_map = AUX_DATA[search_index][1]

for i in range(12):
  if vel_map[i] != 0:
    velocities_map[i] = vel_map[i]

# Loading data...
outy = AUX_DATA[search_index][2][3:]

block_marker = sum([(y * 8) for y in outy if y < 256]) / 1000

inp = [outy] * number_of_batches_to_generate

inp = torch.LongTensor(inp).cuda()

out = model.module.generate(inp, 
                      number_of_tokens_to_generate, 
                      temperature=temperature, 
                      return_prime=True, 
                      eos_token=min_stop_token, 
                      verbose=True)

out0 = out.tolist()
print('=' * 70)
print('Done!')
print('=' * 70)
#======================================================================
print('Rendering results...')

for i in range(number_of_batches_to_generate):

  print('=' * 70)
  print('Batch #', i)
  print('=' * 70)

  out1 = out0[i]

  print('Sample INTs', out1[:12])
  print('=' * 70)

  if len(out) != 0:
      
      song = out1
      song_f = []

      time = 0
      dur = 0
      channel = 0
      pitch = 0
      vel = 90

      for ss in song:

        if ss > 0 and ss < 256:

            time += ss * 8
          
        if ss >= 256 and ss < 256+(12*128):

            dur = ((ss-256) % 128) * 30
            
        if ss >= 256+(12*128) and ss < 256+(12*128)+(12*128):
            channel = (ss-(256+(12*128))) // 128
            pitch = (ss-(256+(12*128))) % 128
            vel = velocities_map[channel]

            song_f.append(['note', time, dur, channel, pitch, vel ])

      detailed_stats = TMIDIX.Tegridy_SONG_to_MIDI_Converter(song_f,
                                                          output_signature = 'Euterpe X',  
                                                          output_file_name = '/content/Euterpe-X-Music-Composition_'+str(i), 
                                                          track_name='Project Los Angeles',
                                                          list_of_MIDI_patches=[0, 24, 32, 40, 42, 46, 56, 71, 73, 0, 53, 19, 0, 0, 0, 0],
                                                          number_of_ticks_per_quarter=500)
      print('=' * 70)
      print('Displaying resulting composition...')
      print('=' * 70)

      fname = '/content/Euterpe-X-Music-Composition_'+str(i)

      x = []
      y =[]
      c = []

      colors = ['red', 'yellow', 'green', 'cyan', 'blue', 'pink', 'orange', 'purple', 'gray', 'white', 'gold', 'silver']

      for s in song_f:
        x.append(s[1] / 1000)
        y.append(s[4])
        c.append(colors[s[3]])

      FluidSynth("/usr/share/sounds/sf2/FluidR3_GM.sf2", 16000).midi_to_audio(str(fname + '.mid'), str(fname + '.wav'))
      display(Audio(str(fname + '.wav'), rate=16000))

      plt.figure(figsize=(14,5))
      ax=plt.axes(title=fname)
      ax.set_facecolor('black')

      plt.scatter(x,y, c=c)

      ax.axvline(x=block_marker, c='w')

      plt.xlabel("Time")
      plt.ylabel("Pitch")
      plt.show()

"""# Congrats! You did it! :)"""