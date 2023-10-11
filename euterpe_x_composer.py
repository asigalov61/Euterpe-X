# -*- coding: utf-8 -*-
"""Euterpe_X_Composer.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/asigalov61/Euterpe-X/blob/main/Euterpe_X_Composer.ipynb

# Euterpe X Composer (ver. 4.5)

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
!git clone --depth 1 https://github.com/asigalov61/Euterpe-X
!pip install huggingface_hub
!pip install torch
!pip install einops
!pip install torch-summary
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
from x_transformer import *

# %cd /content/
print('=' * 70)
print('Loading aux Euterpe X modules...')

import matplotlib.pyplot as plt

from torchsummary import summary
from sklearn import metrics

from midi2audio import FluidSynth
from IPython.display import Audio, display

from huggingface_hub import hf_hub_download

from google.colab import files

print('=' * 70)
print('Done!')
print('Enjoy! :)')
print('=' * 70)

"""# (LOAD MODEL)

## NOTE: Choose one and do not forget to restart the colab runtime if you are switching between the checkpoints
"""

#@title Load Euterpe X Small Model

#@markdown Fast model, 32 layers, 141k MIDIs training corpus

full_path_to_model_checkpoint = "/content/Euterpe-X/Models/Small/Euterpe_X_Small_Trained_Model_58000_steps_0.6865_loss_0.7964_acc.pth" #@param {type:"string"}

#@markdown Model precision option

model_precision = "bfloat16" # @param ["bfloat16", "float16", "float32"]

#@markdown bfloat16 == Third precision/triple speed (if supported, otherwise the model will default to float16)

#@markdown float16 == Half precision/double speed

#@markdown float32 == Full precision/normal speed

plot_tokens_embeddings = True # @param {type:"boolean"}

print('=' * 70)
print('Loading Euterpe X Small Pre-Trained Model...')
print('Please wait...')
print('=' * 70)

if os.path.isfile(full_path_to_model_checkpoint):
  print('Model already exists...')

else:
  hf_hub_download(repo_id='asigalov61/Euterpe-X',
                  filename='Euterpe_X_Small_Trained_Model_58000_steps_0.6865_loss_0.7964_acc.pth',
                  local_dir='/content/Euterpe-X/Models/Small/',
                  local_dir_use_symlinks=False)
print('=' * 70)
print('Instantiating model...')

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda'

if model_precision == 'bfloat16' and torch.cuda.is_bf16_supported():
  dtype = 'bfloat16'
else:
  dtype = 'float16'

if model_precision == 'float16':
  dtype = 'float16'

if model_precision == 'float32':
  dtype = 'float32'

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

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

print('Model will use', dtype, 'precision...')
print('=' * 70)

# Model stats
print('Model summary...')
summary(model)

# Plot Token Embeddings
if plot_tokens_embeddings:
  tok_emb = model.module.net.token_emb.emb.weight.detach().cpu().tolist()

  cos_sim = metrics.pairwise_distances(
    tok_emb, metric='cosine'
  )
  plt.figure(figsize=(7, 7))
  plt.imshow(cos_sim, cmap="inferno", interpolation="nearest")
  im_ratio = cos_sim.shape[0] / cos_sim.shape[1]
  plt.colorbar(fraction=0.046 * im_ratio, pad=0.04)
  plt.xlabel("Position")
  plt.ylabel("Position")
  plt.tight_layout()
  plt.plot()
  plt.savefig("/content/Euterpe-X-Small-Tokens-Embeddings-Plot.png", bbox_inches="tight")

#@title Load Euterpe X Large Model

#@markdown Slow model, 60 layers, 314k MIDIs training corpus

full_path_to_model_checkpoint = "/content/Euterpe-X/Models/Large/Euterpe_X_Large_Trained_Model_100000_steps_0.477_loss_0.8533_acc.pth" #@param {type:"string"}

#@markdown Model precision option

model_precision = "bfloat16" # @param ["bfloat16", "float16", "float32"]

#@markdown bfloat16 == Third precision/triple speed (if supported, otherwise the model will default to float16)

#@markdown float16 == Half precision/double speed

#@markdown float32 == Full precision/normal speed

plot_tokens_embeddings = True # @param {type:"boolean"}

print('=' * 70)
print('Loading Euterpe X Large Pre-Trained Model...')
print('Please wait...')
print('=' * 70)

if os.path.isfile(full_path_to_model_checkpoint):
  print('Model already exists...')

else:
  hf_hub_download(repo_id='asigalov61/Euterpe-X',
                  filename='Euterpe_X_Large_Trained_Model_100000_steps_0.477_loss_0.8533_acc.pth',
                  local_dir='/content/Euterpe-X/Models/Large',
                  local_dir_use_symlinks=False)

print('=' * 70)
print('Instantiating model...')

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda'

if model_precision == 'bfloat16' and torch.cuda.is_bf16_supported():
  dtype = 'bfloat16'
else:
  dtype = 'float16'

if model_precision == 'float16':
  dtype = 'float16'

if model_precision == 'float32':
  dtype = 'float32'

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

SEQ_LEN = 2048

# instantiate the model

model = TransformerWrapper(
    num_tokens = 3344,
    max_seq_len = SEQ_LEN,
    attn_layers = Decoder(dim = 1024, depth = 60, heads = 8)
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

print('Model will use', dtype, 'precision...')
print('=' * 70)

# Model stats
print('Model summary...')
summary(model)

# Plot Token Embeddings
if plot_tokens_embeddings:
  tok_emb = model.module.net.token_emb.emb.weight.detach().cpu().tolist()

  cos_sim = metrics.pairwise_distances(
    tok_emb, metric='cosine'
  )
  plt.figure(figsize=(7, 7))
  plt.imshow(cos_sim, cmap="inferno", interpolation="nearest")
  im_ratio = cos_sim.shape[0] / cos_sim.shape[1]
  plt.colorbar(fraction=0.046 * im_ratio, pad=0.04)
  plt.xlabel("Position")
  plt.ylabel("Position")
  plt.tight_layout()
  plt.plot()
  plt.savefig("/content/Euterpe-X-Large-Tokens-Embeddings-Plot.png", bbox_inches="tight")

"""# (LOAD SEED MIDI)"""

#@title Load Seed MIDI

#@markdown Press play button to to upload your own seed MIDI or to load one of the provided sample seed MIDIs from the dropdown list below

select_seed_MIDI = "Upload your own custom MIDI" # @param ["Upload your own custom MIDI", "Euterpe-X-Piano-Seed-1", "Euterpe-X-Piano-Seed-2", "Euterpe-X-Piano-Seed-3", "Euterpe-X-Piano-Seed-4", "Euterpe-X-Piano-Seed-5", "Euterpe-X-MI-Seed-1", "Euterpe-X-MI-Seed-2", "Euterpe-X-MI-Seed-3", "Euterpe-X-MI-Seed-4", "Euterpe-X-MI-Seed-5"]
number_of_prime_tokens = 300 # @param {type:"slider", min:126, max:2000, step:3}
render_MIDI_to_audio = False # @param {type:"boolean"}

print('=' * 70)
print('Euterpe X Seed MIDI Loader')
print('=' * 70)

f = ''

if select_seed_MIDI != "Upload your own custom MIDI":
  print('Loading seed MIDI...')
  f = '/content/Euterpe-X/Seeds/'+select_seed_MIDI+'.mid'
  score = TMIDIX.midi2single_track_ms_score(open(f, 'rb').read(), recalculate_channels=False)

else:
  print('Upload your own custom MIDI...')
  print('=' * 70)
  uploaded_MIDI = files.upload()
  if list(uploaded_MIDI.keys()):
    score = TMIDIX.midi2single_track_ms_score(list(uploaded_MIDI.values())[0], recalculate_channels=False)
    f = list(uploaded_MIDI.keys())[0]

if f != '':

  print('=' * 70)
  print('File:', f)
  print('=' * 70)

  #=======================================================
  # START PROCESSING

  # INSTRUMENTS CONVERSION CYCLE
  events_matrix = []
  melody_chords_f = []
  melody_chords_f1 = []

  itrack = 1

  patches = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

  patch_map = [
              [0, 1, 2, 3, 4, 5, 6, 7], # Piano
              [24, 25, 26, 27, 28, 29, 30], # Guitar
              [32, 33, 34, 35, 36, 37, 38, 39], # Bass
              [40, 41], # Violin
              [42, 43], # Cello
              [46], # Harp
              [56, 57, 58, 59, 60], # Trumpet
              [64, 65, 66, 67, 68, 69, 70, 71], # Sax
              [72, 73, 74, 75, 76, 77, 78], # Flute
              [-1], # Drums
              [52, 53], # Choir
              [16, 17, 18, 19, 20] # Organ
              ]

  while itrack < len(score):
    for event in score[itrack]:
        if event[0] == 'note' or event[0] == 'patch_change':
            events_matrix.append(event)
    itrack += 1

  events_matrix.sort(key=lambda x: x[1])

  events_matrix1 = []

  for event in events_matrix:
    if event[0] == 'patch_change':
        patches[event[2]] = event[3]

    if event[0] == 'note':
        event.extend([patches[event[3]]])
        once = False

        for p in patch_map:
            if event[6] in p and event[3] != 9: # Except the drums
                event[3] = patch_map.index(p)
                once = True

        if not once and event[3] != 9: # Except the drums
            event[3] = 15 # All other instruments/patches channel
            event[5] = max(80, event[5])

        if event[3] < 12: # We won't write chans 12-16 for now...
            events_matrix1.append(event)

  #=======================================================
  # PRE-PROCESSING

  # checking number of instruments in a composition
  instruments_list_without_drums = list(set([y[3] for y in events_matrix1 if y[3] != 9]))

  if len(events_matrix1) > 0 and len(instruments_list_without_drums) > 0:

    # recalculating timings
    for e in events_matrix1:
        e[1] = int(e[1] / 8) # Max 1 seconds for start-times
        e[2] = int(e[2] / 30) # Max 2 seconds for durations

    # Sorting by pitch, then by start-time
    events_matrix1.sort(key=lambda x: x[4], reverse=True)
    events_matrix1.sort(key=lambda x: x[1])

    #=======================================================
    # FINAL PRE-PROCESSING

    melody_chords = []

    pe = events_matrix1[0]

    for e in events_matrix1:
      if e[1] >= 0 and e[2] >= 0:

        # Cliping all values...
        tim = max(0, min(255, e[1]-pe[1]))
        dur = max(1, min(127, e[2]))
        cha = max(0, min(11, e[3]))
        ptc = max(1, min(127, e[4]))
        vel = max(1, min(127, e[5]))

        # Writing final note
        melody_chords.append([tim, dur, cha, ptc, vel])

        pe = e

  #=======================================================
  # Velocities map
  #=======================================================

  # Default fixed velocities for each channel/instrument
  velocities_map = [80, 80, 70, 100, 90, 80, 100, 100, 100, 90, 110, 100]

  # Extracting velocities from the MIDI file
  for i in range(12):
    vels = [m[4] for m in melody_chords if m[2] == i]

    avg_vel = 0

    if len(vels) > 0:
      avg_vel = int(sum(vels) / len(vels))

    if avg_vel > 20:
      velocities_map[i] = avg_vel

  #=======================================================
  # MAIN PROCESSING CYCLE
  #=======================================================

  for m in melody_chords:

    # WRITING EACH NOTE HERE
    time = m[0]
    cha_dur = (m[2] * 128) + m[1]
    cha_ptc = (m[2] * 128) + m[3]

    melody_chords_f.extend([time, cha_dur+256, cha_ptc+1792])
    melody_chords_f1.append([time, cha_dur+256, cha_ptc+1792])

  melody_chords_f1 = melody_chords_f1[:(number_of_prime_tokens // 3)]
  melody_chords_f = melody_chords_f[:number_of_prime_tokens]

  #=======================================================

  song = melody_chords_f

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

  detailed_stats = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(song_f,
                                                            output_signature = 'Euterpe X',
                                                            output_file_name = '/content/Euterpe-X-Seed-Composition',
                                                            track_name='Project Los Angeles',
                                                            list_of_MIDI_patches=[0, 24, 32, 40, 42, 46, 56, 65, 73, 0, 53, 19, 0, 0, 0, 0]
                                                            )

  #=======================================================

  print('=' * 70)
  print('Composition stats:')
  print('Composition has', len(melody_chords_f1), 'notes')
  print('Composition has', len(melody_chords_f), 'tokens')
  print('=' * 70)

  print('Displaying resulting composition...')
  print('=' * 70)

  fname = '/content/Euterpe-X-Seed-Composition'

  x = []
  y =[]
  c = []

  colors = ['red', 'yellow', 'green', 'cyan', 'blue', 'pink', 'orange', 'purple', 'gray', 'white', 'gold', 'silver']

  block_lines = [(song_f[-1][1] / 1000)]
  block_tokens = [min(len(melody_chords_f), number_of_prime_tokens)]

  for s in song_f:
    x.append(s[1] / 1000)
    y.append(s[4])
    c.append(colors[s[3]])

  if render_MIDI_to_audio:
    FluidSynth("/usr/share/sounds/sf2/FluidR3_GM.sf2", 16000).midi_to_audio(str(fname + '.mid'), str(fname + '.wav'))
    display(Audio(str(fname + '.wav'), rate=16000))

  plt.figure(figsize=(14,5))
  ax=plt.axes(title=fname)
  ax.set_facecolor('black')

  plt.scatter(x,y, c=c)
  plt.xlabel("Time")
  plt.ylabel("Pitch")
  plt.show()

else:
  print('=' * 70)

"""# (COMPOSITION LOOP)

## Run the cells below in a loop to generate endless continuation
"""

#@title Standard Continuation Generator

#@markdown Generation settings

try_to_generate_outro = False #@param {type:"boolean"}
number_of_tokens_to_generate = 120 # @param {type:"slider", min:33, max:1023, step:3}
number_of_batches_to_generate = 4 #@param {type:"slider", min:1, max:16, step:1}
preview_length_in_tokens = 120 # @param {type:"slider", min:33, max:240, step:3}
number_of_memory_tokens = 2046 #@param {type:"slider", min:402, max:2046, step:3}
temperature = 0.9 #@param {type:"slider", min:0.1, max:1, step:0.1}

#@markdown Other settings

render_MIDI_to_audio = True # @param {type:"boolean"}

print('=' * 70)
print('Euterpe X Standard Continuation Model Generator')
print('=' * 70)

preview = melody_chords_f[-preview_length_in_tokens:]

mel_cho = melody_chords_f[-number_of_memory_tokens:]

if try_to_generate_outro:
  mel_cho.extend([3328, 3328, 3328])

inp = [mel_cho] * number_of_batches_to_generate

inp = torch.LongTensor(inp).cuda()

with ctx:
  out = model.module.generate(inp,
                              number_of_tokens_to_generate,
                              temperature=temperature,
                              return_prime=False,
                              eos_token=None,
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

      song = preview + out1
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

      detailed_stats = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(song_f,
                                                          output_signature = 'Euterpe X',
                                                          output_file_name = '/content/Euterpe-X-Music-Composition_'+str(i),
                                                          track_name='Project Los Angeles',
                                                          list_of_MIDI_patches=[0, 24, 32, 40, 42, 46, 56, 65, 73, 0, 53, 19, 0, 0, 0, 0]
                                                          )
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

      if render_MIDI_to_audio:
        FluidSynth("/usr/share/sounds/sf2/FluidR3_GM.sf2", 16000).midi_to_audio(str(fname + '.mid'), str(fname + '.wav'))
        display(Audio(str(fname + '.wav'), rate=16000))

      plt.figure(figsize=(14,5))
      ax=plt.axes(title=fname)
      ax.set_facecolor('black')

      plt.scatter(x,y, c=c)

      pbl = song_f[(int(preview_length_in_tokens / 3))][1] / 1000

      ax.axvline(x=pbl, c='w')

      plt.xlabel("Time")
      plt.ylabel("Pitch")
      plt.show()

#@title Choose one generated block to add to the composition
block_action = "add_last_generated_block" #@param ["add_last_generated_block", "remove_last_added_block"]
add_block_with_batch_number = 0 #@param {type:"slider", min:0, max:15, step:1}
render_MIDI_to_audio = False # @param {type:"boolean"}

print('=' * 70)

if block_action == 'add_last_generated_block':
  melody_chords_f.extend(out0[min(len(out0)-1, add_block_with_batch_number)])
  print('Block added!')
else:
  if len(block_tokens) > 1:
    melody_chords_f = melody_chords_f[:(len(melody_chords_f)-block_tokens[-1])]
    print('Block removed!')
  else:
    print('Nothing to remove!!!')

print('=' * 70)
print('Composition now has', (len(melody_chords_f) // 3), 'notes')
print('Composition now has', len(melody_chords_f), 'tokens')


print('=' * 70)
print('Sample INTs', out1[:12])
print('=' * 70)

if len(melody_chords_f) != 0:

    song = melody_chords_f
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

    detailed_stats = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(song_f,
                                                              output_signature = 'Euterpe X',
                                                              output_file_name = '/content/Euterpe-X-Music-Composition',
                                                              track_name='Project Los Angeles',
                                                              list_of_MIDI_patches=[0, 24, 32, 40, 42, 46, 56, 65, 73, 0, 53, 19, 0, 0, 0, 0]
                                                              )
    print('=' * 70)
    print('Displaying resulting composition...')
    print('=' * 70)

    fname = '/content/Euterpe-X-Music-Composition'

    x = []
    y =[]
    c = []

    colors = ['red', 'yellow', 'green', 'cyan', 'blue', 'pink', 'orange', 'purple', 'gray', 'white', 'gold', 'silver']

    if block_action == 'add_last_generated_block':
      block_lines.append((song_f[-1][1] / 1000))
      block_tokens.append(len(out0[min(len(out0)-1, add_block_with_batch_number)]))
    else:
      if len(block_tokens) > 1:
        block_lines.pop()
        block_tokens.pop()

    for s in song_f:
      x.append(s[1] / 1000)
      y.append(s[4])
      c.append(colors[s[3]])

    if render_MIDI_to_audio:
      FluidSynth("/usr/share/sounds/sf2/FluidR3_GM.sf2", 16000).midi_to_audio(str(fname + '.mid'), str(fname + '.wav'))
      display(Audio(str(fname + '.wav'), rate=16000))

    plt.figure(figsize=(14,5))
    ax=plt.axes(title=fname)
    ax.set_facecolor('black')

    plt.scatter(x,y, c=c)

    for bl in block_lines:
      ax.axvline(x=bl, c='w')

    plt.xlabel("Time")
    plt.ylabel("Pitch")
    plt.show()

"""# Congrats! You did it! :)"""