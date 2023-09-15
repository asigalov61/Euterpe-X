import argparse
import glob
import os.path

import torch
import torch.nn.functional as F

import gradio as gr

from x_transformer import *
import tqdm

from midi_synthesizer import synthesis
import TMIDIX

import matplotlib.pyplot as plt

in_space = os.getenv("SYSTEM") == "spaces"
      
#=================================================================================================

@torch.no_grad()
def GenerateMIDI(num_tok, idrums, iinstr, progress=gr.Progress()):

    print('=' * 70)
    print('Req num tok', num_tok)
    print('Req instr', iinstr)
    print('Drums', idrums)
    print('=' * 70)

    if idrums:
        drums = 3330
    else:
        drums = 3329

    instruments_list = ["Piano", "Guitar", "Bass", "Violin", "Cello", "Harp", "Trumpet", "Sax", "Flute", 'Drums', "Choir", "Organ"]
    first_note_instrument_number = instruments_list.index(iinstr)

    start_tokens = [3343, drums, 3331+first_note_instrument_number]

    print('Selected Improv sequence:')
    print(start_tokens)
    print('=' * 70)
    
    outy = start_tokens

    for i in progress.tqdm(range(num_tok)):
    
        inp = torch.LongTensor([outy]).cpu()
        
        out = model.module.generate(inp,
                              1,
                              temperature=0.9,
                              return_prime=False,
                              verbose=False)
        
        out0 = out[0].tolist()

        outy.extend(out0)

    melody_chords_f = outy
    
    print('Sample INTs', melody_chords_f[:12])
    print('=' * 70)
    
    if len(melody_chords_f) != 0:
    
        song = melody_chords_f
        song_f = []
        time = 0
        dur = 0
        vel = 90
        pitch = 0
        channel = 0
    
        for ss in song:
        
            if ss > 0 and ss < 256:
            
                time += ss * 8
            
            if ss >= 256 and ss < 256+(12*128):
            
                dur = ((ss-256) % 128) * 30
            
            if ss >= 256+(12*128) and ss < 256+(12*128)+(12*128):
                channel = (ss-(256+(12*128))) // 128
                pitch = (ss-(256+(12*128))) % 128
                vel = pitch
            
                song_f.append(['note', time, dur, channel, pitch, vel ])
        
    output_signature = 'Euterpe X'
    output_file_name = 'Euterpe-X-Music-Composition'
    track_name='Project Los Angeles'
    list_of_MIDI_patches=[0, 24, 32, 40, 42, 46, 56, 71, 73, 0, 53, 19, 0, 0, 0, 0]
    number_of_ticks_per_quarter=500
    text_encoding='ISO-8859-1'
    
    output_header = [number_of_ticks_per_quarter, 
            [['track_name', 0, bytes(output_signature, text_encoding)]]]                                                    

    patch_list = [['patch_change', 0, 0, list_of_MIDI_patches[0]], 
                    ['patch_change', 0, 1, list_of_MIDI_patches[1]],
                    ['patch_change', 0, 2, list_of_MIDI_patches[2]],
                    ['patch_change', 0, 3, list_of_MIDI_patches[3]],
                    ['patch_change', 0, 4, list_of_MIDI_patches[4]],
                    ['patch_change', 0, 5, list_of_MIDI_patches[5]],
                    ['patch_change', 0, 6, list_of_MIDI_patches[6]],
                    ['patch_change', 0, 7, list_of_MIDI_patches[7]],
                    ['patch_change', 0, 8, list_of_MIDI_patches[8]],
                    ['patch_change', 0, 9, list_of_MIDI_patches[9]],
                    ['patch_change', 0, 10, list_of_MIDI_patches[10]],
                    ['patch_change', 0, 11, list_of_MIDI_patches[11]],
                    ['patch_change', 0, 12, list_of_MIDI_patches[12]],
                    ['patch_change', 0, 13, list_of_MIDI_patches[13]],
                    ['patch_change', 0, 14, list_of_MIDI_patches[14]],
                    ['patch_change', 0, 15, list_of_MIDI_patches[15]],
                    ['track_name', 0, bytes(track_name, text_encoding)]]

    output = output_header + [patch_list + song_f]

    midi_data = TMIDIX.score2midi(output, text_encoding)
    
    with open(f"Euterpe-X-Music-Composition.mid", 'wb') as f:
        f.write(midi_data)

    output1 = []
    itrack = 1
    
    opus =  TMIDIX.score2opus(output)
          
    while itrack < len(opus):
        for event in opus[itrack]:
            if (event[0] == 'note_on') or (event[0] == 'note_off'): 
                output1.append(event)
        itrack += 1
        
    audio = synthesis([500, output1], 'SGM-v2.01-YamahaGrand-Guit-Bass-v2.7.sf2')
    
    x = []
    y =[]
    c = []
    
    colors = ['red', 'yellow', 'green', 'cyan', 'blue', 'pink', 'orange', 'purple', 'gray', 'white', 'gold', 'silver']
    
    for s in song_f:
      x.append(s[1] / 1000)
      y.append(s[4])
      c.append(colors[s[3]])

    plt.figure(figsize=(14,5))
    ax=plt.axes(title='Euterpe X Composition')
    ax.set_facecolor('black')
    
    plt.scatter(x,y, c=c)
    plt.xlabel("Time")
    plt.ylabel("Pitch")

    yield [500, output1], plt, "Euterpe-X-Music-Composition.mid", (44100, audio)
        
#=================================================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", default=False, help="share gradio app")
    parser.add_argument("--port", type=int, default=7860, help="gradio server port")
    opt = parser.parse_args()

    print('Loading model...')

    SEQ_LEN = 2048
    
    # instantiate the model
    
    model = TransformerWrapper(
        num_tokens = 3344,
        max_seq_len = SEQ_LEN,
        attn_layers = Decoder(dim = 1024, depth = 60, heads = 8)
    )

    model = AutoregressiveWrapper(model)
    
    model = torch.nn.DataParallel(model)
    
    model.cpu()
    print('=' * 70)
    
    print('Loading model checkpoint...')
    
    model.load_state_dict(torch.load('Euterpe_X_Large_Trained_Model_100000_steps_0.477_loss_0.8533_acc.pth', map_location='cpu'))
    print('=' * 70)
    
    model.eval()    
    
    print('Done!')
    
    app = gr.Blocks()
    with app:
        gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>Euterpe X</h1>")
        gr.Markdown("![Visitors](https://api.visitorbadge.io/api/visitors?path=asigalov61.Euterpe-X&style=flat)\n\n"
                    "[SOTA] [PyTorch 2.0] [638M] [85.33% acc] Full-attention multi-instrumental music transformer for supervised music generation, optimized for speed, efficiency, and performance\n\n"
                    "Check out [Euterpe X](https://github.com/asigalov61/Euterpe-X) on GitHub!\n\n"
                    "[Open In Colab]"
                    "(https://colab.research.google.com/github/asigalov61/Euterpe-X/blob/main/Euterpe_X.ipynb)"
                    " for faster execution and endless generation"
                        )
        
        input_drums = gr.Checkbox(label="Drums Controls", value = False, info="Drums present or not")
        input_instrument = gr.Radio(["Piano", "Guitar", "Bass", "Violin", "Cello", "Harp", "Trumpet", "Sax", "Flute", "Choir", "Organ"], value="Piano", label="Lead Instrument Controls", info="Desired lead instrument")       
        input_num_tokens = gr.Slider(16, 512, value=256, label="Number of Tokens", info="Number of tokens to generate")
        run_btn = gr.Button("generate", variant="primary")

        output_midi_seq = gr.Variable()
        output_audio = gr.Audio(label="output audio", format="mp3", elem_id="midi_audio")
        output_plot = gr.Plot(label="output plot")
        output_midi = gr.File(label="output midi", file_types=[".mid"])
        run_event = run_btn.click(GenerateMIDI, [input_num_tokens, input_drums, input_instrument], [output_midi_seq, output_plot, output_midi, output_audio])
        
        app.queue(concurrency_count=1).launch(server_port=opt.port, share=opt.share, inbrowser=True)
