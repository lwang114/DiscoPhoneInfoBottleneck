import os
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
from praatio import textgrid
matplotlib.rcParams['font.size'] = 8.0


def convert_to_segments(phns, frame_rate):
  prev = ''
  segments = []
  for i, phn in enumerate(phns):
    if phn != prev:
      segments.append({'begin': i * frame_rate,
                       'end': (i + 1) * frame_rate,
                       'label': phn})
      prev = phn
    else:
      segments[-1]['end'] += frame_rate
  return segments


def sgram(wav_path):
    fs, x = wavfile.read(wav_path)
    f, t, S = spectrogram(x, fs=fs, 
                    window=('hamming', 0.25),
                    noverlap=160, nfft=256,
                    scaling='spectrum')
    return f, t, S
  
    
def plot_sgram(wav, filename):
    f, t, S = sgram(wav)
    fig, ax = plt.subplots(figsize=(14, 4))
    plt.imshow(S)
    ax.set_xticks(t)
    ax.set_yticks(f)
    #plt.show()
    plt.savefig(filename)
    plt.close()
    return t.max()
    

def plot_segment_bars(paths, filename, dur=None):
    """
    Plot the segment boundaries as vertical bars 
    """
    fig, ax = plt.subplots(figsize=(14, 0.5))
    colors = ['C{i}'.format(i) for i in range(len(paths))]
    offsets = list(range(len(paths)))  
    
    boundaries = []
    for i, path in enumerate(paths):
        tg = textgrid.Textgrid(path, False)
        segments = tg.tierDict['phone segments'].entryList
        if dur is None:
            dur = segments[-1].end
        boundaries = [s.start for s in segments if s.start < dur]
        labels = [s.label for s in segments if s.start < dur]
        boundaries.append(dur)
        
        ax.eventplot(boundaries, 
                     colors=colors[i], 
                     lineoffsets=offsets[i],
                     linestyles='--')
        
        # Annotate the labels
        prev = None
        for start, label in zip(boundaries[:-1], labels):
            if prev is not None and (start - prev) <= 0.02:
                plt.annotate(label, (start, offsets[i]+0.5))
            else:
                plt.annotate(label, (start, offsets[i]-0.5))
            prev = start
            
    # plt.show()
    plt.savefig(filename)
    plt.close()

    
def plot_segmentation(segment_paths,
                      wav_path,
                      out_path,
                      debug=False):
  """
  Args:
      segment_paths: list of str, .TextGrid files containing gold/predicted segmentations
      wav_path: str, filename of a .wav file
      out_path: str, directory to store the spectrogram and its annotations
  """
  out_path = Path(out_path)
  dur = plot_sgram(wav_path, out_path / 'spectrogram.png')
  plot_segment_bar(segment_paths, out_path / 'segment_bar.png', dur)


  
       
