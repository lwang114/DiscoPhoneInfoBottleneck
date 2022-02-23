from speechcoco.speechcoco import SpeechCoco
import os
import json

def create_phone_info(data_path, split):
  db = SpeechCoco(f'{data_path}/val2014/val_2014.sqlite3', verbose=False)
  out_file='mscoco_{split}_phone_info.json'

  segment_file = os.path.join(data_path, f"val2014/mscoco_{split}_word_segments.txt") # TODO Check name
  examples = []
  with open(segment_file, 'r') as segment_f,\
       open(os.path.join(data_path, f"{split}2014", out_file), 'w') as out_f:
    idx = 0
    for line in segment_f:
      idx += 1
      parts = line.strip().split()
      audio_id = parts[0]
      token = parts[1]
      print(audio_id, token)
      begin = float(parts[2])
      end = float(parts[3])
      captions = db.selectCaptions(int(audio_id.split('_')[1]))
      for capt in captions:
         for word_info in capt.timecode.parse():
           if word_info['begin'] == begin and word_info['end'] == end:
              out_dict = {'audio_id': audio_id,
                          'word': token,
                          'begin': begin,
                          'end': end,
                          'phonemes': []}
              for syl in word_info['syllable']:
                for phn in syl['phoneme']:
                  out_dict['phonemes'].append(phn)
              out_f.write('{}\n'.format(json.dumps(out_dict)))

if __name__ == '__main__':
  data_path = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/'
  create_phone_info(data_path) 
