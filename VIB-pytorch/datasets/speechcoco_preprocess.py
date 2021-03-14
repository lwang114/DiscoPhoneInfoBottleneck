from speechcoco.speechcoco import SpeechCOCO
import os

def create_phone_info(data_path, out_file='mscoco_val_phone_info.json'):
  db = SpeechCoco(f'{data_path}/val_2014/val_2014.sqlite3', verbose=False)

  segment_file = os.path.join(data_path, f"val2014/mscoco_val_word_segments.txt") # TODO Check name
  examples = []
  with open(segment_file, 'r') as segment_f, 
       open(out_file, 'w') as out_f:
    
    for line in segment_f:
      parts = line.strip().split()
      audio_id = parts[0]
      token = parts[1]
      print(audio_id, token)
      begin = float(parts[2])
      end = float(parts[2])
      caption = db.selectCaptions(int(audio_id.split('_')[1]))
      for capt in captions:
         for word_info in capt.timecode.parse():
            if word_info['begin'] == begin and word_info['end'] == end:
              print('found') # XXX
              
              out_dict = {'word': token,
                          'phonemes': []}
              for syl in word_info['syllable']:
                for phn in syl['phoneme']:
                  out_dict['phonemes'].append(phn)
              out_f.write('{}\n'.format(json.dumps(out_dict)))

if __name__ == '__main__':
  data_path = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/'
  create_phone_info(data_path) 

