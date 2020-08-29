import text_eval
import public_parsing_ops
import numpy as np

TEXT = 'thermo.txt'
shape = 1024
OUT_TEXT = f'thermo_{shape}.txt'
print(OUT_TEXT)

def append(line):
    with open(OUT_TEXT, 'a', buffering=1) as redf:
        redf.write(line)

text = open(TEXT, "r", encoding="utf-8").read()

_SPM_VOCAB = 'ckpt/c4.unigram.newline.10pct.96000.model'
encoder = public_parsing_ops.create_text_encoder("sentencepiece",
                                                 _SPM_VOCAB)


input_ids_raw = encoder.encode(text)
input_ids_split = list(zip(*([iter(input_ids_raw)]*shape)))

for idx, input_ids in enumerate(input_ids_split):
    arr = np.expand_dims(np.asarray(input_ids), axis=0)
    decoded = text_eval.ids2str(encoder, arr, None)
    append(f'Section {idx}\n{decoded}\n\n')
