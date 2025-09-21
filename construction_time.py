import pandas as pd
import os
import pyterrier as pt
from ir_measures import *
from pyterrier_pisa import PisaIndex
# from corpus_graph import CorpusGraph
import pickle
import os.path
from pyterrier_dr import FlexIndex, TasB, TctColBert
import time

start = time.time()
print("hello")
end = time.time()
print(end - start)

dataset = pt.get_dataset('irds:msmarco-passage/trec-dl-2019/judged')
datasetname = ''

model = TasB.dot(batch_size=1) # or other model
idx = FlexIndex('index/msmarco-passage.tasbtime.flex')

# pipeline = model.doc_encoder() >> idx
# start = time.time()
# pipeline.index(dataset.get_corpus_iter())
# end = time.time()
# print(end - start)

print('main code time')
for k in [64]:
    for r in [1000]:
        for depth in [100]:
            for ef_construction in [10, 20, 30, 40]:
                start = time.time()
                grp = idx.faiss_hnsw_graph(neighbours=k, ef_construction=ef_construction)
                end = time.time()

                print(end - start)