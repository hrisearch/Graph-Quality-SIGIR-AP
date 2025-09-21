import pyterrier as pt
#pt.init()
import pyterrier_dr
from pyterrier_dr import FlexIndex, TctColBert, TasB
import time

dataset = pt.get_dataset('irds:msmarco-passage/trec-dl-2019/judged')

model = TasB.dot()
idx = FlexIndex('index/msmarco-passage.tasb.flex')

pipeline = model.doc_encoder() >> idx
pipeline.index(dataset.get_corpus_iter())

#idx = FlexIndex('index/msmarco-passage.tasb.flex')
idx = FlexIndex('index/msmarco-passage.tasb.flex')
print(idx.__len__())
start = time.time()
graph = idx.corpus_graph(k=128)
end = time.time()
print(end - start)
print(graph)

graph = idx.corpus_graph2(k=128)
print(graph)

