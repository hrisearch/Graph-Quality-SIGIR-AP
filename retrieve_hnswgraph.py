import pandas as pd
import os
import pyterrier as pt
from ir_measures import *
from pyterrier_pisa import PisaIndex
#from corpus_graph import CorpusGraph
import pickle
import os.path
from pyterrier_dr import FlexIndex, TasB, TctColBert

dataset = pt.get_dataset('irds:msmarco-passage/trec-dl-2020/judged')
datasetname = 'd20'
def rnd(v):
  if isinstance(v, float):
    return round(v, 3)
  return v

def test(label, p):
  fname = 'results2/' + label.replace('\t', '_') + '.res'
  if not os.path.exists(fname):
    p = p()
    res = p(dataset.get_topics())
    pt.io.write_results(res, fname)
  else:
    res = pt.io.read_results(fname)
  res = pt.Experiment(
      [pt.Transformer.from_df(res)],
      dataset.get_topics(),
      dataset.get_qrels(),
      [nDCG@1000, nDCG@10, R(rel=2)@1000]
  ).iloc[0]
  print(label, rnd(res['nDCG@10']), rnd(res['nDCG@1000']), rnd(res['R(rel=2)@1000']))


bm25 = PisaIndex.from_dataset('msmarco_passage', threads=1).bm25()
model = TasB.dot(batch_size=1) # or other model
idx = FlexIndex('index/msmarco-passage.tasb.flex')


#exit()
test(f'bm25{datasetname}', lambda: bm25)

for r in [1000]:
  bm25.num_results = r
  test(f'rerank\t{r}{datasetname}', lambda: bm25 >> model.query_encoder() >> idx.scorer())

test(f'np{datasetname}', lambda: model.query_encoder() >> idx.np_retriever())

for n in [64]:
  for ef_construction in [10, 20, 30, 40]:
    for ef_search in [16, 32, 64, 1111]:

      if ef_search != 1111:
        test(f'hnsw\t{n}\t{ef_construction}\t{ef_search}{datasetname}', lambda: model.query_encoder() >> idx.faiss_hnsw_retriever(neighbours=n, ef_construction=ef_construction, ef_search=ef_search, qbatch=1))
      else:
        test(f'hnsw\t{n}\t{ef_construction}\tnsbq{datasetname}', lambda: model.query_encoder() >> idx.faiss_hnsw_retriever(neighbours=n, ef_construction=ef_construction, search_bounded_queue=False, qbatch=1))

for k in [16, 64]:
  for hops in [1]:
    for r in ([1000]):
      for ef_construction in [10, 20, 30, 40]:
        bm25.num_results = r
        #test(f'ladr_hnsw\tk={k}\thops={hops}\t{r}', lambda: bm25 >> model.query_encoder() >> idx.ladr(k, hops))
        test(f'ladr_hnsw\tk={k}\thops={hops}\t{r}\tefc={ef_construction}{datasetname}', lambda: bm25 >> model.query_encoder() >> idx.ladr(idx.faiss_hnsw_graph(neighbours=k, ef_construction=ef_construction)))


for k in [16, 64]:
  for r in [1000]:
    for depth in [100]:
      for ef_construction in [10, 20, 30, 40]:
        bm25.num_results = r
        test(f'adaladr_hnsw\tk={k}\tr={r}\t{depth}\tefc={ef_construction}{datasetname}', lambda: bm25 >> model.query_encoder() >> idx.ada_ladr(idx.faiss_hnsw_graph(neighbours=k, ef_construction=ef_construction), depth=depth))
