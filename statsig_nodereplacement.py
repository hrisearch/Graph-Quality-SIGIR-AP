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
    return round(v, 4)
  return v

def test(label0, label, m):
  fname0 = 'results/' + label0.replace('\t', '_') + '.res'
  fname = 'results/' + label.replace('\t', '_') + '.res'
  if os.path.exists(fname) and os.path.exists(fname0):
    res0 = pt.io.read_results(fname0)
    res = pt.io.read_results(fname)
  else:
    print('ERROR: FILE NOT FOUND')
    exit()
  resu = pt.Experiment(
      [pt.Transformer.from_df(res0), pt.Transformer.from_df(res)],
      dataset.get_topics(),
      dataset.get_qrels(),
      [m],
      names=["Org", "RP:" + str(int(j*10))],
      baseline=0
  )#.iloc[0]
  print(resu)


bm25 = PisaIndex.from_dataset('msmarco_passage', threads=1).bm25()
model = TasB.dot(batch_size=1) # or other model
idx = FlexIndex('index/msmarco-passage.tasb.flex')


for m in [nDCG@10, nDCG@1000, R(rel=2)@1000]:
#for m in [R(rel=2)@1000]:
  print(str(m))
  for j in range(11):
#  for j in [1]:
    for k in [16, 64]:
      print('k: ' + str(k))
      for ni in ['', 'n1', 'n2', 'n3']:

        for hops in [1]:
          for r in ([1000]):
            bm25.num_results = r
            test(f'ladr\tk={k}\thops={hops}\t{r}\tlup{ni}={0}{datasetname}', f'ladr\tk={k}\thops={hops}\t{r}\tlup{ni}={j}{datasetname}', m)

        for r in [1000]:
          for depth in [100]:
            test(f'adaladr\tk={k}\tr={r}\t{depth}\tlup{ni}={0}{datasetname}', f'adaladr\tk={k}\tr={r}\t{depth}\tlup{ni}={j}{datasetname}', m)

  for n in [16]:
    
      for ef in [16, 64, 1111]:
          print('ef: ' + str(ef))
    
          for ni in ['', 'n1', 'n2', 'n3']:
    
            if ef != 1111:
              test(f'hnsw\t{n}\t{ef}\tlup{ni}={0}', f'hnsw\t{n}\t{ef}\tlup{ni}={j}', m)
            else:
              test(f'hnsw\t{n}\tnsbq\tlup{ni}={0}', f'hnsw\t{n}\tnsbq\tlup{ni}={j}', m)
