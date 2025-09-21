import pandas as pd
import os
import pyterrier as pt
from ir_measures import *
from pyterrier_pisa import PisaIndex
#from corpus_graph import CorpusGraph
import pickle
import os.path
from pyterrier_dr import FlexIndex, TasB, TctColBert


def rnd(v):
  if isinstance(v, float):
    return round(v, 4)
  return v

def test(dataset, label0, label, m):
  fname0 = 'results/' + label0.replace('\t', '_') + '.res'
  fname = 'results2/' + label.replace('\t', '_') + '.res'
  if os.path.exists(fname) and os.path.exists(fname0):
    res0 = pt.io.read_results(fname0)
    res = pt.io.read_results(fname)
  else:
    print('ERROR: FILE NOT FOUND')
    exit()
  res = res.groupby('qid').head(10).reset_index(drop=True)#10
  res = res.groupby('qid')
  total = 0
  tdiv = 0
  for qid, df in res:
    #print(qid)
    #print(df)
    #print(res0[res0['qid'] == qid])
    dff = df.docno.isin(res0[res0['qid'] == qid].docno).astype(int)
    #print(dff)
    #print(dff.sum())
    total += dff.sum()
    tdiv += 1
    #exit()
  #exit()
  #res.groupby('qid').head(10).reset_index(drop=True)
  #print(res0)
  #print(res)
  tdiv *= 10
  print(tdiv)
  print(total/tdiv)
  #exit()

bm25 = PisaIndex.from_dataset('msmarco_passage', threads=1).bm25()
model = TasB.dot(batch_size=1) # or other model
idx = FlexIndex('index/msmarco-passage.tasb.flex')


#for m in [nDCG@10, nDCG@1000, R(rel=2)@1000]:
for datasetname in ['', 'd20']:
  if datasetname == '':
    dataset = pt.get_dataset('irds:msmarco-passage/trec-dl-2019/judged')
  else:
    dataset = pt.get_dataset('irds:msmarco-passage/trec-dl-2020/judged')

  for m in [10]:
    print(str(m))
  #  for j in range(11):
    for j in [0]:
      for k in [16, 64]:
        print('k: ' + str(k))
        for ni in ['']:

          for hops in [1]:
            for r in ([1000]):
              bm25.num_results = r
              test(dataset, f'bm25{datasetname}', f'ladr\tk={k}\thops={hops}\t{r}\tlup{ni}={j}{datasetname}', m)
              test(dataset, f'bm25{datasetname}', f'ladr_hnsw\tk={k}\thops={hops}\t{r}\tefc={40}{datasetname}', m)

      for k in [16, 64]:
        print('k: ' + str(k))
        for ni in ['']:

          for r in [1000]:
            for depth in [100]:
              test(dataset, f'bm25{datasetname}', f'adaladr\tk={k}\tr={r}\t{depth}\tlup{ni}={j}{datasetname}', m)
              test(dataset, f'bm25{datasetname}', f'adaladr_hnsw\tk={k}\tr={r}\t{depth}\tefc={40}{datasetname}', m)
