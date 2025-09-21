import pandas as pd
import os
import pyterrier as pt
from ir_measures import *
from pyterrier_pisa import PisaIndex
#from corpus_graph import CorpusGraph
import pickle
import os.path
from pyterrier_dr import FlexIndex, TasB, TctColBert
                
dataset = pt.get_dataset('irds:msmarco-passage/trec-dl-2019/judged')
datasetname = ''

def rnd(v):
    if isinstance(v, float):
        return round(v, 3)
    return v

def test(label, p):
    fname = 'results4/' + label.replace('\t', '_') + '.res'
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
    print(label, rnd(res['R(rel=2)@1000']))


bm25 = PisaIndex.from_dataset('msmarco_passage', threads=1).bm25()
model = TasB.dot(batch_size=1) # or other model
idx = FlexIndex('index/msmarco-passage.tasb.flex')

for datasetname in ['', 'dl20']:
    if datasetname == '':
        dataset = pt.get_dataset('irds:msmarco-passage/trec-dl-2019/judged')
    else:
        dataset = pt.get_dataset('irds:msmarco-passage/trec-dl-2020/judged')

    # #exit()
    test(f'bm25{datasetname}', lambda: bm25)

    for r in [1000]:
        bm25.num_results = r
        test(f'rerank\t{r}{datasetname}', lambda: bm25 >> model.query_encoder() >> idx.scorer())

    test(f'np{datasetname}', lambda: model.query_encoder() >> idx.np_retriever())

    for strategy in ['befs', 'a*']:
        for ni in ['n3']:
            for k in [16, 64]:
                for j in [0, 8]:
                    for r in [1000]:
                        for depth in [100]:
                            mh = None
                            test(f'adaladr\tk={k}\tr={r}\t{depth}\tlup{ni}={j}{datasetname}{strategy}', lambda: bm25 >> model.query_encoder() >> idx.ada_ladr(k, depth=depth, j=j, n=ni, strategy=strategy, max_hops=mh))

    for strategy in ['hc']:
        for ni in ['n3']:
            for k in [8, 16, 32, 64]:
                for j in [0, 8]:
                    for r in [1000]:
                        for depth in [100]:
                            mh = k
                            test(f'adaladr\tk={k}\tr={r}\t{depth}\tlup{ni}={j}{datasetname}{strategy}', lambda: bm25 >> model.query_encoder() >> idx.ada_ladr(k, depth=depth, j=j, n=ni, strategy=strategy, max_hops=mh))

for datasetname in ['', 'dl20']:
    if datasetname == '':
        dataset = pt.get_dataset('irds:msmarco-passage/trec-dl-2019/judged')
    else:
        dataset = pt.get_dataset('irds:msmarco-passage/trec-dl-2020/judged')

    for strategy in ['befs', 'a*']:
        for k in [16, 64]:
          for r in [1000]:
            for depth in [100]:
              for ef_construction in [40, 10]:
                bm25.num_results = r
                gk = k
                mh = None
                test(f'adaladr_hnsw\tk={k}\tr={r}\t{depth}\tefc={ef_construction}{datasetname}{strategy}', lambda: bm25 >> model.query_encoder() >> idx.ada_ladr(idx.faiss_hnsw_graph(neighbours=gk, ef_construction=ef_construction), depth=depth, strategy=strategy, max_hops=mh))

    for strategy in ['hc']:
        for k in [16, 64]:
            for r in [1000]:
                for depth in [100]:
                    for ef_construction in [40, 10]:
                        bm25.num_results = r    
                        gk = 4
                        mh = k
                        test(f'adaladr_hnsw\tk={k}\tr={r}\t{depth}\tefc={ef_construction}{datasetname}{strategy}',
                             lambda: bm25 >> model.query_encoder() >> idx.ada_ladr(
                                 idx.faiss_hnsw_graph(neighbours=gk, ef_construction=ef_construction), depth=depth,
                                 strategy=strategy, max_hops=mh))
