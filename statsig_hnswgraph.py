import pandas as pd
import os
import pyterrier as pt
from ir_measures import *
from pyterrier_pisa import PisaIndex
# from corpus_graph import CorpusGraph
import pickle
import os.path
from pyterrier_dr import FlexIndex, TasB, TctColBert

#dataset = pt.get_dataset('irds:msmarco-passage/trec-dl-2019/judged')
dataset = pt.get_dataset('irds:msmarco-passage/trec-dl-2020/judged')
datasetname = 'd20'

def rnd(v):
    if isinstance(v, float):
        return round(v, 4)
    return v


def test(label0, label, m, efc):
    fname0 = 'results2/' + label0.replace('\t', '_') + '.res'
    fname = 'results2/' + label.replace('\t', '_') + '.res'
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
        names=["Org", "RP:" + str(int(efc))],
        baseline=0
    )  # .iloc[0]
    print(resu)


bm25 = PisaIndex.from_dataset('msmarco_passage', threads=1).bm25()
model = TasB.dot(batch_size=1)  # or other model
idx = FlexIndex('index/msmarco-passage.tasb.flex')

for m in [nDCG@10, nDCG@1000, R(rel=2)@1000]:
#for m in [nDCG @ 1000]:
    print(str(m))

    for k in [16, 64]:
        print('k: ' + str(k))
    
        for hops in [1]:
            for r in ([1000]):
                for ef_construction in [40, 30, 20, 10]:
                    test(f'ladr_hnsw\tk={k}\thops={hops}\t{r}\tefc={40}{datasetname}', f'ladr_hnsw\tk={k}\thops={hops}\t{r}\tefc={ef_construction}{datasetname}', m, ef_construction)
    
    for k in [16, 64]:
        print('k: ' + str(k))
    
        for r in [1000]:
            for depth in [100]:
                for ef_construction in [40, 30, 20, 10]:
                    test(f'adaladr_hnsw\tk={k}\tr={r}\t{depth}\tefc={40}{datasetname}',
                     f'adaladr_hnsw\tk={k}\tr={r}\t{depth}\tefc={ef_construction}{datasetname}', m, ef_construction)

    for n in [64]:

        for ef_search in [16, 32, 64, 1111]:
            print('ef: ' + str(ef_search))

            if ef_search != 1111:
                for ef_construction in [40, 30, 20, 10]:
                    test(f'hnsw\t{n}\t{40}\t{ef_search}{datasetname}', f'hnsw\t{n}\t{ef_construction}\t{ef_search}{datasetname}', m, ef_construction)
            else:
                for ef_construction in [40, 30, 20, 10]:
                    test(f'hnsw\t{n}\t{40}\tnsbq{datasetname}', f'hnsw\t{n}\t{ef_construction}\tnsbq{datasetname}', m, ef_construction)

#    exit()