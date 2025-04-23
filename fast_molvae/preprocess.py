import torch
import torch.nn as nn
from multiprocessing import Pool

import math, random, sys
from optparse import OptionParser
import cPickle as pickle  # will be patched by your sed block

from fast_jtnn import *
import rdkit

def tensorize_with_index(index_smiles, assm=True):
    idx, smiles = index_smiles
    try:
        mol_tree = MolTree(smiles)
        mol_tree.recover()
        if assm:
            mol_tree.assemble()
            for node in mol_tree.nodes:
                if node.label not in node.cands:
                    node.cands.append(node.label)

        del mol_tree.mol
        for node in mol_tree.nodes:
            del node.mol

        return mol_tree
    except Exception as e:
        print("❌ Line %d: Failed to process %s — %s" % (idx + 1, smiles, str(e)), file=sys.stderr)
        return None

if __name__ == "__main__":
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = OptionParser()
    parser.add_option("-t", "--train", dest="train_path")
    parser.add_option("-n", "--split", dest="nsplits", default=10)
    parser.add_option("-j", "--jobs", dest="njobs", default=8)
    opts,args = parser.parse_args()
    opts.njobs = int(opts.njobs)
    num_splits = int(opts.nsplits)

    with open(opts.train_path) as f:
        data = [(i, line.strip().split()[0]) for i, line in enumerate(f)]

    print("📦 Starting preprocessing for %d molecules..." % len(data))

    pool = Pool(opts.njobs)
    all_data = pool.map(tensorize_with_index, data)
    pool.close()
    pool.join()

    all_data = [d for d in all_data if d is not None]
    print("✅ Successfully processed %d / %d molecules." % (len(all_data), len(data)))

    le = (len(all_data) + num_splits - 1) / num_splits  # will be patched by sed to use //
    
    for split_id in xrange(num_splits):  # patched by sed to use range
        st = split_id * le
        sub_data = all_data[st : st + le]

        with open('tensors-%d.pkl' % split_id, 'wb') as f:
            pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)
