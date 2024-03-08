import os
import argparse
import numpy as np
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
# from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.bleu.bleu_b_norm import computeMaps, bleuFromMaps
import bleu

def calc_recall(pred, tgt):
    prds = pred.split()
    tgs = tgt.split()
    count = 0

    for prd in prds:
        if prd in tgs:
            count += 1
    return count / len(tgs)


def calc_prec(pred, tgt):
    prds = pred.split()
    tgs = tgt.split()
    count = 0 
    for prd in prds:
        if prd in tgs:
            count += 1
    return count/ len(prds)

def cal_metrics(prd_dir, gold_dir, prd_index, gold_index):
    predictions = list()
    golds = list()
    
    if prd_index:
        with open(prd_dir) as f:
            predictions =  [line.strip() for line in f.readlines()]
    else:
        with open(prd_dir) as f:
            predictions =  [str(i)+ '\t' + line.strip() for i, line in enumerate(f.readlines())]

    if gold_index:
        with open(gold_dir) as f:
            golds =  [line.strip() for line in f.readlines()]
    else:
        with open(gold_dir) as f:
            golds =  [str(i)+ '\t' + line.strip() for i, line in enumerate(f.readlines())]
        
    tmp_file = 'tmpgold.txt'
    with open(tmp_file,'w+') as f:
        f.write('\n'.join(golds))
    
    EM = list()
    precs = list()
    recall = list()
    for i, (ref, gold) in enumerate(zip(predictions, golds)):
        EM.append(ref.split() == gold.split())
        precs.append(calc_prec(ref, gold))
        recall.append(calc_recall(ref, gold))

    EM = round(np.mean(EM)*100, 3)
    precs = round(np.mean(precs)*100, 3)
    recall = round(np.mean(recall) * 100, 3)

    res = {k: [' '.join(v.split('\t')[1:]).strip().lower()] for k, v in enumerate(predictions)}
    tgt = {k: [' '.join(v.split('\t')[1:]).strip().lower()] for k, v in enumerate(golds)}
    
    # precision 1-gram
    print("predict lines: ", len(res))
    print("refs lines: ", len(tgt))

    print("EM = %s" % (str(EM)))
    print("precs = %s" % (str(precs)))
    print("recall = %s" % (str(recall)))
    score_Meteor, scores_Meteor = Meteor().compute_score(tgt, res)
    print("Meteor: %s" % (float(score_Meteor)*100))
    score_Rouge, scores_Rouge = Rouge().compute_score(tgt, res)
    print("ROUGE-L: %s" % (float(score_Rouge)*100))

    (goldMap, predictionMap) = bleu.computeMaps(predictions, tmp_file)
    dev_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 3)
    (goldMap2, predictionMap2) = bleu.computeMaps(predictions, tmp_file)
    dev_bleu2 = round(bleu.bleuFromMaps(goldMap2, predictionMap2)[0], 3)
    print("  %s = %s " % ("bleu-4", str(dev_bleu)))
    print("  %s = %s " % ("bleu-normal", str(dev_bleu2)))

    # ref_sentence_lst = [x.strip() for x in open(sys.argv[1]) if x.strip()]
    
    # # with open("tmp_ref.txt","w") as f:
    # #   for idx, ref_sentence in enumerate(ref_sentence_lst):
    # #     f.write("{}\t{}\n".format(idx, ref_sentence))
    
    # with open("tmp_ref.txt","w") as f:
    #     for refs in ref_sentence_lst:
    #     idx, ref_sentence = refs.split('\t')
    #     f.write("{}\t{}\n".format(idx, ref_sentence))

    # reference_file = "tmp_ref.txt"
    # predictions = []
    # for row in sys.stdin:
    #     idx, sent = row.split('\t')
    #     predictions.append("{}\t{}".format(idx,sent))
    
    (goldMap, predictionMap) = computeMaps(predictions, tmp_file) 
    print ("Bleu-B-Norm: ", bleuFromMaps(goldMap, predictionMap)[0])

    os.remove(tmp_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prd_dir", default=None, type=str,
                        help="File dir to read predict msg")
    parser.add_argument("--gold_dir", default=None, type=str,
                        help="File dir to read gold msg")
    parser.add_argument("--prd_index", action='store_true',
                        help="Contain index line in file predict")
    parser.add_argument("--gold_index", action='store_true',
                        help="Contain index line in file gold")
    
    args = parser.parse_args()

    cal_metrics(args.prd_dir, args.gold_dir, args.prd_index, args.gold_index)

if __name__ == "__main__":
    main()
