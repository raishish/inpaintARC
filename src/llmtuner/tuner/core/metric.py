import numpy as np
from typing import Dict, Sequence, Tuple, Union
import torch.nn.functional as F
import torch
from llmtuner.extras.constants import IGNORE_INDEX
import re

def f1_score(preds, labels):
    f1 = []
    for pred, label in zip(preds, labels):
        f1.append(len(np.intersect1d(pred, label))/len(pred))
    return np.mean(f1)

def compute_nll(eval_preds: Sequence[Union[np.ndarray, Tuple[np.ndarray]]]) -> Dict[str, float]:
    preds, labels = eval_preds
    f1 = f1_score(preds, labels)
    return {"eval_f1": f1}

def check_eq(left_str, right_str):
    left_matches = re.match(r'(\d+)([+\-*/])(\d+)', left_str)
    if left_matches:
        return eval(left_str) == float(right_str)
    else:
        return False

def compute_rm_acc(eval_preds):
    # (N, s) (N, s+1)
    preds, labels = eval_preds
    score_dict = {}
    score_dict.setdefault(f"acc-err", [])
    score_dict.setdefault(f"acc-cor", [])
    for pred, label in zip(preds, labels):
        t, label = label[0], label[1:]
        # for c, _t in zip(correct, t):
        l = label[label!=-100]
        p = pred[label!=-100] 
        res = l&p 
        score_dict.setdefault(f"acc-{t}-cor", [])
        score_dict.setdefault(f"acc-{t}-err", [])
        score_dict[f"acc-{t}-cor"].extend(res[l==1])
        score_dict[f"acc-{t}-err"].extend(res[l==0])
        score_dict[f"acc-cor"].extend(res[l==1])
        score_dict[f"acc-err"].extend(res[l==0])
    return {k: float(np.mean(v)) for k, v in score_dict.items()}

def compute_acc(eval_preds: Sequence[Union[np.ndarray, Tuple[np.ndarray]]], tokenizer, data_name) -> Dict[str, float]:
    preds, labels = eval_preds
    score_dict = {"acc": []}

    labels = torch.tensor(labels)
    ignore_mask = labels==IGNORE_INDEX
    labels.masked_fill_(ignore_mask, tokenizer.pad_token_id)
    if preds.shape == labels.shape:
        # supppose pred and label have same shape (training stage)
        preds = torch.tensor(preds)
        preds.masked_fill_(ignore_mask, tokenizer.pad_token_id)
    else:
        preds = torch.tensor(preds)
        preds.masked_fill_(preds==IGNORE_INDEX, tokenizer.pad_token_id)

    preds = tokenizer.batch_decode(preds.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
    labels = tokenizer.batch_decode(labels.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)

    for pred, label in zip(preds, labels):
        if 'cd' in data_name: ## countdown
            subequations = pred.split(',')  # sub-equations
            match = True
            for subeq in subequations:
                try:
                    left, right = subeq.split('=')
                    match &= check_eq(left, right)
                except:
                    match = False
                if not match:
                    break
            answer = label.split('=')[-1]
            pred_ans = pred.split('=')[-1]

            score_dict["acc"].append(match and (answer == pred_ans))
        # elif 'sat' in data_name:
        #     # score_dict["acc"].append(0)

        #     # sat-v2
        #     subphases = pred.split('/')
        #     corr = True
        #     for subphase in subphases:
        #         if 'T' not in subphase:
        #             score_dict["acc"].append(0)
        #             corr = False
        #             break
        #     if corr:
        #         score_dict["acc"].append(1)

        elif 'path' in data_name:
            def reverse_check(gold, pred):
                try:
                    items = pred.split('/')
                    reversed_pred = "/".join([f'{i.split(",")[1]},{i.split(",")[0]}' for i in items[::-1]])
                    return reversed_pred == gold
                except:
                    return False

            score_dict["acc"].append((pred == label) or reverse_check(label, pred))
        else: ## chess, sudoku, prime
            pred = pred.strip().split(' ') # pred can have multiple actions
            label = label.strip().split(' ') # labels can have multiple actions
            pred = pred[:len(label)] # chess only take next move

            score_dict["acc"].append(pred==label)

    return {k: float(np.mean(v)) for k, v in score_dict.items()}
