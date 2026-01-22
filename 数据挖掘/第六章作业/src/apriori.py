#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, itertools, collections

def load_data(path):
    with open(path, encoding='utf-8') as f:
        return [line.strip().split() for line in f if line.strip()]

def get_support(itemset, transactions):
    count = sum(1 for tran in transactions if itemset.issubset(tran))
    return count / len(transactions)

def apriori(transactions, min_sup, min_conf):
    C1 = {frozenset([item]) for tran in transactions for item in tran}
    L1 = {c for c in C1 if get_support(c, transactions) >= min_sup}
    L = [L1]
    k = 2
    while True:
        Ck = {i1 | i2 for i1 in L[-1] for i2 in L[-1] if len(i1 | i2) == k}
        Lk = {c for c in Ck if get_support(c, transactions) >= min_sup}
        if not Lk:
            break
        L.append(Lk)
        k += 1
    # 生成关联规则
    rules = []
    for freq_set in {item for subset in L for item in subset}:
        if len(freq_set) < 2:
            continue
        for r in range(1, len(freq_set)):
            for ante in map(frozenset, itertools.combinations(freq_set, r)):
                conf = get_support(freq_set, transactions) / get_support(ante, transactions)
                if conf >= min_conf:
                    rules.append((ante, freq_set - ante, conf))
    return {item for subset in L for item in subset}, rules

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True, help="transaction file")
    parser.add_argument("-s", "--min_sup", type=float, default=0.3)
    parser.add_argument("-c", "--min_conf", type=float, default=0.7)
    args = parser.parse_args()

    transactions = [set(t) for t in load_data(args.file)]
    freq, rules = apriori(transactions, args.min_sup, args.min_conf)
    print("=== Frequent Itemsets ===")
    for item in sorted(freq, key=lambda x: (-len(x), sorted(x))):
        print(f"{sorted(item)}  sup={get_support(item, transactions):.2f}")
    print("\n=== Association Rules ===")
    for ante, cons, conf in sorted(rules, key=lambda x: -x[2]):
        print(f"{sorted(ante)} -> {sorted(cons)}  conf={conf:.2f}")