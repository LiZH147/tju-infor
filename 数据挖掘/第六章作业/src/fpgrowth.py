#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, collections, argparse

class Node:
    def __init__(self, item, cnt=1, parent=None):
        self.item, self.cnt, self.parent = item, cnt, parent
        self.children = {}

def insert_tree(trans, root, header):
    cur = root
    for item in trans:
        if item in cur.children:
            cur.children[item].cnt += 1
        else:
            new_nd = Node(item, parent=cur)
            cur.children[item] = new_nd
            header[item] = header.get(item, []) + [new_nd]
        cur = cur.children[item]

def build_fp(transactions, min_sup_cnt):
    # 1. 第一次扫描：统计频次
    item_cnt = collections.Counter(item for tran in transactions for item in tran)
    item_cnt = {k: v for k, v in item_cnt.items() if v >= min_sup_cnt}
    if not item_cnt:
        return None, {}
    # 2. 第二次扫描：建树
    root = Node(None)
    header = {}
    for tran in transactions:
        filtered = [item for item in tran if item in item_cnt]
        filtered.sort(key=lambda x: (-item_cnt[x], x))
        insert_tree(filtered, root, header)
    return root, header

def mine_tree(header, min_sup_cnt, prefix, freq_list):
    for item in sorted(header, key=lambda x: sum(nd.cnt for nd in header[x])):
        new_freq = prefix + [item]
        freq_list.append(new_freq)
        # 构造条件模式基
        cond_pats = []
        for nd in header[item]:
            path = []
            par = nd.parent
            while par and par.item is not None:
                path.append(par.item)
                par = par.parent
            if path:
                cond_pats.extend([path[::-1]] * nd.cnt)
        # 递归
        cond_root, cond_head = build_fp(cond_pats, min_sup_cnt)
        if cond_head:
            mine_tree(cond_head, min_sup_cnt, new_freq, freq_list)

def fp_growth(transactions, min_sup):
    min_sup_cnt = int(min_sup * len(transactions) + 1e-6)
    root, header = build_fp(transactions, min_sup_cnt)
    freq_list = []
    if header:
        mine_tree(header, min_sup_cnt, [], freq_list)
    return freq_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True)
    parser.add_argument("-s", "--min_sup", type=float, default=0.3)
    args = parser.parse_args()
    transactions = [set(line.strip().split()) for line in open(args.file, encoding='utf-8') if line.strip()]
    freq_list = fp_growth(transactions, args.min_sup)
    print("=== FP-Growth Frequent Itemsets ===")
    for item in sorted(freq_list, key=lambda x: (-len(x), sorted(x))):
        print(sorted(item))