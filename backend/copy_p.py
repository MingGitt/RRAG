import json
import os
from collections import defaultdict

# 1. 检查 qrel 文件内容格式
print("=" * 50)
print("1. 检查 qrel 文件格式")
print("=" * 50)

qrel_file = "D:/code/rag/FSR/data/nfcorpus/qrels/test.tsv"
with open(qrel_file, 'r') as f:
    print("前5行内容:")
    for i, line in enumerate(f):
        if i >= 5:
            break
        print(f"  {line.strip()}")
        parts = line.strip().split('\t')
        print(f"    列数: {len(parts)}, 内容: {parts}")

# 2. 检查评估结果中的召回情况
print("\n" + "=" * 50)
print("2. 检查评估结果详情")
print("=" * 50)

details_file = "D:/code/rag/FSR/backend/nfcorpus_full_retrieval_details.jsonl"
if os.path.exists(details_file):
    query_stats = []
    with open(details_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            retrieved = data.get('retrieved_docs', [])
            
            # 统计前50和前100中相关文档数
            rel_50 = sum(1 for d in retrieved[:50] if d.get('relevance', 0) == 1)
            rel_100 = sum(1 for d in retrieved[:100] if d.get('relevance', 0) == 1)
            
            query_stats.append({
                'qid': data['query_id'],
                'rel_50': rel_50,
                'rel_100': rel_100,
                'total_retrieved': len(retrieved)
            })
            
            if len(query_stats) >= 10:  # 只看前10条
                break
    
    # 统计有多少 query 的 rel_100 > rel_50
    increased = sum(1 for s in query_stats if s['rel_100'] > s['rel_50'])
    print(f"前10个查询中:")
    print(f"  有 {increased} 个查询在 Top100 中找到的相关文档多于 Top50")
    for s in query_stats[:5]:
        print(f"  Query {s['qid']}: Top50相关={s['rel_50']}, Top100相关={s['rel_100']}, 总返回={s['total_retrieved']}")
else:
    print("未找到详情文件")

# 3. 计算理论上的最大 Recall
print("\n" + "=" * 50)
print("3. 理论分析")
print("=" * 50)

# 从 qrel 统计每个 query 的总相关文档数
from collections import defaultdict
total_relevant = defaultdict(int)
with open(qrel_file, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 3:
            qid = parts[0]
            try:
                score = float(parts[2])
                if score >= 1:
                    total_relevant[qid] += 1
            except:
                pass

if total_relevant:
    print(f"test.tsv 统计:")
    print(f"  查询数: {len(total_relevant)}")
    print(f"  平均相关文档数: {sum(total_relevant.values())/len(total_relevant):.1f}")
    print(f"  最大相关文档数: {max(total_relevant.values())}")
    
    # 找出相关文档数 > 50 的查询示例
    high_rel_queries = [(qid, count) for qid, count in total_relevant.items() if count > 50]
    print(f"  相关文档超过50个的查询数: {len(high_rel_queries)}")
    if high_rel_queries:
        print(f"  示例: {high_rel_queries[:3]}")