"""混合信道嵌入模块"""

import pickle
import numpy as np
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import requests


class MixedEmbedding:
    """混合信道嵌入管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = self._create_embedding_client()
        self.embedding_index = None
        self._load_cached_index()
    
    def _create_embedding_client(self) -> Dict[str, str]:
        """创建嵌入API客户端"""
        import os
        return {
            "api_key": os.getenv("EMBEDDING_API_KEY", ""),
            "base_url": os.getenv("EMBEDDING_BASE_URL", "https://api.openai-hk.com/v1"),
            "model": os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-large"),
        }
    
    def _load_cached_index(self):
        """加载缓存的嵌入索引"""
        cache_path = Path(self.config.get("mixed_embedding_cache", "data/cache/hypergraph_mixed_embedding.pkl"))
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                self.embedding_index = pickle.load(f)
    
    def _http_post_json(self, url: str, payload: Dict[str, Any], api_key: str) -> Dict[str, Any]:
        """发送HTTP POST请求"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()
    
    def _embedding_request(self, texts: List[str]) -> np.ndarray:
        """请求嵌入向量"""
        if not texts:
            raise ValueError("texts 不能为空")
        
        url = self.client["base_url"] + "/embeddings"
        payload = {"model": self.client["model"], "input": texts}
        
        resp = self._http_post_json(url, payload, self.client["api_key"])
        items = resp.get("data") or []
        
        embs: List[np.ndarray] = []
        for item in items:
            emb = item.get("embedding")
            if emb is None:
                continue
            embs.append(np.asarray(emb, dtype=np.float32))
        
        if not embs:
            raise ValueError("嵌入结果为空")
        
        return np.stack(embs, axis=0)
    
    def _edge_success_prob(self, edge: Dict[str, Any]) -> float:
        """从超边中提取成功概率"""
        meta = edge.get("eff_metadata", {}) or {}
        p = meta.get("success_probability")
        if isinstance(p, (int, float)) and p > 0:
            return float(p)
        
        sup = edge.get("support_count", 0) or 0
        suc = edge.get("success_count", 0) or 0
        if sup > 0 and suc >= 0:
            return suc / sup
        
        return math.nan
    
    def build_mixed_channel_embedding_index(
        self,
        hypergraph: Dict[str, Any],
        min_support: int = 5,
        channel_weights: Optional[List[float]] = None,
    ) -> Optional[Dict[str, Any]]:
        """构建信息信道混合嵌入索引。
        
        将超边的不同信息通道分别嵌入，然后按权重融合：
        - pre_nodes 通道（前置条件）
        - scene_atoms 通道（场景上下文）
        - eff_nodes 通道（效果节点）
        - operator 通道（操作符）
        
        Args:
            hypergraph: 超图数据
            min_support: 最小支持度
            channel_weights: [w_pre, w_scene, w_eff, w_op]，默认 [0.3, 0.4, 0.2, 0.1]
        
        Returns:
            包含混合嵌入矩阵和元数据的索引
        """
        
        if channel_weights is None:
            channel_weights = [0.3, 0.4, 0.2, 0.1]  # 默认权重
        
        edges = list(hypergraph.get("hyperedges", []) or [])
        
        # 为每个信道准备文本
        pre_texts: List[str] = []
        scene_texts: List[str] = []
        eff_texts: List[str] = []
        op_texts: List[str] = []
        metas: List[Dict[str, Any]] = []
        
        for he in edges:
            sup = he.get("support_count", 0) or 0
            if sup < min_support:
                continue
            
            # 提取各通道的文本
            pre_nodes = he.get("pre_nodes") or []
            scene_atoms = he.get("scene_atoms") or []
            eff_nodes = he.get("eff_nodes") or []
            operator = he.get("operator") or ""
            
            pre_text = " ".join(sorted([str(a) for a in pre_nodes if isinstance(a, str)]))
            scene_text = " ".join(sorted([str(a) for a in scene_atoms if isinstance(a, str)]))
            eff_text = " ".join(sorted([str(a) for a in eff_nodes if isinstance(a, str)]))
            op_text = str(operator)
            
            # 如果某个通道为空，用占位符
            pre_texts.append(pre_text if pre_text else "none")
            scene_texts.append(scene_text if scene_text else "none")
            eff_texts.append(eff_text if eff_text else "none")
            op_texts.append(op_text if op_text else "none")
            
            metas.append({"id": he.get("id"), "operator": operator, "edge": he})
        
        if not metas:
            return None
        
        print(f"\n[混合嵌入] 为 {len(metas)} 条超边构建 4 通道嵌入...")
        
        # 分别调用嵌入 API
        print("  通道 1/4: pre_nodes...")
        pre_embs = self._embedding_request(pre_texts)
        
        print("  通道 2/4: scene_atoms...")
        scene_embs = self._embedding_request(scene_texts)
        
        print("  通道 3/4: eff_nodes...")
        eff_embs = self._embedding_request(eff_texts)
        
        print("  通道 4/4: operator...")
        op_embs = self._embedding_request(op_texts)
        
        # 按权重融合
        print(f"  融合 4 个通道 (权重: {channel_weights})...")
        mixed_embs = (
            channel_weights[0] * pre_embs +
            channel_weights[1] * scene_embs +
            channel_weights[2] * eff_embs +
            channel_weights[3] * op_embs
        )
        
        # L2 归一化
        norms = np.linalg.norm(mixed_embs, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        mixed_embs_norm = mixed_embs / norms
        
        print(f"[混合嵌入完成] 维度={mixed_embs_norm.shape[1]}, 权重={channel_weights}")
        
        result = {
            "embeddings": mixed_embs_norm,
            "meta": metas,
            "client": self.client,
            "channel_weights": channel_weights,
        }
        
        # 保存到缓存
        cache_path = Path(self.config.get("mixed_embedding_cache", "data/cache/hypergraph_mixed_embedding.pkl"))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump(result, f)
        print(f"  [缓存已保存] {cache_path.name}")
        
        self.embedding_index = result
        return result
    
    def match_context(
        self,
        context_text: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """匹配上下文，返回最相关的超边"""
        if not self.embedding_index or not context_text:
            return []
        
        # 获取上下文嵌入
        ctx_emb = self._embedding_request([context_text])[0:1]
        ctx_emb = ctx_emb / (np.linalg.norm(ctx_emb, axis=1, keepdims=True) + 1e-8)
        
        # 计算相似度
        all_embs = self.embedding_index["embeddings"]
        sims = np.dot(all_embs, ctx_emb.T).flatten()
        
        # 获取top-k
        top_idx = np.argsort(-sims)[:top_k]
        
        results = []
        for idx in top_idx:
            if sims[idx] > 0.1:  # 相似度阈值
                result = dict(self.embedding_index["meta"][idx])
                result["similarity"] = float(sims[idx])
                results.append(result)
        
        return results
