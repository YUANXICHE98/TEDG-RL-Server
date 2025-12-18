"""超图加载和匹配模块"""

import os
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import gymnasium as gym
import requests

# 加载 .env 文件
def _load_env():
    env_paths = [
        Path(__file__).parent.parent / ".env",  # src/.env
        Path(__file__).parent.parent.parent / ".env",  # project root/.env
    ]
    for env_path in env_paths:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        value = value.strip().strip('"').strip("'")
                        if key and value:
                            os.environ.setdefault(key, value)
            break

_load_env()


class HypergraphLoader:
    """超图加载器，负责加载超图数据和嵌入索引"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.hypergraph = None
        self.embedding_index = None
        self._load_data()
    
    def _load_data(self):
        """加载超图和嵌入数据"""
        # 加载超图
        hg_path = Path(self.config["file"])
        with open(hg_path, 'r') as f:
            import json
            self.hypergraph = json.load(f)
        
        # 加载嵌入索引
        cache_path = Path(self.config["embedding_cache"])
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                self.embedding_index = pickle.load(f)
    
    def get_applicable_actions(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """根据当前状态获取可应用的动作"""
        # TODO: 实现动作匹配逻辑
        # 这里简化返回所有动作
        edges = self.hypergraph.get("hyperedges", [])
        return edges[:10]  # 返回前10个作为示例


def _load_hypergraph() -> Dict[str, Any]:
    """加载超图数据"""
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    hypergraph_path = PROJECT_ROOT / "data/hypergraph/hypergraph_complete_real.json"
    
    with open(hypergraph_path, 'r') as f:
        import json
        hg = json.load(f)
    
    return hg


def _build_operator_to_action_map(env) -> Dict[str, int]:
    """构建operator到NLE动作索引的映射"""
    try:
        actions = env.unwrapped.actions
    except AttributeError:
        return {}
    
    ascii_to_idx = {}
    for idx in range(len(actions)):
        try:
            action_item = actions[idx]
            if hasattr(action_item, 'value'):
                ascii_code = action_item.value
                ascii_to_idx[ascii_code] = idx
            elif isinstance(action_item, bytes):
                if len(action_item) > 0:
                    ascii_code = action_item[0] if isinstance(action_item[0], int) else ord(action_item[0])
                    ascii_to_idx[ascii_code] = idx
            elif isinstance(action_item, int):
                ascii_to_idx[action_item] = idx
            elif isinstance(action_item, tuple) and len(action_item) > 0:
                ascii_code = action_item[0]
                ascii_to_idx[ascii_code] = idx
        except Exception:
            continue
    
    # NetHack动作映射
    operator_to_action = {
        # 移动类
        "move": 107,  # k
        "move_j": 106,  # j
        "move_h": 104,  # h
        "move_k": 107,  # k
        "move_l": 108,  # l
        "move_y": 121,  # y
        "move_u": 117,  # u
        "move_b": 98,   # b
        "move_n": 110,  # n
        # 探索类
        "search": 115,  # s
        "look": 58,     # :
        # 物品类
        "pickup": 44,   # ,
        "drop": 100,    # d
        "wield": 119,   # w
        "wear": 87,     # W
        "takeoff": 84,  # T
        "eat": 101,     # e
        "quaff": 113,   # q
        "read": 114,    # r
        "zap": 122,     # z
        "apply": 97,    # a
        "throw": 116,   # t
        # 其他
        "wait": 46,     # .
        "pray": 122,    # z (暂用)
        "chat": 99,     # c
    }
    
    result = {k: v for k, v in operator_to_action.items() if v is not None}
    return result


def _create_embedding_client() -> Dict[str, str]:
    """创建嵌入API客户端"""
    import os
    return {
        "api_key": os.getenv("EMBEDDING_API_KEY", ""),
        "base_url": os.getenv("EMBEDDING_BASE_URL", "https://api.openai-hk.com/v1"),
        "model": os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-large"),
    }


def _http_post_json(url: str, payload: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """发送HTTP POST请求"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _embedding_request(client: Dict[str, str], texts: List[str], max_retries: int = 5) -> np.ndarray:
    """请求嵌入向量（带重试机制）"""
    import time
    
    if not texts:
        raise ValueError("texts 不能为空")
    
    url = client["base_url"] + "/embeddings"
    payload = {"model": client["model"], "input": texts}
    
    last_error = None
    for attempt in range(max_retries):
        try:
            resp = _http_post_json(url, payload, client["api_key"])
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
        except Exception as e:
            last_error = e
            wait_time = 2 ** attempt  # 指数退避: 1, 2, 4, 8, 16 秒
            print(f"  [嵌入API] 第{attempt+1}次失败: {e}, {wait_time}秒后重试...")
            time.sleep(wait_time)
    
    raise RuntimeError(f"嵌入API {max_retries}次重试后仍失败: {last_error}")


def _build_hypergraph_embedding_index(
    min_support: int = 5,
) -> Optional[Dict[str, Any]]:
    """构建超图嵌入索引"""
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    cache_dir = PROJECT_ROOT / "data/cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"hypergraph_embedding_index_minsup{min_support}.pkl"
    
    # 尝试从缓存加载
    if cache_file.exists():
        try:
            with cache_file.open("rb") as f:
                cached = pickle.load(f)
            print(f"  [缓存命中] 从 {cache_file.name} 加载已有嵌入索引")
            return cached
        except Exception as e:
            print(f"  [缓存加载失败] {e}, 重新构建索引")
    
    # 构建新索引
    hg = _load_hypergraph()
    edges = list(hg.get("hyperedges", []) or [])
    client = _create_embedding_client()
    
    texts: List[str] = []
    metas: List[Dict[str, Any]] = []
    
    for he in edges:
        sup = he.get("support_count", 0) or 0
        if sup < min_support:
            continue
        
        parts: List[str] = []
        op = he.get("operator")
        if op:
            parts.append(str(op))
        
        for key in ("pre_nodes", "scene_atoms", "eff_nodes"):
            vals = he.get(key) or []
            for a in vals:
                if isinstance(a, str):
                    parts.append(a)
        
        if not parts:
            continue
        
        text = " ".join(sorted(set(parts)))
        texts.append(text)
        metas.append({"id": he.get("id"), "operator": he.get("operator"), "edge": he})
    
    if not texts:
        return None
    
    # 批量获取嵌入
    emb_mat = _embedding_request(client, texts)
    norms = np.linalg.norm(emb_mat, axis=1, keepdims=True)
    emb_mat = emb_mat / (norms + 1e-8)
    
    index = {
        "embeddings": emb_mat,
        "meta": metas,
        "min_support": min_support,
    }
    
    # 保存缓存
    with cache_file.open("wb") as f:
        pickle.dump(index, f)
    
    return index


def _match_context(
    embedding_index: Dict[str, Any],
    context_text: str,
    client: Dict[str, str],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """匹配上下文，返回最相关的超边"""
    if not embedding_index or not context_text:
        return []
    
    # 获取上下文嵌入
    ctx_emb = _embedding_request(client, [context_text])[0:1]
    ctx_emb = ctx_emb / (np.linalg.norm(ctx_emb, axis=1, keepdims=True) + 1e-8)
    
    # 计算相似度
    all_embs = embedding_index["embeddings"]
    sims = np.dot(all_embs, ctx_emb.T).flatten()
    
    # 获取top-k
    top_idx = np.argsort(-sims)[:top_k]
    
    results = []
    for idx in top_idx:
        if sims[idx] > 0.1:  # 相似度阈值
            result = dict(embedding_index["meta"][idx])
            result["similarity"] = float(sims[idx])
            results.append(result)
    
    return results


class EmbeddingMatcher:
    """基于嵌入的超图匹配器，支持 atom 嵌入缓存"""
    
    def __init__(self, min_support: int = 5):
        self.client = _create_embedding_client()
        self.embedding_index = _build_hypergraph_embedding_index(min_support)
        self.atom_cache: Dict[str, np.ndarray] = {}  # atom -> embedding 缓存
        self._load_atom_cache()
        print(f"[EmbeddingMatcher] 初始化: {len(self.embedding_index['embeddings'])} 条超边嵌入, {len(self.atom_cache)} 个 atom 缓存")
    
    def _get_cache_path(self) -> Path:
        PROJECT_ROOT = Path(__file__).parent.parent.parent
        return PROJECT_ROOT / "data/cache/atom_embedding_cache.pkl"
    
    def _load_atom_cache(self):
        """加载 atom 嵌入缓存"""
        cache_path = self._get_cache_path()
        if cache_path.exists():
            try:
                with cache_path.open("rb") as f:
                    self.atom_cache = pickle.load(f)
                print(f"  [缓存命中] 加载 {len(self.atom_cache)} 个 atom 嵌入")
            except Exception as e:
                print(f"  [缓存加载失败] {e}")
                self.atom_cache = {}
    
    def _save_atom_cache(self):
        """保存 atom 嵌入缓存"""
        cache_path = self._get_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump(self.atom_cache, f)
    
    def _get_atom_embedding(self, atom: str) -> np.ndarray:
        """获取单个 atom 的嵌入，优先从缓存读取"""
        if atom in self.atom_cache:
            return self.atom_cache[atom]
        
        # 现场计算
        try:
            emb = _embedding_request(self.client, [atom])[0]
            emb = emb / (np.linalg.norm(emb) + 1e-8)  # L2 归一化
            self.atom_cache[atom] = emb
            return emb
        except Exception as e:
            print(f"  [警告] atom '{atom}' 嵌入失败: {e}")
            # 返回零向量
            dim = int(os.getenv("EMBEDDING_DIM", "1536"))
            return np.zeros(dim, dtype=np.float32)
    
    def _get_atoms_embedding(self, atoms: List[str]) -> np.ndarray:
        """获取一组 atoms 的聚合嵌入（均值池化）"""
        dim = int(os.getenv("EMBEDDING_DIM", "1536"))
        if not atoms:
            return np.zeros(dim, dtype=np.float32)
        
        # 分离已缓存和未缓存的 atoms
        cached = [a for a in atoms if a in self.atom_cache]
        uncached = [a for a in atoms if a not in self.atom_cache]
        
        # 批量获取未缓存的嵌入
        if uncached:
            try:
                embs = _embedding_request(self.client, uncached)
                for i, atom in enumerate(uncached):
                    emb = embs[i] / (np.linalg.norm(embs[i]) + 1e-8)
                    self.atom_cache[atom] = emb
            except Exception as e:
                print(f"  [警告] 批量嵌入失败: {e}")
        
        # 聚合所有嵌入
        all_embs = [self.atom_cache.get(a, np.zeros(dim, dtype=np.float32)) for a in atoms]
        if not all_embs:
            return np.zeros(dim, dtype=np.float32)
        
        # 均值池化 + L2 归一化
        mean_emb = np.mean(all_embs, axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb = mean_emb / norm
        
        return mean_emb
    
    def match(
        self,
        pre_nodes: List[str],
        scene_atoms: List[str],
        effect_atoms: List[str],
        rule_atoms: List[str],
        top_k: int = 8,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        基于嵌入匹配超边
        
        Returns:
            confidence: 最高相似度作为置信度
            topk_edges: Top-K 匹配的超边
        """
        if self.embedding_index is None:
            return 0.0, []
        
        # 将所有 atoms 合并为查询文本
        all_atoms = pre_nodes + scene_atoms + effect_atoms + rule_atoms
        if not all_atoms:
            return 0.0, []
        
        # 获取查询嵌入
        query_emb = self._get_atoms_embedding(all_atoms)
        query_emb = query_emb.reshape(1, -1)  # (1, dim)
        
        # 计算与所有超边的余弦相似度
        edge_embs = self.embedding_index["embeddings"]  # (n_edges, dim)
        sims = np.dot(edge_embs, query_emb.T).flatten()  # (n_edges,)
        
        # 获取 Top-K
        top_idx = np.argsort(-sims)[:top_k]
        
        topk_edges = []
        for idx in top_idx:
            meta = self.embedding_index["meta"][idx]
            topk_edges.append({
                "edge": meta["edge"],
                "similarity": float(sims[idx]),
                "operator": meta["operator"],
            })
        
        # confidence = 最高相似度
        confidence = float(sims[top_idx[0]]) if len(top_idx) > 0 else 0.0
        
        return confidence, topk_edges
    
    def save_cache(self):
        """手动保存缓存（训练结束时调用）"""
        self._save_atom_cache()
        print(f"[EmbeddingMatcher] 保存 {len(self.atom_cache)} 个 atom 嵌入到缓存")
