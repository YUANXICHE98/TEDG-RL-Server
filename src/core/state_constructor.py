"""状态构造器 - 组装115维state向量"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict


class StateConstructor:
    """构造115维state = [belief(50) + q_pre(15) + q_scene(15) + q_effect(8) + q_rule(10) + confidence(1) + goal(16)]"""
    
    def __init__(self, hypergraph_path: str):
        self.hypergraph = self._load_hypergraph(hypergraph_path)
        self.pre_nodes_vocab = self._build_vocab('pre_nodes')
        self.scene_atoms_vocab = self._build_vocab('scene_atoms')
        self.eff_nodes_vocab = self._build_vocab('eff_nodes')
        
        print(f"[StateConstructor] 加载超图: {len(self.hypergraph['hyperedges'])} 条超边")
        print(f"  - pre_nodes词汇: {len(self.pre_nodes_vocab)}")
        print(f"  - scene_atoms词汇: {len(self.scene_atoms_vocab)}")
        print(f"  - eff_nodes词汇: {len(self.eff_nodes_vocab)}")
    
    def _load_hypergraph(self, path: str) -> Dict[str, Any]:
        """加载超图JSON"""
        with open(path, 'r') as f:
            return json.load(f)
    
    def _build_vocab(self, key: str) -> Dict[str, int]:
        """构建词汇表"""
        vocab = {}
        for edge in self.hypergraph['hyperedges']:
            nodes = edge.get(key, [])
            for node in nodes:
                if node not in vocab:
                    vocab[node] = len(vocab)
        return vocab
    
    def construct_state(
        self,
        belief_vector: np.ndarray,  # (50,)
        pre_nodes: List[str],  # 当前前置条件
        scene_atoms: List[str],  # 当前场景原子
        eff_metadata: Dict[str, Any],  # 效果元数据
        conditional_effects: List[Dict[str, str]],  # 条件效果
        confidence: float,  # 置信度 [0, 1]
        goal_embedding: np.ndarray,  # (16,)
    ) -> np.ndarray:
        """
        构造115维state向量
        
        Args:
            belief_vector: (50,) 游戏状态向量
            pre_nodes: 当前活跃的前置条件节点列表
            scene_atoms: 当前活跃的场景原子列表
            eff_metadata: 效果元数据 {success_probability, safety_score, ...}
            conditional_effects: 条件效果列表
            confidence: 超图匹配置信度
            goal_embedding: (16,) 目标嵌入向量
        
        Returns:
            (115,) 状态向量
        """
        # 1. belief_vector (50,)
        assert belief_vector.shape == (50,), f"belief_vector shape {belief_vector.shape} != (50,)"
        
        # 2. q_pre (15,) - 前置条件嵌入
        q_pre = self._embed_pre_nodes(pre_nodes)
        
        # 3. q_scene (15,) - 场景原子嵌入
        q_scene = self._embed_scene_atoms(scene_atoms)
        
        # 4. q_effect (8,) - 效果/风险嵌入
        q_effect = self._embed_effect_metadata(eff_metadata)
        
        # 5. q_rule (10,) - 规则/条件效果嵌入
        q_rule = self._embed_conditional_effects(conditional_effects)
        
        # 6. confidence (1,)
        conf_vec = np.array([confidence], dtype=np.float32)
        
        # 7. goal_embedding (16,)
        assert goal_embedding.shape == (16,), f"goal_embedding shape {goal_embedding.shape} != (16,)"
        
        # 拼接: 50 + 15 + 15 + 8 + 10 + 1 + 16 = 115
        state = np.concatenate([
            belief_vector,
            q_pre,
            q_scene,
            q_effect,
            q_rule,
            conf_vec,
            goal_embedding
        ]).astype(np.float32)
        
        assert state.shape == (115,), f"state shape {state.shape} != (115,)"
        return state
    
    def _embed_pre_nodes(self, pre_nodes: List[str]) -> np.ndarray:
        """
        嵌入前置条件节点 -> (15,)
        
        使用简单的方法：
        - 前10维: 前置条件的one-hot编码的加权和
        - 后5维: 统计特征 (数量、多样性、覆盖率等)
        """
        embedding = np.zeros(15, dtype=np.float32)
        
        if not pre_nodes:
            return embedding
        
        # 前10维: 对pre_nodes进行哈希编码
        for i, node in enumerate(pre_nodes[:10]):
            if node in self.pre_nodes_vocab:
                idx = self.pre_nodes_vocab[node] % 10
                embedding[idx] += 1.0 / (i + 1)  # 衰减权重
        
        # 归一化前10维
        norm = np.linalg.norm(embedding[:10])
        if norm > 0:
            embedding[:10] /= norm
        
        # 后5维: 统计特征
        embedding[10] = len(pre_nodes) / 30.0  # 节点数量 (归一化到30)
        embedding[11] = len(set(pre_nodes)) / len(pre_nodes) if pre_nodes else 0  # 多样性
        embedding[12] = sum(1 for n in pre_nodes if 'hp' in n.lower()) / len(pre_nodes) if pre_nodes else 0
        embedding[13] = sum(1 for n in pre_nodes if 'hunger' in n.lower()) / len(pre_nodes) if pre_nodes else 0
        embedding[14] = sum(1 for n in pre_nodes if 'power' in n.lower()) / len(pre_nodes) if pre_nodes else 0
        
        return embedding
    
    def _embed_scene_atoms(self, scene_atoms: List[str]) -> np.ndarray:
        """
        嵌入场景原子 -> (15,)
        
        使用简单的方法：
        - 前10维: 场景原子的one-hot编码的加权和
        - 后5维: 统计特征 (深度、位置、危险度等)
        """
        embedding = np.zeros(15, dtype=np.float32)
        
        if not scene_atoms:
            return embedding
        
        # 前10维: 对scene_atoms进行哈希编码
        for i, atom in enumerate(scene_atoms[:10]):
            if atom in self.scene_atoms_vocab:
                idx = self.scene_atoms_vocab[atom] % 10
                embedding[idx] += 1.0 / (i + 1)
        
        # 归一化前10维
        norm = np.linalg.norm(embedding[:10])
        if norm > 0:
            embedding[:10] /= norm
        
        # 后5维: 统计特征
        embedding[10] = len(scene_atoms) / 40.0  # 原子数量 (归一化到40)
        embedding[11] = sum(1 for a in scene_atoms if 'dlvl' in a.lower()) / len(scene_atoms) if scene_atoms else 0
        embedding[12] = sum(1 for a in scene_atoms if 'shop' in a.lower() or 'altar' in a.lower()) / len(scene_atoms) if scene_atoms else 0
        embedding[13] = sum(1 for a in scene_atoms if 'monster' in a.lower()) / len(scene_atoms) if scene_atoms else 0
        embedding[14] = sum(1 for a in scene_atoms if 'ac_poor' in a.lower()) / len(scene_atoms) if scene_atoms else 0
        
        return embedding
    
    def _embed_effect_metadata(self, eff_metadata: Dict[str, Any]) -> np.ndarray:
        """
        嵌入效果元数据 -> (8,)
        
        包含: success_probability, safety_score, avg_score_gain, failure_mode_counts等
        """
        embedding = np.zeros(8, dtype=np.float32)
        
        if not eff_metadata:
            return embedding
        
        # 直接使用元数据中的数值特征
        embedding[0] = float(eff_metadata.get('success_probability', 0.5))
        embedding[1] = float(eff_metadata.get('safety_score', 0.5))
        embedding[2] = float(eff_metadata.get('applicability_confidence', 0.5))
        embedding[3] = float(eff_metadata.get('avg_score_gain', 0.0))
        
        # 失败模式统计
        failure_modes = eff_metadata.get('failure_modes', {})
        if failure_modes:
            total_failures = sum(failure_modes.values())
            embedding[4] = min(total_failures / 1000.0, 1.0)  # 失败率
            embedding[5] = float(failure_modes.get('precondition_violation', 0)) / max(total_failures, 1)
            embedding[6] = float(failure_modes.get('bad_aim', 0)) / max(total_failures, 1)
            embedding[7] = float(failure_modes.get('need_unlock', 0)) / max(total_failures, 1)
        
        return np.clip(embedding, 0, 1).astype(np.float32)
    
    def _embed_conditional_effects(self, conditional_effects: List[Dict[str, str]]) -> np.ndarray:
        """
        嵌入条件效果 -> (10,)
        
        编码隐藏的规则和陷阱
        """
        embedding = np.zeros(10, dtype=np.float32)
        
        if not conditional_effects:
            return embedding
        
        # 统计条件效果的类型
        effect_types = defaultdict(int)
        for ce in conditional_effects:
            effect = ce.get('effect', '')
            if 'blessed' in effect.lower():
                effect_types['blessed'] += 1
            elif 'cursed' in effect.lower():
                effect_types['cursed'] += 1
            elif 'poison' in effect.lower():
                effect_types['poison'] += 1
            elif 'confused' in effect.lower():
                effect_types['confused'] += 1
            elif 'blind' in effect.lower():
                effect_types['blind'] += 1
            else:
                effect_types['other'] += 1
        
        # 编码到向量
        embedding[0] = len(conditional_effects) / 10.0  # 条件效果数量
        embedding[1] = effect_types.get('blessed', 0) / len(conditional_effects) if conditional_effects else 0
        embedding[2] = effect_types.get('cursed', 0) / len(conditional_effects) if conditional_effects else 0
        embedding[3] = effect_types.get('poison', 0) / len(conditional_effects) if conditional_effects else 0
        embedding[4] = effect_types.get('confused', 0) / len(conditional_effects) if conditional_effects else 0
        embedding[5] = effect_types.get('blind', 0) / len(conditional_effects) if conditional_effects else 0
        embedding[6] = effect_types.get('other', 0) / len(conditional_effects) if conditional_effects else 0
        
        # 风险评估
        high_risk_count = effect_types.get('cursed', 0) + effect_types.get('poison', 0)
        embedding[7] = min(high_risk_count / len(conditional_effects), 1.0) if conditional_effects else 0
        
        # 未知规则的不确定性
        embedding[8] = 0.5 if len(conditional_effects) > 0 else 0.0
        embedding[9] = len(conditional_effects) / 10.0
        
        return np.clip(embedding, 0, 1).astype(np.float32)
    
    def get_applicable_actions(self, state: np.ndarray) -> List[Dict[str, Any]]:
        """
        根据state获取可应用的超边 (动作)
        
        简化版本: 返回所有超边，由action_masking在RL中过滤
        """
        return self.hypergraph['hyperedges']
