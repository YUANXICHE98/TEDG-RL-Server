import numpy as np
from typing import List, Dict, Any

def calculate_proper_confidence(
    pre_nodes: List[str],
    scene_atoms: List[str],
    hyperedge: Dict[str, Any],
    steps_since_match: int = 0,
    decay_rate: float = 0.01
) -> float:
    """
    计算正确的置信度（基于超图匹配，而非HP）
    
    Args:
        pre_nodes: 当前前置条件
        scene_atoms: 当前场景原子
        hyperedge: 匹配的超边
        steps_since_match: 自上次强匹配以来的步数
        decay_rate: 衰减率
    
    Returns:
        置信度 [0, 1]
    """
    # 1. 计算前置条件匹配度
    edge_pre_conditions = set(hyperedge.get('pre_conditions', []))
    pre_match = len(set(pre_nodes) & edge_pre_conditions)
    pre_score = pre_match / max(len(edge_pre_conditions), 1)
    
    # 2. 计算场景匹配度
    edge_scene_context = set(hyperedge.get('scene_context', []))
    scene_match = len(set(scene_atoms) & edge_scene_context)
    scene_score = scene_match / max(len(edge_scene_context), 1)
    
    # 3. 综合匹配分数
    match_score = 0.6 * pre_score + 0.4 * scene_score
    
    # 4. 时间衰减
    decay_factor = np.exp(-decay_rate * steps_since_match)
    
    # 5. 最终置信度
    confidence = match_score * decay_factor
    
    # 确保在[0.3, 0.9]范围内（避免极端值）
    confidence = np.clip(confidence, 0.3, 0.9)
    
    return confidence


def select_best_matching_hyperedge(
    pre_nodes: List[str],
    scene_atoms: List[str],
    hypergraph: Dict[str, Any]
) -> (Dict[str, Any], float):
    """
    选择最佳匹配的超边（而非随机选择）
    
    Returns:
        (最佳超边, 匹配置信度)
    """
    best_edge = None
    best_confidence = 0.0
    
    for edge in hypergraph['hyperedges']:
        conf = calculate_proper_confidence(pre_nodes, scene_atoms, edge)
        if conf > best_confidence:
            best_confidence = conf
            best_edge = edge
    
    # 如果没有匹配，返回随机超边但给予低置信度
    if best_edge is None:
        best_edge = hypergraph['hyperedges'][0]
        best_confidence = 0.3
    
    return best_edge, best_confidence
