"""动作掩蔽 - 超图约束的动作空间"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple


# NLE NetHackScore-v0 动作空间映射 (23个动作)
# 将超图 operator 映射到 NLE action_id
OPERATOR_TO_ACTION_IDS: Dict[str, List[int]] = {
    # 移动类 (action 1-8: N/E/S/W/NE/SE/SW/NW)
    "move": [1, 2, 3, 4, 5, 6, 7, 8],
    "move_no_fight": [1, 2, 3, 4, 5, 6, 7, 8],
    "move_no_pickup": [1, 2, 3, 4, 5, 6, 7, 8],
    "move_until_near": [1, 2, 3, 4, 5, 6, 7, 8],
    "flee": [1, 2, 3, 4, 5, 6, 7, 8],  # 逃跑也是移动
    
    # 跑步类 (action 9-16: 大写方向)
    # 在安全情况下可用
    
    # 楼梯类
    "go_up": [17],      # UP (<)
    "climb": [17],      # 也是上楼
    "go_down": [18],    # DOWN (>)
    
    # 等待
    "wait": [19],
    
    # 踢击
    "kick": [20],
    "force_lock": [20],  # 踢开门
    "open_door": [20],   # 可以用踢
    "close_door": [20],  # 也可以用踢
    
    # 吃东西
    "eat": [21],
    
    # 搜索
    "search": [22],
    "look": [22],        # 查看也用search
    "identify_trap": [22],
    
    # 攻击类 - 映射到移动（NetHack中移动到怪物位置即攻击）
    "attack": [1, 2, 3, 4, 5, 6, 7, 8],
    "attack_melee": [1, 2, 3, 4, 5, 6, 7, 8],
    "melee_attack": [1, 2, 3, 4, 5, 6, 7, 8],
    
    # 特殊场景策略 - 映射到基础动作
    "minetown": [1, 2, 3, 4, 5, 6, 7, 8, 17, 18],  # 探索
    "oracle": [1, 2, 3, 4, 5, 6, 7, 8, 17, 18],
    "sokoban": [1, 2, 3, 4, 5, 6, 7, 8, 17, 18],
    "medusa": [1, 2, 3, 4, 5, 6, 7, 8],
    "vlad": [1, 2, 3, 4, 5, 6, 7, 8],
    
    # 祭坛相关
    "altar": [19],       # 在祭坛上等待
    "pray": [19],        # 祈祷需要等待
    "sacrifice": [19],   # 献祭
    "offer": [19],
    
    # 其他操作 - 在 NetHackScore-v0 中不可用，映射到空
    # 但为了不完全封锁，映射到 wait
    "inventory": [19],
    "count_gold": [19],
    "elbereth": [19],    # 写Elbereth需要等待
    "save_game": [19],
}

# 默认允许的动作（当没有匹配的operator时）
DEFAULT_ALLOWED_ACTIONS = [1, 2, 3, 4, 5, 6, 7, 8, 19, 22]  # 移动 + 等待 + 搜索


class ActionMasker:
    """动作掩蔽器 - 根据超图约束mask不可行动作"""
    
    def __init__(self, hypergraph: Dict[str, Any], num_actions: int = 23):
        self.hypergraph = hypergraph
        self.num_actions = num_actions
        self.edges = hypergraph.get('hyperedges', [])
        
        # 构建operator到edge的映射
        self.operator_to_edges = self._build_operator_map()
        
        # operator到action_id的映射
        self.op_to_actions = OPERATOR_TO_ACTION_IDS
        
        print(f"[ActionMasker] 初始化: {len(self.edges)} 条超边, {len(self.operator_to_edges)} 个操作符, 映射覆盖 {len(self.op_to_actions)} 个")
    
    def _build_operator_map(self) -> Dict[str, List[Dict[str, Any]]]:
        """构建operator到超边的映射"""
        op_map = {}
        for edge in self.edges:
            op = edge.get('operator', 'unknown')
            if op not in op_map:
                op_map[op] = []
            op_map[op].append(edge)
        return op_map
    
    def get_applicable_edges(
        self,
        pre_nodes: List[str],
        scene_atoms: List[str],
        confidence: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        获取可应用的超边
        
        简化版本: 返回满足以下条件的超边:
        1. 前置条件至少有50%匹配
        2. 场景原子至少有50%匹配
        3. 置信度 >= 0.3
        
        Args:
            pre_nodes: 当前前置条件列表
            scene_atoms: 当前场景原子列表
            confidence: 超图匹配置信度
        
        Returns:
            可应用的超边列表
        """
        if confidence < 0.3:
            return []  # 置信度太低，不应用任何超边约束
        
        applicable = []
        pre_set = set(pre_nodes)
        scene_set = set(scene_atoms)
        
        for edge in self.edges:
            edge_pre = set(edge.get('pre_nodes', []))
            edge_scene = set(edge.get('scene_atoms', []))
            
            # 计算匹配度
            pre_match = len(pre_set & edge_pre) / max(len(edge_pre), 1)
            scene_match = len(scene_set & edge_scene) / max(len(edge_scene), 1)
            
            # 阈值: 至少50%匹配
            if pre_match >= 0.5 and scene_match >= 0.5:
                applicable.append(edge)
        
        return applicable
    
    def get_action_mask(
        self,
        pre_nodes: List[str],
        scene_atoms: List[str],
        confidence: float = 0.5,
    ) -> np.ndarray:
        """
        获取动作掩蔽向量
        
        Args:
            pre_nodes: 当前前置条件列表
            scene_atoms: 当前场景原子列表
            confidence: 超图匹配置信度
        
        Returns:
            mask: (num_actions,) 布尔数组，True表示可行，False表示不可行
        """
        mask = np.ones(self.num_actions, dtype=bool)
        
        if confidence < 0.3:
            return mask  # 置信度太低，允许所有动作
        
        # 获取可应用的超边
        applicable = self.get_applicable_edges(pre_nodes, scene_atoms, confidence)
        
        if not applicable:
            return mask  # 没有可应用的超边，允许所有动作
        
        # 收集可应用的operator
        applicable_ops = set(edge.get('operator', '') for edge in applicable)
        
        # 根据 operator 映射到 action_id
        allowed_actions = set()
        for op in applicable_ops:
            if op in self.op_to_actions:
                allowed_actions.update(self.op_to_actions[op])
        
        # 如果没有匹配的操作，使用默认允许动作
        if not allowed_actions:
            allowed_actions = set(DEFAULT_ALLOWED_ACTIONS)
        
        # 构建 mask: 只允许 allowed_actions 中的动作
        mask = np.zeros(self.num_actions, dtype=bool)
        for aid in allowed_actions:
            if 0 <= aid < self.num_actions:
                mask[aid] = True
        
        # 始终允许 action 0 (MORE/确认)
        mask[0] = True
        
        return mask
    
    def apply_mask_to_logits(
        self,
        logits: torch.Tensor,
        pre_nodes: List[str],
        scene_atoms: List[str],
        confidence: float = 0.5,
    ) -> torch.Tensor:
        """
        将掩蔽应用到logits
        
        Args:
            logits: (batch_size, num_actions) 或 (num_actions,)
            pre_nodes: 当前前置条件列表
            scene_atoms: 当前场景原子列表
            confidence: 超图匹配置信度
        
        Returns:
            masked_logits: 应用掩蔽后的logits
        """
        mask = self.get_action_mask(pre_nodes, scene_atoms, confidence)
        mask_tensor = torch.FloatTensor(mask).to(logits.device)
        
        # 将不可行动作的logits设为-inf
        masked_logits = logits.clone()
        if logits.dim() == 1:
            masked_logits[~mask] = float('-inf')
        else:
            masked_logits[:, ~mask] = float('-inf')
        
        return masked_logits
    
    def check_precondition_violation(
        self,
        action_edge: Dict[str, Any],
        current_pre_nodes: List[str],
    ) -> bool:
        """
        检查是否违反前置条件
        
        Args:
            action_edge: 动作对应的超边
            current_pre_nodes: 当前前置条件
        
        Returns:
            True如果违反前置条件，False否则
        """
        required_pre = set(action_edge.get('pre_nodes', []))
        current_pre = set(current_pre_nodes)
        
        # 检查是否有必需的前置条件缺失
        missing = required_pre - current_pre
        
        # 如果缺失的前置条件超过50%，则认为违反
        violation_ratio = len(missing) / max(len(required_pre), 1)
        return violation_ratio > 0.5
