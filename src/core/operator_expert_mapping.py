"""
Operator到Expert的映射
用于Manager内层约束：将超图的Operator节点映射到对应的Expert

基于76个Operators的语义分析：
- Expert 0 (Survival): 生存相关 - 18个operators
- Expert 1 (Combat): 战斗相关 - 15个operators  
- Expert 2 (Exploration): 探索相关 - 28个operators
- Expert 3 (General): 通用/兜底 - 15个operators
"""

# Operator名称 -> Expert索引的映射
OPERATOR_TO_EXPERT = {
    # ========== Expert 0: Survival (生存专家) ==========
    # 吃喝相关
    'eat': 0,
    'drink': 0,
    'quaff': 0,
    
    # 生存策略
    'pray': 0,
    'flee': 0,
    'wait': 0,
    'teleport': 0,
    
    # 装备管理
    'wear': 0,
    'put_on': 0,
    'remove': 0,
    'takeoff': 0,
    'remove_ring': 0,
    
    # 物品管理
    'drop': 0,
    'drop_multi': 0,
    'pickup': 0,
    
    # 其他生存
    'count_gold': 0,
    'save_game': 0,
    'sit': 0,
    
    # ========== Expert 1: Combat (战斗专家) ==========
    # 攻击动作
    'attack': 1,
    'attack_melee': 1,
    'attack_ranged': 1,
    'melee_attack': 1,
    'ranged_attack': 1,
    
    # 远程攻击
    'fire': 1,
    'throw': 1,
    
    # 近战技能
    'kick': 1,
    
    # 武器管理
    'wield': 1,
    'swap_weapon': 1,
    'enhance': 1,
    
    # 战斗策略
    'elbereth': 1,
    'turn_undead': 1,
    
    # Boss战
    'medusa': 1,
    'vlad': 1,
    
    # ========== Expert 2: Exploration (探索专家) ==========
    # 移动
    'move': 2,
    'move_no_pickup': 2,
    'move_no_fight': 2,
    'move_until_near': 2,
    
    # 搜索和观察
    'search': 2,
    'look': 2,
    'whatis': 2,
    'identify': 2,
    'identify_trap': 2,
    
    # 门和锁
    'open': 2,
    'open_door': 2,
    'close': 2,
    'close_door': 2,
    'unlock_door': 2,
    'force_lock': 2,
    
    # 楼层切换
    'go_up': 2,
    'go_down': 2,
    'climb': 2,
    'jump': 2,
    
    # 信息查询
    'inventory': 2,
    'known': 2,
    'call_name': 2,
    'name': 2,
    
    # 交互
    'loot': 2,
    'untrap': 2,
    
    # 特殊地点
    'minetown': 2,
    'oracle': 2,
    'sokoban': 2,
    
    # ========== Expert 3: General (通用专家) ==========
    # 物品使用
    'apply': 3,
    'invoke': 3,
    'zap': 3,
    'rub': 3,
    'read': 3,
    
    # 书写和清理
    'engrave': 3,
    'wipe': 3,
    
    # 物品组合
    'dip': 3,
    'put_in': 3,
    
    # 宗教相关
    'altar': 3,
    'offer': 3,
    'sacrifice': 3,
    'pay': 3,
    
    # 系统命令
    'extended_command': 3,
    'toggle_autopickup': 3,
}

# Expert名称列表
EXPERT_NAMES = ['Survival', 'Combat', 'Exploration', 'General']

# Expert描述
EXPERT_DESCRIPTIONS = {
    0: 'Survival - 生存相关（吃喝、回血、逃跑、装备管理）',
    1: 'Combat - 战斗相关（攻击、武器、战斗策略）',
    2: 'Exploration - 探索相关（移动、搜索、开门、信息查询）',
    3: 'General - 通用/兜底（物品使用、宗教、系统命令）',
}


def get_expert_for_operator(operator_name: str) -> int:
    """
    获取Operator对应的Expert索引
    
    Args:
        operator_name: Operator名称（如'move', 'attack'等）
    
    Returns:
        expert_idx: Expert索引（0-3），默认返回3（General）
    """
    return OPERATOR_TO_EXPERT.get(operator_name, 3)


def get_operators_for_expert(expert_idx: int) -> list:
    """
    获取某个Expert对应的所有Operators
    
    Args:
        expert_idx: Expert索引（0-3）
    
    Returns:
        operators: Operator名称列表
    """
    return [op for op, idx in OPERATOR_TO_EXPERT.items() if idx == expert_idx]


def print_mapping_stats():
    """打印映射统计信息"""
    print("\n" + "="*60)
    print("Operator到Expert映射统计")
    print("="*60)
    
    for expert_idx in range(4):
        operators = get_operators_for_expert(expert_idx)
        print(f"\n{EXPERT_DESCRIPTIONS[expert_idx]}")
        print(f"  Operators数量: {len(operators)}")
        print(f"  示例: {', '.join(operators[:5])}")
    
    print(f"\n总计: {len(OPERATOR_TO_EXPERT)} 个Operators")
    print("="*60 + "\n")


if __name__ == "__main__":
    # 测试
    print_mapping_stats()
    
    # 测试几个例子
    test_ops = ['move', 'attack', 'eat', 'apply', 'unknown_op']
    print("\n测试映射:")
    for op in test_ops:
        expert_idx = get_expert_for_operator(op)
        expert_name = EXPERT_NAMES[expert_idx]
        print(f"  {op:15s} -> Expert {expert_idx} ({expert_name})")
