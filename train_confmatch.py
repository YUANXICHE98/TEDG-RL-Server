#!/usr/bin/env python3
"""TEDG-RL NetHack训练 - ConfMatch(多通道匹配+置信度)版本

与 train_verbose.py 保持训练流程一致，但替换“随机选超边 + HP伪置信度”为：
- Top-K 超边匹配（4通道覆盖率 + 时间衰减）
- 在 Top-K 内做通道内选择（pre/scene/effect/rule 各选一条）
- confidence = max(score_i)（可视作当前解释整体可靠度）

默认输出到 results_confmatch/，避免影响正在跑的旧进程/日志。
"""

import os
import sys

# 兼容不同组件/插件使用的Key变量名：
# - 本仓库配置默认用 OPENAI_API_KEY
# - 某些外部插件/脚本会强制要求 CRS_OAI_KEY
if not os.getenv("CRS_OAI_KEY") and os.getenv("OPENAI_API_KEY"):
    os.environ["CRS_OAI_KEY"] = os.environ["OPENAI_API_KEY"]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import time
import gymnasium as gym
import nle.env
import nle.nethack as nh

from src.core.state_constructor import StateConstructor
from src.core.networks_correct import MultiChannelPolicyNet
from src.core.ppo_trainer import PPOTrainer
from src.core.action_masking import ActionMasker
from src.core.hypergraph_matcher import HypergraphMatcher
from src.core.hypergraph_loader import EmbeddingMatcher


# ============================================================================
# 开跑前自检 (Preflight Checks)
# ============================================================================
def preflight_checks(env, state_constructor, matcher, policy_net, device):
    """开跑前自检，任何失败直接抛异常"""
    print("\n" + "="*80)
    print("  [PREFLIGHT] 开跑前自检")
    print("="*80)
    errors = []
    
    # 1. 环境检查
    print("\n[1/6] 环境检查...")
    try:
        obs, info = env.reset()
        assert "blstats" in obs, "obs缺少blstats"
        assert "glyphs" in obs, "obs缺少glyphs"
        print(f"  ✓ env.reset() 成功")
        print(f"  ✓ obs keys: {list(obs.keys())[:8]}...")
    except Exception as e:
        errors.append(f"环境检查失败: {e}")
    
    # 2. blstats检查
    print("\n[2/6] blstats检查...")
    try:
        blstats = obs.get("blstats", np.zeros(1))
        assert blstats.shape[0] == nh.NLE_BLSTATS_SIZE, f"blstats长度={blstats.shape[0]}, 期望={nh.NLE_BLSTATS_SIZE}"
        hp = blstats[nh.NLE_BL_HP]
        hpmax = blstats[nh.NLE_BL_HPMAX]
        depth = blstats[nh.NLE_BL_DEPTH]
        score = blstats[nh.NLE_BL_SCORE]
        print(f"  ✓ blstats长度: {blstats.shape[0]}")
        print(f"  ✓ HP={hp}/{hpmax}, Depth={depth}, Score={score}")
        # 合理性检查
        assert 0 <= hp <= 500, f"HP={hp} 不合理"
        assert 0 <= hpmax <= 500, f"HPMax={hpmax} 不合理"
        assert 1 <= depth <= 60, f"Depth={depth} 不合理（新游戏应为1）"
    except Exception as e:
        errors.append(f"blstats检查失败: {e}")
    
    # 3. 超图数据检查
    print("\n[3/6] 超图数据检查...")
    try:
        n_edges = len(matcher.edges)
        assert n_edges > 0, "超图为空"
        sample_edge = matcher.edges[0]
        assert "operator" in sample_edge, "超边缺少operator字段"
        print(f"  ✓ 超边数量: {n_edges}")
        print(f"  ✓ 样例超边: id={sample_edge.get('id','NA')}, op={sample_edge.get('operator','NA')}")
    except Exception as e:
        errors.append(f"超图检查失败: {e}")
    
    # 4. 动作空间检查
    print("\n[4/6] 动作空间检查...")
    try:
        env_actions = env.action_space.n
        net_actions = policy_net.action_dim
        print(f"  ✓ 环境动作空间: {env_actions}")
        print(f"  ✓ 网络动作维度: {net_actions}")
        if env_actions != net_actions:
            print(f"  ⚠ 动作空间不匹配！env={env_actions}, net={net_actions}")
            print(f"    训练时会对动作取模: action % {env_actions}")
    except Exception as e:
        errors.append(f"动作空间检查失败: {e}")
    
    # 5. 网络前向检查
    print("\n[5/6] 网络前向检查...")
    try:
        dummy_state = np.zeros(115, dtype=np.float32)
        dummy_state[0] = 1.0  # hp_ratio
        state_tensor = torch.FloatTensor(dummy_state).to(device)
        with torch.no_grad():
            logits, alpha, value = policy_net(state_tensor)
        assert logits.shape[-1] == net_actions, f"logits维度错误: {logits.shape}"
        assert alpha.shape[-1] == 4, f"alpha维度错误: {alpha.shape}"
        print(f"  ✓ 前向传播成功")
        print(f"  ✓ logits shape: {logits.shape}")
        print(f"  ✓ alpha: [{alpha[0]:.3f}, {alpha[1]:.3f}, {alpha[2]:.3f}, {alpha[3]:.3f}]")
        print(f"  ✓ value: {value.item():.4f}")
    except Exception as e:
        errors.append(f"网络前向检查失败: {e}")
    
    # 6. 状态构造检查
    print("\n[6/6] 状态构造检查...")
    try:
        state = extract_state_from_nethack_obs(obs, state_constructor, matcher, t_now=0, verbose=False)
        assert state.shape == (115,), f"状态维度错误: {state.shape}"
        print(f"  ✓ 状态维度: {state.shape}")
        print(f"  ✓ belief[:5]: {state[:5]}")
        print(f"  ✓ confidence: {state[98]:.4f}")
    except Exception as e:
        errors.append(f"状态构造检查失败: {e}")
    
    # 汇总
    print("\n" + "="*80)
    if errors:
        print("  ❌ 自检失败！")
        for err in errors:
            print(f"    - {err}")
        print("="*80)
        raise RuntimeError(f"Preflight检查失败: {errors}")
    else:
        print("  ✅ 所有自检通过！")
        print("="*80)
    
    return True


def print_section(title):
    """打印分节标题"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def print_step(step, content):
    """打印步骤"""
    print(f"[步骤{step}] {content}")


def get_device():
    """检测设备"""
    try:
        import torch_musa

        if torch.musa.is_available():
            device = torch.device("musa:0")
            print(f"✓ MUSA GPU: {torch.musa.get_device_name(0)}")
            print(f"  显存: {torch.musa.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            return device
    except Exception:
        pass

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"✓ CUDA GPU: {torch.cuda.get_device_name(0)}")
        return device

    print("⚠ 使用CPU")
    return torch.device("cpu")


# ============================================================================
# 动态置信度阈值路由
# ============================================================================

class ConfidenceRouter:
    """基于滚动窗口的动态置信度阈值计算器"""
    
    def __init__(self, window_size: int = 500, warmup_steps: int = 100):
        self.window_size = window_size
        self.warmup_steps = warmup_steps
        self.history: list[float] = []
        self.high_threshold = 0.7  # 初始默认值
        self.low_threshold = 0.3   # 初始默认值
        
    def update(self, confidence: float):
        """添加新的置信度观测并更新阈值"""
        self.history.append(confidence)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        
        # warmup 阶段后开始动态计算阈值（可通过环境变量禁用）
        use_dynamic = os.getenv("TEDG_DYNAMIC_TH", "1") == "1"
        if use_dynamic and len(self.history) >= self.warmup_steps:
            sorted_conf = sorted(self.history)
            n = len(sorted_conf)
            self.high_threshold = sorted_conf[int(n * 0.75)]
            self.low_threshold = sorted_conf[int(n * 0.25)]
    
    def route(self, confidence: float) -> str:
        """根据置信度返回路由类型: 'high', 'mid', 'low'"""
        if confidence >= self.high_threshold:
            return "high"
        elif confidence >= self.low_threshold:
            return "mid"
        else:
            return "low"
    
    def get_stats(self) -> dict:
        """返回当前统计信息"""
        if not self.history:
            return {"count": 0, "high_th": self.high_threshold, "low_th": self.low_threshold}
        return {
            "count": len(self.history),
            "mean": np.mean(self.history),
            "std": np.std(self.history),
            "min": min(self.history),
            "max": max(self.history),
            "high_th": self.high_threshold,
            "low_th": self.low_threshold,
        }


def _safe_intersection(primary: list[str], other: list[str]) -> list[str]:
    """交集为空时回退primary，避免把通道向量打成全0。"""
    a = set(primary)
    b = set(other)
    inter = list(a & b)
    return inter if inter else list(primary)


# ============================================================================
# 完整 Atoms 解析（覆盖超图全部 65 pre_nodes + 82 scene_atoms）
# ============================================================================

# NLE 物品类别常量
OCLASS_WEAPON = 2
OCLASS_ARMOR = 3
OCLASS_RING = 4
OCLASS_AMULET = 5
OCLASS_POTION = 6
OCLASS_FOOD = 7
OCLASS_SCROLL = 8
OCLASS_WAND = 9
OCLASS_TOOL = 10
OCLASS_GEM = 11

# NLE Glyph 范围
GLYPH_MON_OFF = 0
GLYPH_PET_OFF = 381
GLYPH_INVIS_OFF = 762
GLYPH_DETECT_OFF = 763
GLYPH_BODY_OFF = 1144
GLYPH_RIDDEN_OFF = 1525
GLYPH_OBJ_OFF = 1906
GLYPH_CMAP_OFF = 2359

# 特定怪物 glyph 偏移（基于 NLE monst.c 顺序）
MONSTER_GLYPHS = {
    "newt": (GLYPH_MON_OFF + 56, GLYPH_PET_OFF + 56),
    "lichen": (GLYPH_MON_OFF + 95, GLYPH_PET_OFF + 95),
    "grid_bug": (GLYPH_MON_OFF + 196, GLYPH_PET_OFF + 196),
    "floating_eye": (GLYPH_MON_OFF + 79, GLYPH_PET_OFF + 79),
    "acid_blob": (GLYPH_MON_OFF + 60, GLYPH_PET_OFF + 60),
    "blue_jelly": (GLYPH_MON_OFF + 61, GLYPH_PET_OFF + 61),
    "brown_mold": (GLYPH_MON_OFF + 94, GLYPH_PET_OFF + 94),
    "yellow_light": (GLYPH_MON_OFF + 125, GLYPH_PET_OFF + 125),
    "gas_spore": (GLYPH_MON_OFF + 80, GLYPH_PET_OFF + 80),
    "flaming_sphere": (GLYPH_MON_OFF + 81, GLYPH_PET_OFF + 81),
    "freezing_sphere": (GLYPH_MON_OFF + 82, GLYPH_PET_OFF + 82),
    "shocking_sphere": (GLYPH_MON_OFF + 83, GLYPH_PET_OFF + 83),
    "shrieker": (GLYPH_MON_OFF + 93, GLYPH_PET_OFF + 93),
}

# 亡灵类怪物范围（大致）
UNDEAD_GLYPH_RANGES = [(GLYPH_MON_OFF + 220, GLYPH_MON_OFF + 280)]


def _parse_inventory(obs: dict) -> dict:
    """解析物品栏，返回物品类别统计和关键物品检测"""
    inv_oclasses = obs.get("inv_oclasses", np.zeros(55, dtype=np.uint8))
    inv_letters = obs.get("inv_letters", np.zeros(55, dtype=np.uint8))
    inv_strs = obs.get("inv_strs", np.zeros((55, 80), dtype=np.uint8))
    
    result = {
        "has_weapon": False,
        "has_armor": False,
        "has_food": False,
        "has_potion": False,
        "has_scroll": False,  # has_readable
        "has_wand": False,
        "has_ring": False,
        "has_amulet": False,  # has_accessory
        "has_tool": False,
        "has_container": False,
        "has_key_or_lockpick": False,
        "has_lamp_or_stone": False,
        "has_corpse": False,
        "has_ranged_weapon": False,
        "has_throwable": False,
        "has_ammo": False,
        "has_artifact": False,
        "has_liquid": False,
        "wearing_armor": False,
        "wearing_ring": False,
        "wearing_item": False,
        "item_count": 0,
        "slot_available": True,
    }
    
    for i in range(55):
        if inv_letters[i] == 0:
            continue
        result["item_count"] += 1
        oclass = inv_oclasses[i]
        
        # 解码物品描述
        desc_bytes = bytes(inv_strs[i].tolist()).split(b'\x00')[0]
        desc = desc_bytes.decode('latin-1', errors='ignore').lower()
        
        # 检测穿戴状态
        is_worn = "(being worn)" in desc or "(weapon in hand)" in desc or "(wielded)" in desc
        
        if oclass == OCLASS_WEAPON:
            result["has_weapon"] = True
            if "bow" in desc or "crossbow" in desc or "sling" in desc:
                result["has_ranged_weapon"] = True
            if "dart" in desc or "shuriken" in desc or "arrow" in desc or "bolt" in desc:
                result["has_ammo"] = True
                result["has_throwable"] = True
        elif oclass == OCLASS_ARMOR:
            result["has_armor"] = True
            if is_worn:
                result["wearing_armor"] = True
                result["wearing_item"] = True
        elif oclass == OCLASS_FOOD:
            result["has_food"] = True
            if "corpse" in desc:
                result["has_corpse"] = True
        elif oclass == OCLASS_POTION:
            result["has_potion"] = True
            result["has_liquid"] = True
        elif oclass == OCLASS_SCROLL:
            result["has_scroll"] = True  # has_readable
        elif oclass == OCLASS_WAND:
            result["has_wand"] = True
        elif oclass == OCLASS_RING:
            result["has_ring"] = True
            if is_worn:
                result["wearing_ring"] = True
                result["wearing_item"] = True
        elif oclass == OCLASS_AMULET:
            result["has_amulet"] = True
            if is_worn:
                result["wearing_item"] = True
        elif oclass == OCLASS_TOOL:
            result["has_tool"] = True
            if "key" in desc or "lock pick" in desc or "credit card" in desc:
                result["has_key_or_lockpick"] = True
            if "lamp" in desc or "lantern" in desc or "candle" in desc:
                result["has_lamp_or_stone"] = True
            if "bag" in desc or "sack" in desc or "chest" in desc:
                result["has_container"] = True
        elif oclass == OCLASS_GEM:
            if "stone" in desc or "rock" in desc:
                result["has_lamp_or_stone"] = True
            result["has_throwable"] = True
        
        # 神器检测
        if "excalibur" in desc or "mjollnir" in desc or "stormbringer" in desc:
            result["has_artifact"] = True
    
    result["slot_available"] = result["item_count"] < 52
    result["inventory_space"] = result["item_count"] < 52
    return result


def _analyze_glyphs(obs: dict, x: int, y: int) -> dict:
    """分析 glyphs 地图，检测邻近实体和地形"""
    glyphs = obs.get("glyphs", np.zeros((21, 79), dtype=np.int16))
    chars = obs.get("chars", np.zeros((21, 79), dtype=np.uint8))
    
    result = {
        "adjacent_to_monster": False,
        "adjacent_to_door": False,
        "adjacent_to_item": False,
        "adjacent_to_trap": False,
        "adjacent_to_container": False,
        "adjacent_to_target": False,
        "adjacent_to": False,
        "on_stairs": False,
        "on_upstairs": False,
        "on_downstairs": False,
        "on_altar": False,
        "near_altar": False,
        "in_shop": False,
        "monsters_present": False,
        "combat_situation": False,
        "see_monster": False,
        "target_adjacent": False,
        "monster_types": [],
    }
    
    # 玩家所在位置
    player_glyph = glyphs[y, x] if 0 <= y < 21 and 0 <= x < 79 else 0
    player_char = chr(chars[y, x]) if 0 <= y < 21 and 0 <= x < 79 else ' '
    
    # 检查玩家站立位置
    if player_char == '<':
        result["on_stairs"] = True
        result["on_upstairs"] = True
    elif player_char == '>':
        result["on_stairs"] = True
        result["on_downstairs"] = True
    elif player_char == '_':
        result["on_altar"] = True
    
    # 扫描周围 5x5 区域
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            ny, nx = y + dy, x + dx
            if not (0 <= ny < 21 and 0 <= nx < 79):
                continue
            if dy == 0 and dx == 0:
                continue
            
            g = glyphs[ny, nx]
            c = chr(chars[ny, nx])
            is_adjacent = abs(dy) <= 1 and abs(dx) <= 1
            
            # 怪物检测 (glyph 0-380 普通怪物, 381-761 宠物)
            if GLYPH_MON_OFF <= g < GLYPH_INVIS_OFF:
                result["monsters_present"] = True
                result["see_monster"] = True
                if is_adjacent:
                    result["adjacent_to_monster"] = True
                    result["adjacent_to_target"] = True
                    result["target_adjacent"] = True
                    result["combat_situation"] = True
                    result["adjacent_to"] = True
                
                # 特定怪物类型
                for mname, (gmin, gmax) in MONSTER_GLYPHS.items():
                    if g == gmin or g == gmax:
                        result["monster_types"].append(mname)
                
                # 亡灵检测
                for umin, umax in UNDEAD_GLYPH_RANGES:
                    if umin <= g <= umax:
                        result["nearby_undead"] = True
            
            # 门检测
            if c == '+' or c == '|' and GLYPH_CMAP_OFF <= g < GLYPH_CMAP_OFF + 100:
                if is_adjacent:
                    result["adjacent_to_door"] = True
                    result["adjacent_to"] = True
            
            # 物品检测
            if GLYPH_OBJ_OFF <= g < GLYPH_CMAP_OFF:
                if is_adjacent:
                    result["adjacent_to_item"] = True
                    result["adjacent_to"] = True
            
            # 陷阱检测
            if c == '^':
                if is_adjacent:
                    result["adjacent_to_trap"] = True
            
            # 祭坛检测
            if c == '_':
                result["near_altar"] = True
            
            # 楼梯检测（非玩家位置）
            if c in '<>':
                if is_adjacent:
                    result["adjacent_to"] = True
    
    return result


def extract_state_from_nethack_obs(
    obs: dict,
    state_constructor: StateConstructor,
    matcher: HypergraphMatcher,
    t_now: int,
    verbose: bool = False,
    embedding_matcher: "EmbeddingMatcher | None" = None,
) -> np.ndarray:
    """从NetHack观测提取state（完整atoms解析 + ConfMatch匹配超边）
    
    如果 embedding_matcher 不为 None，则使用嵌入匹配计算 confidence
    """
    blstats = obs.get("blstats", np.zeros(nh.NLE_BLSTATS_SIZE))
    
    # 基础数值
    hp = int(blstats[nh.NLE_BL_HP])
    hpmax = int(blstats[nh.NLE_BL_HPMAX])
    depth = int(blstats[nh.NLE_BL_DEPTH])
    gold = int(blstats[nh.NLE_BL_GOLD])
    hunger = int(blstats[nh.NLE_BL_HUNGER])
    x, y = int(blstats[nh.NLE_BL_X]), int(blstats[nh.NLE_BL_Y])
    ac = int(blstats[nh.NLE_BL_AC])
    exp_level = int(blstats[nh.NLE_BL_XLEVEL]) if hasattr(nh, 'NLE_BL_XLEVEL') else int(blstats[nh.NLE_BL_EXP])
    power = int(blstats[nh.NLE_BL_ENE]) if hasattr(nh, 'NLE_BL_ENE') else 0
    power_max = int(blstats[nh.NLE_BL_ENEMAX]) if hasattr(nh, 'NLE_BL_ENEMAX') else 1
    condition = int(blstats[nh.NLE_BL_CONDITION]) if hasattr(nh, 'NLE_BL_CONDITION') else 0
    
    hp_ratio = hp / max(hpmax, 1)
    power_ratio = power / max(power_max, 1)
    
    if verbose:
        print(f"\n  [观测解析 - 完整版]")
        print(f"    HP: {hp}/{hpmax} ({hp_ratio*100:.0f}%)")
        print(f"    深度: {depth}层, 位置: ({x}, {y})")
        print(f"    金币: {gold}, AC: {ac}, Exp: {exp_level}")
        print(f"    饥饿值: {hunger}, 魔力: {power}/{power_max}")
        print(f"    状态位: {bin(condition)}")

    # 构造belief (50维)
    belief = np.zeros(50, dtype=np.float32)
    belief[0] = hp_ratio
    belief[1] = depth / 50.0
    belief[2] = min(gold / 1000.0, 1.0)
    belief[3] = hunger / 1000.0
    belief[4] = blstats[nh.NLE_BL_STR25] / 25.0
    belief[5] = blstats[nh.NLE_BL_DEX] / 25.0
    belief[6] = blstats[nh.NLE_BL_CON] / 25.0
    belief[7] = blstats[nh.NLE_BL_INT] / 25.0
    belief[8] = blstats[nh.NLE_BL_WIS] / 25.0
    belief[9] = blstats[nh.NLE_BL_CHA] / 25.0
    belief[10] = x / 79.0
    belief[11] = y / 21.0
    belief[12] = blstats[nh.NLE_BL_SCORE] / 10000.0
    belief[13] = ac / 20.0 + 0.5  # 归一化 AC
    belief[14] = exp_level / 30.0
    belief[15] = power_ratio
    belief[30] = 1.0 if hp_ratio < 0.3 else 0.0
    belief[31] = 1.0 if hunger > 800 else 0.0
    
    # 解析物品栏
    inv_info = _parse_inventory(obs)
    
    # 分析地图
    glyph_info = _analyze_glyphs(obs, x, y)
    
    # 解析状态效果 (condition bits)
    is_blind = bool(condition & nh.BL_MASK_BLIND) if hasattr(nh, 'BL_MASK_BLIND') else False
    is_confused = bool(condition & nh.BL_MASK_CONF) if hasattr(nh, 'BL_MASK_CONF') else False
    is_stunned = bool(condition & nh.BL_MASK_STUN) if hasattr(nh, 'BL_MASK_STUN') else False
    is_hallucinating = bool(condition & nh.BL_MASK_HALLU) if hasattr(nh, 'BL_MASK_HALLU') else False

    # ========== 构建 pre_nodes (65个词汇) ==========
    pre_nodes: list[str] = []
    
    # HP 状态
    pre_nodes.append("player_alive")
    pre_nodes.append("game_active")
    pre_nodes.append("any_hp")
    if hp_ratio >= 0.9:
        pre_nodes.append("hp_full")
    elif hp_ratio < 0.15:
        pre_nodes.append("hp_critical")
    elif hp_ratio < 0.3:
        pre_nodes.append("hp_low")
    
    # 饥饿状态
    if hunger < 150:
        pre_nodes.append("hunger_satiated")
    elif hunger < 500:
        pre_nodes.append("hunger_normal")
        pre_nodes.append("not_full")
    elif hunger < 800:
        pre_nodes.append("hunger_hungry")
        pre_nodes.append("not_full")
    else:
        pre_nodes.append("hunger_weak")
        pre_nodes.append("not_full")
    
    # 金币
    if gold > 0:
        pre_nodes.append("has_gold")
    else:
        pre_nodes.append("no_gold")
    
    # 魔力
    if power_ratio >= 0.9:
        pre_nodes.append("power_full")
    elif power_ratio < 0.1:
        pre_nodes.append("power_empty")
    
    # 状态效果
    if is_blind:
        pre_nodes.append("blind")
    else:
        pre_nodes.append("not_blind")
    if is_confused:
        pre_nodes.append("confused")
    if is_stunned:
        pre_nodes.append("stunned")
    if is_hallucinating:
        pre_nodes.append("hallucinating")
    
    # 物品栏状态
    pre_nodes.append("hands_free")  # 默认假设
    if inv_info["has_weapon"]:
        pre_nodes.append("has_weapon")
    if inv_info["has_armor"]:
        pre_nodes.append("has_armor")
    if inv_info["has_food"]:
        pre_nodes.append("has_food")
    if inv_info["has_potion"]:
        pre_nodes.append("has_potion")
    if inv_info["has_scroll"]:
        pre_nodes.append("has_readable")
    if inv_info["has_wand"]:
        pre_nodes.append("has_wand")
        pre_nodes.append("wand_has_charges")  # 假设有电荷
    if inv_info["has_ring"] or inv_info["has_amulet"]:
        pre_nodes.append("has_accessory")
    if inv_info["has_tool"]:
        pre_nodes.append("has_item")
    if inv_info["has_container"]:
        pre_nodes.append("has_container")
    if inv_info["has_key_or_lockpick"]:
        pre_nodes.append("has_key_OR_lockpick")
    if inv_info["has_lamp_or_stone"]:
        pre_nodes.append("has_lamp_or_stone")
    if inv_info["has_corpse"]:
        pre_nodes.append("has_corpse")
    if inv_info["has_ranged_weapon"]:
        pre_nodes.append("has_ranged_weapon")
    if inv_info["has_throwable"]:
        pre_nodes.append("has_throwable")
    if inv_info["has_ammo"]:
        pre_nodes.append("has_ammo")
    if inv_info["has_artifact"]:
        pre_nodes.append("has_artifact")
    if inv_info["has_liquid"]:
        pre_nodes.append("has_liquid")
    if inv_info["wearing_armor"]:
        pre_nodes.append("wearing_armor")
    if inv_info["wearing_ring"]:
        pre_nodes.append("wearing_ring")
    if inv_info["wearing_item"]:
        pre_nodes.append("wearing_item")
    if inv_info["slot_available"]:
        pre_nodes.append("slot_available")
        pre_nodes.append("inventory_space")
    
    # 地图/怪物相关前置条件
    if glyph_info["see_monster"]:
        pre_nodes.append("see_monster")
        pre_nodes.append("monster_alive")
    if glyph_info["target_adjacent"]:
        pre_nodes.append("target_adjacent")
        pre_nodes.append("target_in_range")
        pre_nodes.append("target_not_blocked")
        pre_nodes.append("target_passable")
    if glyph_info.get("nearby_undead"):
        pre_nodes.append("nearby_undead")
    if glyph_info["combat_situation"]:
        pre_nodes.append("keep_distance")
    if glyph_info["on_altar"]:
        pre_nodes.append("on_altar")
    if glyph_info["on_upstairs"]:
        pre_nodes.append("on_upstairs")
    if glyph_info["on_downstairs"]:
        pre_nodes.append("on_downstairs")
    if glyph_info["adjacent_to_item"]:
        pre_nodes.append("item_collectible")
    if glyph_info["adjacent_to_trap"]:
        pre_nodes.append("trap_known")
    
    # 门状态（如果邻近门）
    if glyph_info["adjacent_to_door"]:
        pre_nodes.append("is_not_open")  # 假设关闭
        pre_nodes.append("is_not_locked")  # 假设未锁
    
    # ========== 构建 scene_atoms (82个词汇) ==========
    scene_atoms: list[str] = []
    
    # 深度等级
    if depth == 1:
        scene_atoms.append("dlvl_1")
    elif 2 <= depth <= 5:
        scene_atoms.append("dlvl_2_5")
    elif depth == 5:
        scene_atoms.append("dlvl_5")
    elif depth == 7:
        scene_atoms.append("dlvl_7")
    elif depth == 9:
        scene_atoms.append("dlvl_9")
    elif depth == 10:
        scene_atoms.append("dlvl_10")
    elif depth == 15:
        scene_atoms.append("dlvl_15")
    elif depth == 26:
        scene_atoms.append("dlvl_26")
    elif depth == 36:
        scene_atoms.append("dlvl_36")
    
    # 经验等级
    if exp_level == 1:
        scene_atoms.append("exp_1")
    elif 2 <= exp_level <= 5:
        scene_atoms.append("exp_2_5")
    
    # AC 状态
    if ac < 0:
        scene_atoms.append("ac_good")
    elif ac > 5:
        scene_atoms.append("ac_poor")
    
    # 地图实体
    if glyph_info["adjacent_to"]:
        scene_atoms.append("adjacent_to")
    if glyph_info["adjacent_to_monster"]:
        scene_atoms.append("adjacent_to_monster")
        scene_atoms.append("adjacent_to_target")
        scene_atoms.append("combat_situation")
    if glyph_info["adjacent_to_door"]:
        scene_atoms.append("adjacent_to_door")
    if glyph_info["adjacent_to_item"]:
        scene_atoms.append("adjacent_to_item")
    if glyph_info["adjacent_to_trap"]:
        scene_atoms.append("adjacent_to_trap")
    if glyph_info["adjacent_to_container"]:
        scene_atoms.append("adjacent_to_container")
    if glyph_info["monsters_present"]:
        scene_atoms.append("monsters_present")
    if glyph_info["on_stairs"]:
        scene_atoms.append("on_stairs")
    if glyph_info["near_altar"]:
        scene_atoms.append("near_altar")
    if glyph_info["in_shop"]:
        scene_atoms.append("in_shop")
    
    # 特定怪物类型
    for mtype in glyph_info["monster_types"]:
        scene_atoms.append(f"monster_{mtype}")
    
    # 爆炸风险（gas spore, sphere 等）
    if any(m in glyph_info["monster_types"] for m in ["gas_spore", "flaming_sphere", "freezing_sphere", "shocking_sphere"]):
        scene_atoms.append("explosion_risk")
    
    # 策略场景（基于深度和状态推断）
    if depth <= 3 and exp_level <= 2:
        scene_atoms.append("strategy_eat")  # 早期生存
    if glyph_info["near_altar"]:
        scene_atoms.append("strategy_altar")
        scene_atoms.append("strategy_pray")
    if glyph_info["combat_situation"]:
        scene_atoms.append("strategy_attack")
        if hp_ratio < 0.3:
            scene_atoms.append("strategy_flee")
    if depth == 5 or depth == 6:
        scene_atoms.append("strategy_minetown")
    if depth == 8 or depth == 9:
        scene_atoms.append("strategy_oracle")
    if 5 <= depth <= 9:
        scene_atoms.append("strategy_sokoban")
    
    # 安全战斗（HP足够+邻近弱怪）
    if glyph_info["combat_situation"] and hp_ratio > 0.5:
        weak_monsters = ["newt", "lichen", "grid_bug"]
        if any(m in glyph_info["monster_types"] for m in weak_monsters):
            scene_atoms.append("safe_combat")
    
    # 基于规则的默认标记
    scene_atoms.append("rule_based")
    
    # ========== 构建 effect_atoms（使用超图中实际的 eff_nodes 词汇） ==========
    effect_atoms: list[str] = []
    
    # 基于当前状态推断期望的效果（使用超边中的实际词汇）
    if hp_ratio < 0.5:
        effect_atoms.append("hp_restored")
        effect_atoms.append("healed")
    if hunger > 500:
        effect_atoms.append("ate_food")
        effect_atoms.append("hunger_reduced")
    if glyph_info["adjacent_to_monster"]:
        effect_atoms.append("combat_success")
        effect_atoms.append("corpse_created")
        effect_atoms.append("xp_gained")
        effect_atoms.append("easy_kill")
    if glyph_info["adjacent_to_item"]:
        effect_atoms.append("item_obtained")
        effect_atoms.append("inventory_updated")
    if glyph_info["on_stairs"]:
        effect_atoms.append("level_changed")
        effect_atoms.append("exploration_done")
    if inv_info["has_weapon"]:
        effect_atoms.append("attack_melee_enabled")
    if inv_info["has_armor"] and not inv_info["wearing_armor"]:
        effect_atoms.append("ac_updated")
        effect_atoms.append("armor_equipped")
    if glyph_info["adjacent_to_door"]:
        effect_atoms.append("door_opened")
        effect_atoms.append("door_found")
    # 通用效果
    effect_atoms.append("area_safe")
    effect_atoms.append("exploration_done")
    
    # ========== 构建 rule_atoms（使用超图中的 conditional_effects 词汇） ==========
    rule_atoms: list[str] = []
    
    # 基于物品和状态推断适用的规则
    if inv_info["has_food"]:
        effect_atoms.append("ate_food")
        rule_atoms.append("corpse_consumed")
    if inv_info["has_potion"]:
        effect_atoms.append("potion_consumed")
        rule_atoms.append("got_blessed")
        rule_atoms.append("got_cursed")
    if inv_info["has_scroll"]:
        effect_atoms.append("scroll_read")
    if inv_info["has_wand"]:
        effect_atoms.append("charges_decreased")
    if glyph_info["near_altar"]:
        effect_atoms.append("prayed")
        rule_atoms.append("got_blessed")
    if glyph_info["adjacent_to_door"]:
        effect_atoms.append("door_opened")
        effect_atoms.append("door_broken_chance")
    if is_blind:
        rule_atoms.append("avoid_contact")
    if is_confused:
        rule_atoms.append("became_confused")
    # 通用规则
    rule_atoms.append("maybe_poisoned")
    rule_atoms.append("if item.blessed == True")
    
    if verbose:
        print(f"    提取 pre_nodes: {len(pre_nodes)}个 - {pre_nodes[:10]}...")
        print(f"    提取 scene_atoms: {len(scene_atoms)}个 - {scene_atoms[:10]}...")
        print(f"    提取 effect_atoms: {len(effect_atoms)}个 - {effect_atoms[:8]}...")
        print(f"    提取 rule_atoms: {len(rule_atoms)}个 - {rule_atoms[:8]}...")

    # ConfMatch: plot_atoms -> Top-K matched hyperedges
    plot_atoms = {"pre": pre_nodes, "scene": scene_atoms, "effect": effect_atoms, "rule": rule_atoms}
    
    # 根据是否有 embedding_matcher 选择匹配方式
    if embedding_matcher is not None:
        # 嵌入匹配模式：基于余弦相似度
        emb_confidence, emb_topk = embedding_matcher.match(pre_nodes, scene_atoms, effect_atoms, rule_atoms, top_k=8)
        confidence = emb_confidence
        
        if verbose:
            print(f"\n  [嵌入匹配模式]")
            print(f"    atoms总数: {len(pre_nodes)+len(scene_atoms)+len(effect_atoms)+len(rule_atoms)}")
            print(f"    atom缓存: {len(embedding_matcher.atom_cache)} 个")
            print(f"    confidence(余弦相似度): {confidence:.3f}")
            for j, item in enumerate(emb_topk[:5], start=1):
                print(f"    #{j} op={item['operator']} sim={item['similarity']:.3f}")
        
        # 使用嵌入匹配的 top-1 边获取元数据
        if emb_topk:
            best_edge = emb_topk[0]["edge"]
            eff_metadata = best_edge.get("eff_metadata", {}) or {}
            conditional_effects = eff_metadata.get("conditional_effects", []) or []
            pre_for_state = _safe_intersection(pre_nodes, list(best_edge.get("pre_nodes", []) or []))
            scene_for_state = _safe_intersection(scene_atoms, list(best_edge.get("scene_atoms", []) or []))
        else:
            eff_metadata = {}
            conditional_effects = []
            pre_for_state = list(pre_nodes)
            scene_for_state = list(scene_atoms)
    else:
        # 覆盖率匹配模式：基于字符串集合交集
        topk = matcher.match(plot_atoms, t_now=float(t_now), t_i=float(t_now), top_k=8)
        if not topk:
            eff_metadata = {}
            conditional_effects = []
            confidence = 0.0
            pre_for_state = list(pre_nodes)
            scene_for_state = list(scene_atoms)
        else:
            selected = HypergraphMatcher.select_channel_edges(topk)
            confidence = float(max(r.score for r in topk))

            pre_edge = selected["pre"].edge
            scene_edge = selected["scene"].edge
            effect_edge = selected["effect"].edge
            rule_edge = selected["rule"].edge

            pre_for_state = _safe_intersection(pre_nodes, list(pre_edge.get("pre_nodes", []) or []))
            scene_for_state = _safe_intersection(scene_atoms, list(scene_edge.get("scene_atoms", []) or []))
            eff_metadata = effect_edge.get("eff_metadata", {}) or {}
            rule_meta = rule_edge.get("eff_metadata", {}) or {}
            conditional_effects = rule_meta.get("conditional_effects", []) or []

            if verbose:
                print(f"\n  [覆盖率匹配模式]")
                print(f"    Top-K=8, confidence(max score)={confidence:.3f}")
                for j, r in enumerate(topk[:5], start=1):
                    eid = r.edge.get("id", "NA")
                    op = r.edge.get("operator", "NA")
                    mv = r.match_vec
                    print(
                        f"    #{j} {eid} op={op} score={r.score:.3f} "
                        f"cov=[{mv[0]:.2f},{mv[1]:.2f},{mv[2]:.2f},{mv[3]:.2f}] argmax={r.channel_argmax}"
                    )
                print(
                    f"    选边: pre={pre_edge.get('id','NA')} scene={scene_edge.get('id','NA')} "
                    f"effect={effect_edge.get('id','NA')} rule={rule_edge.get('id','NA')}"
                )
                print(f"    pre证据: {pre_for_state[:5]}")
                print(f"    scene证据: {scene_for_state[:6]}")

    if verbose:
        print(f"\n  [置信度]")
        print(f"    confidence: {confidence:.3f}")

    goal = np.zeros(16, dtype=np.float32)
    goal[0] = 1.0

    state = state_constructor.construct_state(
        belief_vector=belief,
        pre_nodes=pre_for_state,
        scene_atoms=scene_for_state,
        eff_metadata=eff_metadata,
        conditional_effects=conditional_effects,
        confidence=confidence,
        goal_embedding=goal,
    )

    if verbose:
        print(f"\n  [状态构造]")
        print(f"    state维度: {state.shape}")
        print(f"    belief: {belief[:5]} ...")
        print(f"    q_pre: {state[50:55]} ...")
        print(f"    q_scene: {state[65:70]} ...")
        print(f"    q_effect: {state[80:85]} ...")
        print(f"    q_rule: {state[88:93]} ...")

    return state


def main():
    """主训练循环"""
    print_section("TEDG-RL NetHack训练 - ConfMatch(多通道匹配)版")

    # 设备检测
    print("\n[初始化]")
    print_step(1, "检测计算设备")
    device = get_device()

    # 创建输出目录（避免覆盖旧进程日志）
    output_dir = Path(os.getenv("TEDG_OUTPUT_DIR", "results_confmatch"))
    output_dir.mkdir(exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    # 加载超图
    print_step(2, "加载超图数据")
    state_constructor = StateConstructor("data/hypergraph/hypergraph_complete_real.json")
    print(f"  ✓ 加载完成")

    # 初始化匹配器（支持两种模式：coverage 覆盖率 / embedding 嵌入）
    use_embedding = os.getenv("TEDG_USE_EMBEDDING", "0") == "1"
    print_step(3, f"初始化匹配器 (mode={'embedding' if use_embedding else 'coverage'})")
    
    if use_embedding:
        # 嵌入匹配：基于余弦相似度
        embedding_matcher = EmbeddingMatcher(min_support=5)
        matcher = HypergraphMatcher(state_constructor.hypergraph, weights=(0.35, 0.35, 0.2, 0.1), tau=200.0)
        print(f"  ✓ EmbeddingMatcher就绪: {len(embedding_matcher.atom_cache)} 个 atom 缓存")
    else:
        # 覆盖率匹配：基于字符串集合交集
        embedding_matcher = None
        matcher = HypergraphMatcher(state_constructor.hypergraph, weights=(0.35, 0.35, 0.2, 0.1), tau=200.0)
        print(f"  ✓ HypergraphMatcher就绪: edges={len(matcher.edges)}, tau={matcher.tau}")

    # 初始化动作掩蔽（当前训练脚本未使用，但保持初始化以对齐旧版）
    print_step(4, "初始化动作掩蔽器")
    action_masker = ActionMasker(state_constructor.hypergraph, num_actions=23)
    print(f"  ✓ 初始化完成")

    # 创建NetHack环境
    print_step(5, "创建NetHack环境")
    try:
        env = gym.make("NetHackScore-v0")
        print(f"  ✓ NetHackScore-v0")
    except Exception:
        env = gym.make("NetHack-v0")
        print(f"  ✓ NetHack-v0")
    print(f"  动作空间: {env.action_space.n}个动作")

    # 创建网络
    print_step(6, "初始化多通道策略网络")
    policy_net = MultiChannelPolicyNet(
        state_dim=115,
        action_dim=23,
        actor_hidden_dim=128,
        attention_hidden_dim=64,
    )
    total_params = sum(p.numel() for p in policy_net.parameters())
    print(f"  ✓ 网络参数: {total_params:,}")
    print(f"  - 4个独立Actor")
    print(f"  - 1个AttentionWeightNet")
    print(f"  - 1个共享Critic")

    # 创建训练器
    print_step(7, "初始化PPO训练器")
    trainer = PPOTrainer(
        policy_net=policy_net,
        learning_rate=3e-4,
        clip_ratio=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        ppo_epochs=3,
        batch_size=128,
        device=device,
    )
    print(f"  ✓ 训练器就绪")

    # 开跑前自检（可通过 TEDG_PREFLIGHT=1 启用）
    if os.getenv("TEDG_PREFLIGHT", "0") == "1":
        preflight_checks(env, state_constructor, matcher, policy_net, device)

    # 超详细日志开关（可通过 TEDG_VERBOSE_STEP=1 启用）
    verbose_step_mode = os.getenv("TEDG_VERBOSE_STEP", "0") == "1"
    if verbose_step_mode:
        print("\n⚠️ 超详细日志模式已启用（TEDG_VERBOSE_STEP=1）")

    # 训练参数
    num_episodes = int(os.getenv("TEDG_NUM_EPISODES", "10000"))
    max_steps = int(os.getenv("TEDG_MAX_STEPS", "1000"))
    eval_interval = int(os.getenv("TEDG_EVAL_INTERVAL", "50"))
    checkpoint_interval = int(os.getenv("TEDG_CKPT_INTERVAL", "500"))
    verbose_interval = int(os.getenv("TEDG_VERBOSE_INTERVAL", "10"))

    # 统计
    episode_rewards = []
    episode_lengths = []
    episode_scores = []
    alpha_history = []
    best_reward = float("-inf")
    best_score = 0
    
    # 动态置信度路由器
    conf_router = ConfidenceRouter(window_size=500, warmup_steps=100)
    route_counts = {"high": 0, "mid": 0, "low": 0}

    start_time = time.time()

    print_section("开始训练")
    print(f"总Episodes: {num_episodes}")
    print(f"每Episode最大步数: {max_steps}")
    print(f"设备: {device}")
    print(f"输出目录: {output_dir}")
    print(f"目标: 学习α权重动态分配 + 最大化NetHack分数")

    global_step = 0

    # 主训练循环
    for episode in range(num_episodes):
        verbose = episode % verbose_interval == 0

        if verbose:
            print(f"\n{'─'*80}")
            print(f"Episode {episode+1}/{num_episodes}")
            print(f"{'─'*80}")

        if verbose:
            print(f"\n[1. 重置NetHack环境]")
        obs, info = env.reset()
        state = extract_state_from_nethack_obs(obs, state_constructor, matcher, t_now=global_step, verbose=verbose, embedding_matcher=embedding_matcher)

        done = False
        truncated = False
        total_reward = 0
        steps = 0
        episode_alphas = []

        while not (done or truncated) and steps < max_steps:
            # 提取当前置信度并更新路由器
            confidence = float(state[98])  # 置信度在state的第98维
            conf_router.update(confidence)
            route = conf_router.route(confidence)
            route_counts[route] += 1
            
            # 获取动作掩码（基于当前置信度路由）
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(device)
                logits, alpha, value = trainer.policy_net(state_tensor)
                episode_alphas.append(alpha.cpu().numpy())
                
                # 应用动作掩码（中/高置信度时启用，可通过环境变量禁用）
                use_mask = os.getenv("TEDG_USE_MASK", "1") == "1"
                if use_mask and route in ("high", "mid"):
                    mask = action_masker.get_action_mask([], [], confidence)
                    mask_tensor = torch.BoolTensor(mask).to(device)
                    masked_logits = logits.clone()
                    masked_logits[~mask_tensor] = float('-inf')
                else:
                    masked_logits = logits
                
                # 从掩码后的logits采样动作
                dist = torch.distributions.Categorical(logits=masked_logits)
                action_tensor = dist.sample()
                log_prob = dist.log_prob(action_tensor)
                action = action_tensor.item()
                
                # 超详细日志：每步都打印
                if verbose_step_mode:
                    blstats = obs.get("blstats", np.zeros(nh.NLE_BLSTATS_SIZE))
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    top5_idx = np.argsort(probs)[-5:][::-1]
                    print(f"\n  ┌─ Step {steps} ─────────────────────────────────")
                    print(f"  │ raw blstats: HP={int(blstats[nh.NLE_BL_HP])}/{int(blstats[nh.NLE_BL_HPMAX])}, "
                          f"Depth={int(blstats[nh.NLE_BL_DEPTH])}, Gold={int(blstats[nh.NLE_BL_GOLD])}, "
                          f"Score={int(blstats[nh.NLE_BL_SCORE])}")
                    print(f"  │ confidence: {state[98]:.4f}")
                    print(f"  │ α权重: pre={alpha[0]:.3f}, scene={alpha[1]:.3f}, effect={alpha[2]:.3f}, rule={alpha[3]:.3f}")
                    print(f"  │ V(s): {value.item():.4f}")
                    print(f"  │ Top-5动作概率: {[(int(i), f'{probs[i]:.3f}') for i in top5_idx]}")
                    print(f"  │ 选择动作: {action} (prob={probs[action]:.4f}, log_prob={log_prob:.4f})")

            if verbose and steps == 0:
                print(f"\n[2. 网络决策 - 第1步] route={route}")
                print(f"  置信度: {confidence:.3f} (th: {conf_router.low_threshold:.3f}/{conf_router.high_threshold:.3f})")
                print(f"  α权重: pre={alpha[0]:.3f}, scene={alpha[1]:.3f}, effect={alpha[2]:.3f}, rule={alpha[3]:.3f}")
                print(f"  state维度检查: belief[0:5]={state[:5]}, q_pre[50:55]={state[50:55]}")
                print(f"  选择动作: {action}, V(s)={value.item():.4f}")

            obs, reward, done, truncated, info = env.step(action)

            if verbose_step_mode:
                print(f"  │ 执行后: reward={reward:.3f}, done={done or truncated}")
                print(f"  └─────────────────────────────────────────────")

            if verbose and steps == 0:
                print(f"\n[3. 执行动作]")
                print(f"  动作ID: {action}")
                print(f"  奖励: {reward:.3f}")
                print(f"  完成: {done or truncated}")

            global_step += 1
            next_state = extract_state_from_nethack_obs(obs, state_constructor, matcher, t_now=global_step, verbose=verbose_step_mode, embedding_matcher=embedding_matcher)
            trainer.store_transition(state, action, reward, next_state, done or truncated, log_prob)

            state = next_state
            total_reward += reward
            steps += 1

        if verbose:
            print(f"\n[4. 学习更新]")
            print(f"  收集经验: {len(trainer.buffer)}条")
            conf_stats = conf_router.get_stats()
            print(f"  置信度统计: mean={conf_stats.get('mean', 0):.3f}, std={conf_stats.get('std', 0):.3f}")
            print(f"  置信度范围: [{conf_stats.get('min', 0):.3f}, {conf_stats.get('max', 0):.3f}]")
            print(f"  动态阈值: low={conf_stats['low_th']:.3f}, high={conf_stats['high_th']:.3f}")
            total_routes = route_counts['high'] + route_counts['mid'] + route_counts['low']
            print(f"  路由分布: high={route_counts['high']}({route_counts['high']/max(total_routes,1)*100:.1f}%), "
                  f"mid={route_counts['mid']}({route_counts['mid']/max(total_routes,1)*100:.1f}%), "
                  f"low={route_counts['low']}({route_counts['low']/max(total_routes,1)*100:.1f}%)")

        update_stats = trainer.update()

        if verbose and update_stats:
            print(f"  Actor Loss: {update_stats.get('actor_loss', 0):.4f}")
            print(f"  Critic Loss: {update_stats.get('critic_loss', 0):.4f}")
            print(f"  平均优势: {update_stats.get('avg_advantage', 0):.4f}")

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        final_score = (
            obs.get("blstats", [0] * nh.NLE_BLSTATS_SIZE)[nh.NLE_BL_SCORE] if isinstance(obs, dict) else 0
        )
        episode_scores.append(final_score)

        if episode_alphas:
            avg_alpha = np.mean(episode_alphas, axis=0)
            alpha_history.append(avg_alpha)

        if total_reward > best_reward:
            best_reward = total_reward
            trainer.save_checkpoint(str(output_dir / "checkpoints" / "best_model.pth"))
            if verbose:
                print(f"\n  🎉 新最佳奖励: {best_reward:.2f}")

        if final_score > best_score:
            best_score = final_score
            if verbose:
                print(f"  🎉 新最佳分数: {best_score:.0f}")

        if (episode + 1) % checkpoint_interval == 0:
            trainer.save_checkpoint(str(output_dir / "checkpoints" / f"model_{episode+1:05d}.pth"))
            # 同时保存 atom 嵌入缓存（防止中断丢失）
            if embedding_matcher is not None:
                embedding_matcher.save_cache()
            print(f"\n💾 保存检查点: model_{episode+1:05d}.pth")

        if (episode + 1) % eval_interval == 0:
            avg_reward = np.mean(episode_rewards[-eval_interval:])
            avg_length = np.mean(episode_lengths[-eval_interval:])
            avg_score = np.mean(episode_scores[-eval_interval:])

            if len(alpha_history) >= eval_interval:
                recent_alphas = np.array(alpha_history[-eval_interval:])
                avg_alpha = recent_alphas.mean(axis=0)

                print(f"\n{'─'*80}")
                print(f"📊 评估统计 [Episode {episode+1}]")
                print(f"{'─'*80}")
                print(f"  奖励: 平均={avg_reward:.2f}, 最佳={best_reward:.2f}")
                print(f"  分数: 平均={avg_score:.0f}, 最佳={best_score:.0f}")
                print(f"  长度: {avg_length:.0f}步")
                print(f"  α权重: pre={avg_alpha[0]:.3f}, scene={avg_alpha[1]:.3f}, effect={avg_alpha[2]:.3f}, rule={avg_alpha[3]:.3f}")

                elapsed = time.time() - start_time
                eps_per_sec = (episode + 1) / elapsed
                print(f"  速度: {eps_per_sec:.2f} eps/s")
                print(f"  已用时间: {elapsed/60:.1f}分钟")

    env.close()
    
    # 保存 atom 嵌入缓存（如果使用嵌入匹配模式）
    if embedding_matcher is not None:
        embedding_matcher.save_cache()

    trainer.save_checkpoint(str(output_dir / "checkpoints" / "model_final.pth"))

    log_data = {
        "episode_rewards": [float(r) for r in episode_rewards],
        "episode_lengths": [int(l) for l in episode_lengths],
        "episode_scores": [int(s) for s in episode_scores],
        "alpha_history": [a.tolist() for a in alpha_history],
        "best_reward": float(best_reward),
        "best_score": int(best_score),
        "total_episodes": num_episodes,
        "total_time_seconds": time.time() - start_time,
        "device": str(device),
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_dir / "logs" / "training_log.json", "w") as f:
        json.dump(log_data, f, indent=2)

    if alpha_history:
        alpha_array = np.array(alpha_history)
        alpha_mean = alpha_array.mean(axis=0)
        alpha_std = alpha_array.std(axis=0)

        print_section("训练完成")
        print(f"总时间: {(time.time() - start_time)/60:.1f}分钟")
        print(f"最佳奖励: {best_reward:.2f}")
        print(f"最佳分数: {best_score:.0f}")
        print(f"平均奖励: {np.mean(episode_rewards):.2f}")
        print(f"平均分数: {np.mean(episode_scores):.0f}")
        print(f"\nα权重分布:")
        print(f"  α_pre:    {alpha_mean[0]:.3f} ± {alpha_std[0]:.3f}")
        print(f"  α_scene:  {alpha_mean[1]:.3f} ± {alpha_std[1]:.3f}")
        print(f"  α_effect: {alpha_mean[2]:.3f} ± {alpha_std[2]:.3f}")
        print(f"  α_rule:   {alpha_mean[3]:.3f} ± {alpha_std[3]:.3f}")
        print(f"\n检查点: {output_dir}/checkpoints/")
        print(f"日志: {output_dir}/logs/training_log.json")


if __name__ == "__main__":
    main()
