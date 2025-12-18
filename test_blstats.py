#!/usr/bin/env python3
"""快速测试脚本：验证 blstats 索引是否正确"""

import gymnasium as gym
import nle.env
import nle.nethack as nh

print("=" * 80)
print("测试 blstats 索引正确性")
print("=" * 80)

env = gym.make("NetHackScore-v0")
obs, info = env.reset()

blstats = obs["blstats"]
print(f"\nblstats 长度: {len(blstats)}")
print(f"\n原始 blstats 向量:")
print(blstats)

print(f"\n使用 NLE 官方常量解析:")
print(f"  HP (NLE_BL_HP={nh.NLE_BL_HP}):           {blstats[nh.NLE_BL_HP]}")
print(f"  HPMAX (NLE_BL_HPMAX={nh.NLE_BL_HPMAX}):     {blstats[nh.NLE_BL_HPMAX]}")
print(f"  DEPTH (NLE_BL_DEPTH={nh.NLE_BL_DEPTH}):     {blstats[nh.NLE_BL_DEPTH]}")
print(f"  GOLD (NLE_BL_GOLD={nh.NLE_BL_GOLD}):      {blstats[nh.NLE_BL_GOLD]}")
print(f"  SCORE (NLE_BL_SCORE={nh.NLE_BL_SCORE}):     {blstats[nh.NLE_BL_SCORE]}")
print(f"  HUNGER (NLE_BL_HUNGER={nh.NLE_BL_HUNGER}):   {blstats[nh.NLE_BL_HUNGER]}")
print(f"  AC (NLE_BL_AC={nh.NLE_BL_AC}):         {blstats[nh.NLE_BL_AC]}")
print(f"  EXP (NLE_BL_EXP={nh.NLE_BL_EXP}):        {blstats[nh.NLE_BL_EXP]}")
print(f"  X (NLE_BL_X={nh.NLE_BL_X}):           {blstats[nh.NLE_BL_X]}")
print(f"  Y (NLE_BL_Y={nh.NLE_BL_Y}):           {blstats[nh.NLE_BL_Y]}")

print(f"\n预期结果（开局）:")
print(f"  HP 应该 > 0 且 HP == HPMAX (满血)")
print(f"  DEPTH 应该 = 1 (第1层)")
print(f"  GOLD 应该 = 0 (无金币)")
print(f"  SCORE 应该 = 0 (0分)")

hp = blstats[nh.NLE_BL_HP]
hpmax = blstats[nh.NLE_BL_HPMAX]
depth = blstats[nh.NLE_BL_DEPTH]
gold = blstats[nh.NLE_BL_GOLD]
score = blstats[nh.NLE_BL_SCORE]

print(f"\n验证结果:")
if hp > 0 and hp == hpmax:
    print(f"  ✅ HP 正确: {hp}/{hpmax} (100%)")
else:
    print(f"  ❌ HP 异常: {hp}/{hpmax}")

if depth == 1:
    print(f"  ✅ DEPTH 正确: {depth}层")
else:
    print(f"  ❌ DEPTH 异常: {depth}层 (应该是1层)")

if gold == 0:
    print(f"  ✅ GOLD 正确: {gold}")
else:
    print(f"  ⚠️  GOLD 非零: {gold} (可能拾取了金币)")

if score == 0:
    print(f"  ✅ SCORE 正确: {score}")
else:
    print(f"  ⚠️  SCORE 非零: {score}")

print("\n" + "=" * 80)
if hp > 0 and hp == hpmax and depth == 1:
    print("✅ blstats 索引验证通过！")
else:
    print("❌ blstats 索引可能有问题！")
print("=" * 80)

env.close()
