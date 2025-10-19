import random
import time
from collections import defaultdict, deque
from typing import Tuple, List, Dict
import pdb

def draw_card(rng: random.Random) -> int:
    card = rng.randint(1, 13)
    return min(card, 10)

def draw_hand(rng: random.Random) -> List[int]:
    return [draw_card(rng), draw_card(rng)]

def usable_ace(hand: List[int]) -> bool:
    return 1 in hand and sum(hand) + 10 <= 21

def hand_value(hand: List[int]) -> int:
    total = sum(hand)
    if usable_ace(hand):
        return total + 10
    return total

def is_bust(hand: List[int]) -> bool:
    return hand_value(hand) > 21

def dealer_policy(dealer: List[int], rng: random.Random) -> List[int]:
    while hand_value(dealer) < 17:
        dealer.append(draw_card(rng))
    return dealer

State = Tuple[int, int, bool]
Action = int  # 0=stick, 1=hit

class MCAgentES:
    def __init__(self, seed: int = 0, epsilon: float = 0.1):
        self.rng = random.Random(seed)
        self.Q: Dict[State, Dict[Action, float]] = defaultdict(lambda: {0: 0.0, 1: 0.0})
        self.returns_count: Dict[Tuple[State, Action], int] = defaultdict(int)
        self.epsilon = epsilon

    def policy(self, s: State) -> Action:
        if self.rng.random() < self.epsilon:
            return self.rng.choice([0, 1])
        q0, q1 = self.Q[s][0], self.Q[s][1]
        return 0 if q0 >= q1 else 1

def play_one_episode_with_es(agent: MCAgentES, rng: random.Random) -> Tuple[List[Tuple[State, Action]], float]:
    # 探索开始：随机选择初始状态-动作对
    print("🎲 开始新的episode - 探索开始模式")
    while True:
        target_sum = rng.randint(12, 21)
        dealer_show = rng.randint(1, 10)
        ua = bool(rng.randint(0, 1))
        print(f"🎯 目标状态: 玩家点数={target_sum}, 庄家明牌={dealer_show}, 软A={ua}")
        
        # 生成可行的初始状态
        success = False
        for _ in range(5000):
            player = draw_hand(rng)
            for _ in range(rng.randint(0, 3)):
                if hand_value(player) < 21:
                    player.append(draw_card(rng))
            if 12 <= hand_value(player) <= 21 and usable_ace(player) == ua and hand_value(player) == target_sum:
                success = True
                break
        if not success:
            print("❌ 无法生成目标玩家手牌，重新尝试...")
            continue
            
        # 生成可行的庄家牌
        for _ in range(5000):
            dealer = draw_hand(rng)
            if dealer[0] == dealer_show or dealer[1] == dealer_show:
                if dealer[0] != dealer_show:
                    dealer = [dealer[1], dealer[0]]
                break
            else:
                continue
        break
    
    print(f"✅ 成功生成初始状态: 玩家={player}({hand_value(player)}), 庄家={dealer}({dealer[0]}明牌)")

    s = (hand_value(player), dealer[0], usable_ace(player))
    a = rng.choice([0, 1])
    action_name = "停牌" if a == 0 else "要牌"
    print(f"🎯 探索开始动作: {action_name} (随机选择)")

    episode_sa: List[Tuple[State, Action]] = []
    episode_sa.append((s, a))

    # 执行第一个动作
    print(f"🃏 执行动作: {action_name}")
    if a == 1:
        player.append(draw_card(rng))
        print(f"📈 要牌后: 玩家={player}({hand_value(player)})")
        if is_bust(player):
            print("💥 玩家爆牌！游戏结束，奖励=-1")
            return episode_sa, -1.0
    else:
        print("🛑 玩家停牌，庄家开始行动...")
        dealer_policy(dealer, rng)
        print(f"🏦 庄家最终手牌: {dealer}({hand_value(dealer)})")
        if is_bust(dealer):
            print("🎉 庄家爆牌！玩家获胜，奖励=+1")
            return episode_sa, +1.0
        pv, dv = hand_value(player), hand_value(dealer)
        result = 1 if pv > dv else -1 if pv < dv else 0
        result_text = "玩家获胜" if result > 0 else "庄家获胜" if result < 0 else "平局"
        print(f"🏆 最终结果: 玩家={pv}, 庄家={dv}, {result_text}, 奖励={result}")
        return episode_sa, float(result)

    # 继续使用ε-贪婪策略
    print("🔄 继续游戏，使用ε-贪婪策略...")
    while True:
        s = (hand_value(player), dealer[0], usable_ace(player))
        a = agent.policy(s)
        action_name = "停牌" if a == 0 else "要牌"
        print(f"🤖 智能体决策: 状态={s}, 动作={action_name}")
        episode_sa.append((s, a))
        if a == 1:
            player.append(draw_card(rng))
            print(f"📈 要牌后: 玩家={player}({hand_value(player)})")
            if is_bust(player):
                print("💥 玩家爆牌！游戏结束，奖励=-1")
                return episode_sa, -1.0
        else:
            print("🛑 玩家停牌，庄家开始行动...")
            dealer_policy(dealer, rng)
            print(f"🏦 庄家最终手牌: {dealer}({hand_value(dealer)})")
            if is_bust(dealer):
                print("🎉 庄家爆牌！玩家获胜，奖励=+1")
                return episode_sa, +1.0
            pv, dv = hand_value(player), hand_value(dealer)
            result = 1 if pv > dv else -1 if pv < dv else 0
            result_text = "玩家获胜" if result > 0 else "庄家获胜" if result < 0 else "平局"
            print(f"🏆 最终结果: 玩家={pv}, 庄家={dv}, {result_text}, 奖励={result}")
            return episode_sa, float(result)

def derive_greedy_policy(Q: Dict[State, Dict[Action, float]]) -> Dict[State, Action]:
    policy = {}
    for s, q in Q.items():
        policy[s] = 0 if q[0] >= q[1] else 1
    return policy

def policy_diff_rate(pi_a: Dict[State, Action], pi_b: Dict[State, Action]) -> float:
    if not pi_a and not pi_b:
        return 0.0
    keys = set(pi_a.keys()) | set(pi_b.keys())
    if not keys:
        return 0.0
    diff = sum(1 for k in keys if pi_a.get(k) != pi_b.get(k))
    return diff / len(keys)

# -------------------------
# MC Control with ES + progress
# -------------------------

def mc_control_exploring_starts(
    num_episodes: int = 500_000,
    epsilon: float = 0.1,
    seed: int = 123,
    log_every: int = 10_000,
    recent_window: int = 50_000,
    verbose: bool = True
) -> Tuple[Dict[State, Dict[Action, float]], Dict[State, int]]:
    rng = random.Random(seed)
    agent = MCAgentES(seed=seed, epsilon=epsilon)

    # Rolling performance estimates
    rewards_window = deque(maxlen=recent_window)
    start_time = time.time()

    last_policy_snapshot: Dict[State, Action] = {}
    last_log_time = start_time

    print(f"🚀 开始蒙特卡洛探索开始训练，总episodes: {num_episodes:,}")
    print("=" * 60)
    
    for ep in range(1, num_episodes + 1):
        print(f"\n --------------------------📊 Episode {ep}/{num_episodes}--------------------------")
        episode_sa, G = play_one_episode_with_es(agent, rng)
        
        print(f"📈 Episode {ep} 结果: 奖励={G}, 状态-动作序列长度={len(episode_sa)}")
        
        rewards_window.append(G)
        if ep % 100000 == 0:
            avg_visits = sum(agent.returns_count.values()) / max(1, len(agent.returns_count))
            print(f"📊 Episode {ep}/{num_episodes}, 不同状态-动作对: {len(agent.returns_count)}, 平均访问次数: {avg_visits:.2f}")
        
        # 更新Q值表
        visited = set()
        for (s, a) in episode_sa:
            if (s, a) in visited:
                continue
            visited.add((s, a))
            agent.returns_count[(s, a)] += 1
            n = agent.returns_count[(s, a)]
            q_old = agent.Q[s][a]
            agent.Q[s][a] = q_old + (G - q_old) / n  # 增量平均，无需存储历史样本省内存
            
            print(f"🔄 更新Q值: 状态={s}, 动作={a}, 旧值={q_old:.3f}, 新值={agent.Q[s][a]:.3f}, 访问次数={n}")

        if verbose and (ep % log_every == 0 or ep == 1):
            now = time.time()
            elapsed = now - start_time
            since_last = now - last_log_time
            last_log_time = now

            # Stats
            distinct_sa = len(agent.returns_count)
            avg_visits = sum(agent.returns_count.values()) / max(1, distinct_sa)
            mean_recent = sum(rewards_window) / max(1, len(rewards_window))
            win_rate = (sum(1 for r in rewards_window if r > 0) / max(1, len(rewards_window)))
            lose_rate = (sum(1 for r in rewards_window if r < 0) / max(1, len(rewards_window)))
            draw_rate = 1.0 - win_rate - lose_rate

            # Policy stability
            current_policy = derive_greedy_policy(agent.Q)
            diff_rate = policy_diff_rate(current_policy, last_policy_snapshot)
            last_policy_snapshot = current_policy

            # Throughput
            eps_per_sec = log_every / max(1e-9, since_last) if ep > 1 else ep / max(1e-9, elapsed)
            eta = (num_episodes - ep) / max(1e-9, eps_per_sec)

            msg = (
                f"[{ep:,}/{num_episodes:,}] "
                f"elapsed={elapsed:,.1f}s, eta={eta:,.1f}s, "
                f"eps/s={eps_per_sec:,.0f}, "
                f"distinct(s,a)={distinct_sa:,}, avg_visits={avg_visits:.2f}, "
                f"recent_win={win_rate:.3f}, lose={lose_rate:.3f}, draw={draw_rate:.3f}, "
                f"policy_change={diff_rate:.3f}"
            )
            print(msg, flush=True)

    # Aggregate state visits
    state_visits: Dict[State, int] = defaultdict(int)
    for (s, a), c in agent.returns_count.items():
        state_visits[s] += c

    return agent.Q, state_visits

def print_policy(policy: Dict[State, Action]):
    print("\n🎯 学习到的最优策略:")
    print("=" * 50)
    print("说明: S=停牌(Stick), H=要牌(Hit)")
    print("PS=玩家点数, DS=庄家明牌")
    print()
    
    def render(ua: bool):
        ace_type = "有软A" if ua else "无软A"
        print(f"📋 策略表 (usable_ace={ua}) - {ace_type}")
        header = "PS\\DS | " + " ".join(f"{d:2d}" for d in range(1, 11))
        print(header)
        print("-" * len(header))
        for ps in range(21, 11, -1):
            row = [f"{ps:2d}  |"]
            for ds in range(1, 11):
                a = policy.get((ps, ds, ua), 0)
                row.append(" S" if a == 0 else " H")
            print(" ".join(row))
        print()

    render(False)
    render(True)

if __name__ == "__main__":
    print("🎮 21点游戏蒙特卡洛探索开始算法")
    print("=" * 60)
    print("算法说明:")
    print("1. 🎯 探索开始: 每个episode从随机状态-动作对开始")
    print("2. 🧠 ε-贪婪策略: 平衡探索和利用")
    print("3. 📊 蒙特卡洛方法: 通过大量采样估计价值函数")
    print("4. 🏆 目标: 学习最优的21点游戏策略")
    print("=" * 60)
    
    NUM_EPISODES = 300000
    EPSILON = 0.1
    SEED = 2025
    LOG_EVERY = 100  # 减少日志间隔用于演示
    RECENT_WINDOW = 50_000

    print(f"🔧 训练参数:")
    print(f"   - 总episodes: {NUM_EPISODES:,}")
    print(f"   - ε值: {EPSILON}")
    print(f"   - 随机种子: {SEED}")
    print(f"   - 日志间隔: {LOG_EVERY:,}")
    print()

    Q, visits = mc_control_exploring_starts(
        num_episodes=NUM_EPISODES,
        epsilon=EPSILON,
        seed=SEED,
        log_every=LOG_EVERY,
        recent_window=RECENT_WINDOW,
        verbose=True
    )

    print("\n🎉 训练完成！")
    print(f"📊 总共学习了 {len(Q)} 个状态的价值函数")
    print(f"📈 状态-动作对总访问次数: {sum(visits.values()):,}")
    
    policy = derive_greedy_policy(Q)
    print_policy(policy)
    
    print("\n💡 策略解读:")
    print("- 高点数(17-21)时通常停牌，避免爆牌")
    print("- 低点数(12-16)时根据庄家明牌决定要牌或停牌")
    print("- 有软A时策略更激进，因为A可以按1或11计算")
