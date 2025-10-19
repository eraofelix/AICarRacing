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
    # Sample a feasible starting state by rejection
    while True:
        target_sum = rng.randint(12, 21)
        dealer_show = rng.randint(1, 10)
        ua = bool(rng.randint(0, 1))
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

    s = (hand_value(player), dealer[0], usable_ace(player))
    a = rng.choice([0, 1])

    episode_sa: List[Tuple[State, Action]] = []
    episode_sa.append((s, a))

    # Execute first action
    if a == 1:
        player.append(draw_card(rng))
        if is_bust(player):
            return episode_sa, -1.0
    else:
        dealer_policy(dealer, rng)
        if is_bust(dealer):
            return episode_sa, +1.0
        pv, dv = hand_value(player), hand_value(dealer)
        return episode_sa, float(1 if pv > dv else -1 if pv < dv else 0)

    # Continue with ε-greedy policy
    while True:
        s = (hand_value(player), dealer[0], usable_ace(player))
        a = agent.policy(s)
        episode_sa.append((s, a))
        if a == 1:
            player.append(draw_card(rng))
            if is_bust(player):
                return episode_sa, -1.0
        else:
            dealer_policy(dealer, rng)
            if is_bust(dealer):
                return episode_sa, +1.0
            pv, dv = hand_value(player), hand_value(dealer)
            return episode_sa, float(1 if pv > dv else -1 if pv < dv else 0)

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

    for ep in range(1, num_episodes + 1):
        episode_sa, G = play_one_episode_with_es(agent, rng)
        print(f"ep: {ep}, episode_sa: {episode_sa}, G: {G}")
        rewards_window.append(G)
        if ep % 100000 == 0:
            print(f"Episode {ep}/{num_episodes}, distinct (s,a): {len(agent.returns_count)}, avg visits: {avg_visits:.2f}")
        visited = set()
        for (s, a) in episode_sa:
            if (s, a) in visited:
                continue
            visited.add((s, a))
            agent.returns_count[(s, a)] += 1
            n = agent.returns_count[(s, a)]
            q_old = agent.Q[s][a]
            agent.Q[s][a] = q_old + (G - q_old) / n

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
    def render(ua: bool):
        print(f"Policy (usable_ace={ua})")
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
    NUM_EPISODES = 1000_000
    EPSILON = 0.1
    SEED = 2025
    LOG_EVERY = 10_000
    RECENT_WINDOW = 50_000

    Q, visits = mc_control_exploring_starts(
        num_episodes=NUM_EPISODES,
        epsilon=EPSILON,
        seed=SEED,
        log_every=LOG_EVERY,
        recent_window=RECENT_WINDOW,
        verbose=True
    )

    policy = derive_greedy_policy(Q)
    print_policy(policy)

# Policy (usable_ace=False)  30w episodes
# PS\DS |  1  2  3  4  5  6  7  8  9 10
# -------------------------------------
# 21  |  S  S  S  S  S  S  S  S  S  S
# 20  |  S  S  S  S  S  S  S  S  S  S
# 19  |  S  S  S  S  S  S  S  S  S  S
# 18  |  S  S  S  S  S  S  S  S  S  S
# 17  |  S  S  S  S  S  S  S  S  S  S
# 16  |  H  S  S  S  S  S  H  H  S  S
# 15  |  H  S  S  S  S  S  H  H  H  S
# 14  |  H  S  S  S  S  S  H  H  H  H
# 13  |  H  S  H  S  S  S  H  H  H  H
# 12  |  H  H  H  S  S  S  H  H  H  H
# Policy (usable_ace=True)
# PS\DS |  1  2  3  4  5  6  7  8  9 10
# -------------------------------------
# 21  |  S  S  S  S  S  S  S  S  S  S
# 20  |  S  S  S  S  S  S  S  S  S  S
# 19  |  S  S  S  S  S  S  S  S  S  S
# 18  |  H  S  S  S  S  S  S  S  H  S
# 17  |  H  H  H  H  H  H  H  H  H  H
# 16  |  H  H  H  H  H  H  H  H  H  H
# 15  |  H  H  H  H  H  H  H  H  H  H
# 14  |  H  H  H  H  H  H  H  H  H  H
# 13  |  H  H  H  H  H  H  H  H  H  H
# 12  |  H  H  H  H  H  H  H  H  H  H


# Policy (usable_ace=False)
# PS\DS |  1  2  3  4  5  6  7  8  9 10
# -------------------------------------
# 21  |  S  S  S  S  S  S  S  S  S  S
# 20  |  S  S  S  S  S  S  S  S  S  S
# 19  |  S  S  S  S  S  S  S  S  S  S
# 18  |  S  S  S  S  S  S  S  S  S  S
# 17  |  S  S  S  S  S  S  S  S  S  S
# 16  |  H  S  S  S  S  S  H  H  S  S
# 15  |  H  S  S  S  S  S  H  H  H  S
# 14  |  H  S  S  S  S  S  H  H  H  H
# 13  |  H  S  S  S  S  S  H  H  H  H
# 12  |  H  S  H  S  S  S  H  H  H  H
# Policy (usable_ace=True)
# PS\DS |  1  2  3  4  5  6  7  8  9 10
# -------------------------------------
# 21  |  S  S  S  S  S  S  S  S  S  S
# 20  |  S  S  S  S  S  S  S  S  S  S
# 19  |  S  S  S  S  S  S  S  S  S  S
# 18  |  H  S  S  S  S  S  S  S  H  S
# 17  |  H  H  H  H  H  H  H  H  H  H
# 16  |  H  H  H  H  H  H  H  H  H  H
# 15  |  H  H  H  H  H  H  H  H  H  H
# 14  |  H  H  H  H  H  H  H  H  H  H
# 13  |  H  H  H  H  H  H  H  H  H  H
# 12  |  H  H  H  H  H  H  H  H  H  H