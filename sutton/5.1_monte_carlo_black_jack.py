import random
from collections import defaultdict

def draw():
    # 均匀抽牌：2..9, 10(四种) 和A(1)
    x = random.randint(1,13)
    return 11 if x==1 else (10 if x>=10 else x)  # A先当11，后续再调整

def usable_adjust(total, aces_as_11):
    # 若爆则将A(11)降为1，直到不爆或没有可降的A
    while total > 21 and aces_as_11:
        total -= 10
        aces_as_11 -= 1
    return total, aces_as_11

def hand_value(cards):  # 计算21点游戏中手牌价值的核心函数
    total = sum(cards)
    aces_as_11 = sum(c==11 for c in cards)
    total, aces_as_11 = usable_adjust(total, aces_as_11)
    usable = aces_as_11 > 0  # 仍有A按11计
    return total, usable

def policy(player_sum):
    # 只在20或21停牌，其余要牌
    return player_sum >= 20

def play_episode():
    # 发初始两张
    player = [draw(), draw()]
    dealer = [draw(), draw()]
    # 调整A
    psum, puse = hand_value(player)
    dsum, duse = hand_value(dealer)
    dealer_up = dealer[0] if dealer[0] != 11 else 1  # 观测A显示为1更直观

    episode = []  # 存储(state, reward暂空)
    # 玩家回合
    while True:
        state = (psum, puse, dealer_up if dealer_up<=10 else 10)
        episode.append(state)
        if policy(psum):  # 停牌
            break
        # 要牌
        player.append(draw())
        psum, puse = hand_value(player)
        if psum > 21:  # 爆了
            return episode, -1

    # 庄家回合：规则为17点及以上停（软17也停）
    while dsum < 17:
        dealer.append(draw())
        dsum, duse = hand_value(dealer)

    # 结算
    if dsum > 21: reward = 1
    elif psum > dsum: reward = 1
    elif psum < dsum: reward = -1
    else: reward = 0

    return episode, reward

def mc_evaluate(n_episodes=200000, every_visit=True):
    V = defaultdict(float)
    N = defaultdict(int)
    for _ in range(n_episodes):
        episode, G = play_episode()
        seen = set()
        for s in episode:
            if every_visit or s not in seen:
                N[s] += 1
                V[s] += (G - V[s]) / N[s]  # 增量均值
                seen.add(s)
    return V, N

if __name__ == "__main__":
    V, N = mc_evaluate(200000, every_visit=True)
    # 示例：打印部分状态价值
    for state in sorted(V.keys())[:20]:
        print(state, round(V[state], 3), N[state])