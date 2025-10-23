from brian2 import *

# 시뮬레이션 단위 설정
start_scope()
duration = 100*ms  # 100ms 동안 시뮬레이션

# 뉴런 모델: Leaky Integrate-and-Fire
eqs = '''
dv/dt = (I - v) / (10*ms) : 1
I : 1  # 입력 전류
'''

# 뉴런 생성
G = NeuronGroup(1, eqs, threshold='v > 1', reset='v = 0', method='exact')
G.I = 1.5  # 일정한 전류 입력

# 모니터링
M = StateMonitor(G, 'v', record=True)
spikemon = SpikeMonitor(G)

# 시뮬레이션 실행
run(duration)

# 결과 시각화
plot(M.t/ms, M.v[0])
xlabel('시간 (ms)')
ylabel('막 전위 (v)')
title('LIF 뉴런의 스파이크 발생')
show()

print("스파이크 발생 시점 (ms):", spikemon.t/ms)