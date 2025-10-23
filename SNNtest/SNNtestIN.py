from brian2 import *

start_scope()

# 뉴런 모델
eqs = '''
dv/dt = (I - v) / (10*ms) : 1
I : 1
'''

# 입력 뉴런(1개), 출력 뉴런(1개)
input_neuron = NeuronGroup(1, 'I : 1', threshold='I > 1', reset='I = 0')
output_neuron = NeuronGroup(1, eqs, threshold='v > 1', reset='v = 0', method='exact')

# 시냅스 연결
S = Synapses(input_neuron, output_neuron, on_pre='v += 0.5')
S.connect()

# 입력 뉴런의 주기적 스파이크 생성
@network_operation(dt=10*ms)
def stimulate():
    input_neuron.I = 2

# 모니터링
M_out = StateMonitor(output_neuron, 'v', record=True)
spikes_in = SpikeMonitor(input_neuron)
spikes_out = SpikeMonitor(output_neuron)

run(200*ms)

plot(M_out.t/ms, M_out.v[0])
xlabel('시간 (ms)')
ylabel('출력 뉴런 막전위')
title('입력 스파이크 → 출력 뉴런 발화')
show()