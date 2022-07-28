import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.modules
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from brian2 import *

def readData(batch_size = 64):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = datasets.MNIST(root='../dataset/mnist/',
                                   train=True,
                                   download=True,
                                   transform=transform)
    train_loader = DataLoader(train_dataset,
                              shuffle=True,
                              batch_size=batch_size)
    test_dataset = datasets.MNIST(root='../dataset/mnist/',
                                  train=False,
                                  download=True,
                                  transform=transform)
    test_loader = DataLoader(test_dataset,
                             shuffle=True,
                             batch_size=batch_size)
    return train_loader, test_loader

def getTh_binary(weight, test_loader, samples_num = 100):
    indices = array([])
    times = array([]) * ms
    inp = SpikeGeneratorGroup(784, indices, times)
    G = NeuronGroup(10, eqs, threshold='v>5', reset='v = 0', method='exact')
    S = Synapses(inp, G, 'w : 1', on_pre='v_post += w')
    S.connect(True)
    S.w = weight[S.i, S.j]
    spikemon = SpikeMonitor(G)

    pre_cnt = 0
    for i, data in enumerate(test_loader):
        if i >= samples_num: break
        img, label = data
        img = torch.where(img > 0.1307, torch.tensor([1.]), torch.tensor([0.]))
        tmp = img.numpy().reshape(784, 1).squeeze()
        indices = np.nonzero(tmp)[0]
        times = array([0 for _ in range(len(indices))]) * ms
        inp.set_spikes(indices, times + i * ms)
        run(1 * ms)
        spikes_num = sum(spikemon.count) - pre_cnt
        print('%d neurons firing a spike' % spikes_num)
        pre_cnt = sum(spikemon.count)
        G.v = 0
    return sum(spikemon.count)/samples_num * 5

def mySNN_binary(weight, test_loader, threshold = 4.5, test_num = 100):
    indices = array([])
    times = array([]) * ms
    inp = SpikeGeneratorGroup(784, indices, times)
    G = NeuronGroup(10, eqs, threshold='v>{}'.format(threshold), reset='v = 0', method='exact')
    S = Synapses(inp, G, 'w : 1', on_pre='v_post += w')
    S.connect(True)
    S.w = weight[S.i, S.j]
    # statemon = StateMonitor(G, 'v', record=True)
    spikemon = SpikeMonitor(G)

    running_time = []
    errors = 0
    pre_cnt = 0
    for i, data in enumerate(test_loader):
        if i >= test_num: break
        img, label = data
        img = torch.where(img > 0.1307, torch.tensor([1.]), torch.tensor([0.]))
        tmp = img.numpy().reshape(784, 1).squeeze()
        indices = np.nonzero(tmp)[0]
        times = array([0 for _ in range(len(indices))]) * ms
        start = time.time()
        inp.set_spikes(indices, times + i * ms)
        # defaultclock.dt = 0.01
        run(1 * ms)
        end = time.time()
        spikes_num = sum(spikemon.count) - pre_cnt
        print('%d neurons firing a spike' % spikes_num)
        if spikes_num != 1:
            print('recognition is wrong!!!')
            errors += 1
        else:
            if spikemon.i[-1] == label.numpy()[0]:
                print('recognition is right!!!')
            else:
                print('recognition is wrong!!!')
                errors += 1
        print('computing time: ', end - start)
        running_time.append(end - start)
        pre_cnt = sum(spikemon.count)
        G.v = 0
    print('max computing time:', max(running_time))
    print('min computing time:', min(running_time))
    print('average computing time: ', sum(running_time) / len(running_time))
    print('recognition accuracy: ', (test_num-errors) / test_num)

def mySNN_intensity(weight, test_loader, time_step, test_num):
    inp = PoissonGroup(784, 0 * Hz)
    G = NeuronGroup(10, eqs, threshold='v>6', reset='v = 0', method='exact')
    S = Synapses(inp, G, 'w : 1', on_pre='v_post += w')
    S.connect(True)
    S.w = weight[S.i, S.j]
    statemon = StateMonitor(G, 'v', record=True)
    spikemon = SpikeMonitor(G)

    running_time = []
    errors = 0
    pre_cnt = 0
    for i, data in enumerate(test_loader):
        if i >= test_num: break
        img, label = data
        tmp = img.numpy().reshape(784, 1).squeeze()
        start = time.time()
        inp.rates = tmp * time_step * Hz
        # defaultclock.dt = 0.01
        run(1 * second)
        end = time.time()
        spikes_num = sum(spikemon.count) - pre_cnt
        print('%d neurons firing a spike' % spikes_num)
        if spikes_num != 1:
            print('recognition is wrong!!!')
            errors += 1
        else:
            if spikemon.i[-1] == label.numpy()[0]:
                print('recognition is right!!!')
            else:
                print('recognition is wrong!!!')
                errors += 1
        print('computing time: ', end - start)
        running_time.append(end - start)
        pre_cnt = sum(spikemon.count)
        G.v = 0
    print('max computing time:', max(running_time))
    print('min computing time:', min(running_time))
    print('average computing time: ', sum(running_time) / len(running_time))
    print('recognition accuracy: ', (test_num-errors) / test_num)
    # plt.plot(statemon.t / second, statemon.v[1])
    # plt.show()

if __name__ == '__main__':
    eqs = '''
        v : 1
        '''
    # tau = 5 * ms
    # eqs = '''
    #     dv/dt = (-v)/tau : 1
    #     '''
    train_loader, test_loader = readData(1)
    PATH = 'E:/dvs_images/3'

    file1 = torch.load('weights_binary.pt', map_location='cpu')
    weight1 = np.float32(file1[list(file1.keys())[0]].t_().numpy())
    threshold = getTh_binary(weight1, test_loader, 100)
    print('The adaptive threshold is set to:', threshold)
    mySNN_binary(weight1, test_loader, threshold, 500)

    # time_step = 1000
    # file2 = torch.load('weights_intensity.pt', map_location='cpu')
    # weight2 = np.float32(file2[list(file2.keys())[0]].t_().numpy()) / time_step
    # mySNN_intensity(weight2, test_loader, time_step, 500)