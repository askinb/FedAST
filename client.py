import logging
import torch
import torch.nn as nn
import torch.optim as optim
import utils.delay as delay
import numpy as np
from torch.utils.data import TensorDataset
import time


class Client(object):
    """Simulated federated learning client."""

    def __init__(self, client_id):
        self.client_id = client_id
        self.events_in_queue = 0
        self.last_finish_time = 0
    def __repr__(self):
        return 'Client #{}\n'.format(self.client_id) 
    
    # Set non-IID data configurations
    def set_bias(self, pref, bias):
        
        self.pref_list = pref
        self.bias_list = bias
    # Server interactions
    def download(self, argv): #For possible future works 
        # Download from the server.
        try:
            return argv.copy()
        except:
            return argv

    def upload(self, argv): #For possible future works 
        # Upload to the server
        try:
            return argv.copy()
        except:
            return argv
            
    def set_delay_generator(self, config, inv_speed =1):
        self.delayGenerator = delay.delay(config, inv_speed)

    def set_data(self, data_list, config):
        
        self.config = self.download(config)
        self.trainset_list = []
        self.testset_list = []
        # Extract from config
        do_test = self.do_test = config.clients.do_test
        test_partition = self.test_partition = config.clients.test_partition
        for i in range(config.models.__len__()):
            repeatedTask = False
            if i != 0:
                for sameModel_i, m in enumerate(config.models[:i]):
                    if config.models[i].name == m.name:
                        if config.models[i].data.data_seed == m.data.data_seed:
                            repeatedTask =True
                            self.trainset_list.append(self.trainset_list[sameModel_i])
                            self.testset_list.append(self.testset_list[sameModel_i]) 
            
            if not repeatedTask:
                # Download data
                data = self.download(data_list[i])
                
                # Extract trainset, testset (if applicable)
                if do_test:  # Partition for testset if applicable (not used in the study)
                    trainset = data[:int(len(data) * (1 - test_partition))] if not self.config.models[i].data.leaf else data[0]
                    testset = data[int(len(data) * (1 - test_partition)):] if not self.config.models[i].data.leaf else data[1]
                else:
                    trainset = data if not self.config.models[i].data.leaf else data[0]
                    testset = None
                    
                if config.models[i].data.trainset_device == 'cuda':
                    device = config.models[i].data.trainset_device
                    if config.models[i].name != 'Shakespeare':
                        for idx, (data, label) in enumerate(trainset):
                            trainset[idx] = (data.to(device), torch.tensor(label).to(device))
                    else:
                        x_train = np.array(trainset['x'], dtype=np.float32)
                        x_train = torch.LongTensor(x_train)
                        y_train = np.array(trainset['y'], dtype=np.int32)
                        y_train = torch.Tensor(y_train).type(torch.int64)
                        trainset = TensorDataset(x_train.to(device), y_train.to(device))          
                elif config.models[i].name == 'Shakespeare':
                    x_train = np.array(trainset['x'], dtype=np.float32)
                    x_train = torch.Tensor(x_train)
                    y_train = np.array(trainset['y'], dtype=np.int32)
                    y_train = torch.Tensor(y_train).type(torch.int64)
                
                    trainset = TensorDataset(x_train, y_train)
                
                self.trainset_list.append(trainset)
                self.testset_list.append(testset) 
            
    def generateDelay(self, config, modelIdx):

        k = round(config.models[modelIdx].delay.local_multi)
        
        return k*self.delayGenerator.generate(modelIdx)

        
    def configure(self, modelIdx, config, weights, fl_module):

        config = self.download(config)
        self.local_iter = round(config.models[modelIdx].local_iter)
        self.epochs = config.models[modelIdx].epochs
        self.batch_size = config.models[modelIdx].batch_size


    # Machine learning tasks
    def train(self, config, modelIdx, old_weights, fl_module, agg_weight):

        # logging.info('Training Model {} on client #{}'.format(1+modelIdx, self.client_id))
        model = fl_module.Net().to(config.models[modelIdx].model_device)
        fl_module.load_weights(model, old_weights)
        model.train()

        # Create optimizer
        optimizer = fl_module.get_optimizer(model,
                                            config.models[modelIdx].local_lr,
                                            config.models[modelIdx].local_momentum,
                                            config.models[modelIdx].weight_decay
                                            )
        
        # Perform model training
        trainloader = fl_module.get_trainloader(self.trainset_list[modelIdx], self.batch_size)
        model, tau, trainingLoss = fl_module.train(model, trainloader,
                       optimizer, local_iter = self.local_iter, epochs = self.epochs)

        # Extract model weights and biases
        new_weights = fl_module.extract_weights(model, config.models[modelIdx].model_device)

        
        # Calculate weighted delta and return to the asyncevent:
        delta = []
        for i, (name, w) in enumerate(new_weights):
            delta.append((name, agg_weight*(w-old_weights[i][1])))
        return delta, trainingLoss


    def test(self):
        # Perform local model testing - never used local testing
        raise NotImplementedError

