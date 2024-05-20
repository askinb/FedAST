import client
import load_data
import logging
import importlib
import numpy as np
import time
import math
import random
import sys
from threading import Thread
import torch
import utils.dists as dists  
from queue import PriorityQueue
from utils.asyncEvent import asyncEvent
from threading import Thread
import copy
from torch.utils.data import DataLoader, Subset
import torch.nn as nn

class Server(object):
    """Multimodel federated learning server."""
    def __init__(self, config):
        self.config = config
        self.total_clients = self.config.clients.total
        self.nb_of_models = self.config.models.__len__()
        if self.config.wandb:
            self.wandb =  __import__('wandb')
            self.wandb.init(project=config.wandb_project_name,name=config.wandb_runname+'_'+config.algo+'_clients'+\
                            str(config.clients.total))
            distInfos = []
            locItInfos = []
            for i in range(len(config.models)):
                if self.config.models[i].data.bias:
                    temp = '_bias_'+str(config.models[i].data.bias['primary'])+'_'+str(config.models[i].data.bias['secondary'])
                elif self.config.models[i].data.IID:
                    temp = '_IID_'
                elif self.config.models[i].data.dirichlet:
                    temp = '_drchlt_'+str(config.models[i].data.dirichlet['alpha'])
                else:
                    temp = '_noDistInfo'
                distInfos.append(temp)
                if config.models[i].epochs:
                    locItInfos.append('_epochs'+str(config.models[i].local_iter))
                else:
                    locItInfos.append('_locit'+str(config.models[i].local_iter))
            if config.algo == 'fedast':
                self.model_run_names = ['m'+str(i)+'_'+config.models[i].name+'_batch'+str(config.models[i].batch_size)\
                               +locItInfos[i]+'_buff'+str(config.models[i].buffer_size)+'_active'\
                              +str(config.models[i].nb_of_active_works)+'_locLR'+str(config.models[i].local_lr)\
                              +'_gLR'+str(config.models[i].global_lr)+'_'+'wd'+str(config.models[i].weight_decay)\
                              +distInfos[i]\
                              +'seed'+str(config.models[i].data.data_seed)+'_'\
                                  for i in range(self.nb_of_models)]
            elif config.algo == 'sync' or config.algo == 'sync-bods' or config.algo == 'sync-ucb':
                self.model_run_names = ['m'+str(i)+'_'+config.models[i].name+'_batch'+str(config.models[i].batch_size)\
                               +locItInfos[i]+'_locLR'+str(config.models[i].local_lr)+'_active'\
                              +str(config.models[i].nb_of_active_works)+'_gLR'+str(config.models[i].global_lr)+'_'\
                              +'wd'+str(config.models[i].weight_decay)+'_actWrk'+str(config.partialNbofClients)+'_'\
                              +'firstC'+ str(config.models[i].firstC) +'_'+distInfos[i]\
                              +'seed'+str(config.models[i].data.data_seed)+'_'\
                                  for i in range(self.nb_of_models)]
            [self.wandb.define_metric('acc_'+self.model_run_names[i], step_metric='time_'+\
                self.model_run_names[i]) for i in range(self.nb_of_models)]
            [self.wandb.define_metric('loss_'+self.model_run_names[i], step_metric='time_'+\
                self.model_run_names[i]) for i in range(self.nb_of_models)]
            [self.wandb.define_metric('meanLocIt_'+self.model_run_names[i], step_metric='time_'+\
                self.model_run_names[i]) for i in range(self.nb_of_models)]
            self.wandb.define_metric('tasksAvgAcc', step_metric='time')
            self.wandb.define_metric('tasksAvgLoss', step_metric='time')
            [self.wandb.define_metric('globalRound_m'+str(i), step_metric='time') for i in range(self.nb_of_models)]
            [self.wandb.define_metric('clientTrips_m'+str(i), step_metric='time') for i in range(self.nb_of_models)]
            self.wandb.define_metric('totalClientTrips', step_metric='time')
            self.wandb_dict = {'clientTrips_m'+str(i):0 for i in range(self.nb_of_models)}
            self.wandb_dict['totalClientTrips'] = 0
    # Set up server
    def boot(self):
        logging.info('Booting the server...')
        for i in range(self.nb_of_models):
            sys.path.append(self.config.models[i].model_path)
        sys.path.append('~/models')
        self.fl_modules = list()
        # Import seperate model modules
        for i in range(self.nb_of_models):
            self.fl_modules.append(getattr(__import__(self.config.models[i].name+'.fl_model'), 'fl_model'))
        

        # Set up simulated server
        self.load_data()
        self.load_model()
        self.make_clients(self.total_clients)
        self.stalenessTemp =[[] for i in range(self.nb_of_models)] #For developing purpose
        self.R_bs_ratio = self.config.R_bs_ratio

        self.R_total = sum([j.nb_of_active_works for j in self.config.models])
        self.full_async = self.config.full_async


        self.varBasedR = self.config.varBasedR
        if self.varBasedR:
            self.varEstModels = [[] for _ in range(self.nb_of_models)]
            self.varEstWindowSizes = [8 for _ in range(self.nb_of_models)]
            self.varUpdatePeriodsC = sum([j.nb_of_active_works for j in self.config.models])*3//4
            self.varUpdatePeriods = self.varUpdatePeriodsC * len([1 for j in self.config.models if j.nb_of_active_works>0])
            logging.info("varUpdatePeriods: "+str(self.varUpdatePeriods))
            self.varUpdatePeriodsCounter = 0
            self.varVariances = [[] for _ in range(self.nb_of_models)]
        self.howManyActiveWorkPerModel = []

        self.gradNormBased = False
        if self.gradNormBased:
            self.gradNorms = [[] for _ in range(self.nb_of_models)]
            self.gradNormWindowSizes = [10 for _ in range(self.nb_of_models)]

    def calculate_norm_of_model(self,update, norm_type=2):
        total_norm = 0.0
        total_params = 0
        model_i = update.modelIdx
        for layer_name, weight_tensor in update.delta:
            # Skip bias layers
            if 'bias' in layer_name:
                continue
            param_norm = torch.norm(weight_tensor, norm_type)
            total_norm += param_norm.item() ** norm_type
            total_params += weight_tensor.numel()
        avg_norm = (total_norm/total_params/self.config.models[model_i].local_lr**norm_type\
                    /self.config.models[model_i].delay.local_multi**norm_type)** (1. / norm_type)
        return avg_norm
    def redistributeActiveWorks(self, a): #a=variances
        R_total = R_to_dist = sum([self.config.models[i].nb_of_active_works for i in range(self.nb_of_models)])
        newRs = [self.R_bs_ratio for _ in self.config.models]
        for i,j in enumerate(self.config.models):
            if j.nb_of_active_works <= 0:
                a[i] = 0
        temp = [int(R_total*i/sum(a))-self.R_bs_ratio for i in a]
        for i,j in enumerate(self.config.models):
            if j.nb_of_active_works <= 0:
                newRs[i] = 0 
                temp[i] = -99999
        R_to_dist -= sum(newRs)
        while R_to_dist > 0:
            i = temp.index(max(temp))
            newRs[i] += self.R_bs_ratio
            temp[i] -= self.R_bs_ratio
            R_to_dist -= self.R_bs_ratio
        return newRs

    def calculateUpdVariance(self, model_i):
        import torch
        
        variance = 0
        num_of_params = 0
        if self.config.algo == 'fedast':
            deltas =self.varEstModels[model_i]
        else:
            if len(self.buffers[model_i]) == 0:
                return 0
            minbuffsize = min([len(j) for j in self.buffers if len(j)!=0])
            deltas = random.sample(self.buffers[model_i],minbuffsize)
        num_of_models = len(deltas)
        total_norm = 0.0
        
        # Iterate over each layer by index
        for layer_idx in range(len(deltas[0])):
            layer_name, _ = deltas[0][layer_idx]
        
            # Skip bias layers
            if 'bias' in layer_name:
                continue
        
            # Calculate the average weight for this layer across all models
            sum_weights = None
            for modelIdx in range(num_of_models):
                weight_tensor = deltas[modelIdx][layer_idx][1]
                if sum_weights is None:
                    sum_weights = weight_tensor.clone()
                else:
                    sum_weights += weight_tensor
        
            # Average weight for this layer
            avg_weight = sum_weights / num_of_models
            
            param_norm = torch.norm(avg_weight, 2)
            total_norm += param_norm.item() ** 2
            
            # Sum of squared distances from the average
            sum_squared_distance = 0
            for modelIdx in range(num_of_models):
                weight_tensor = deltas[modelIdx][layer_idx][1]
                sum_squared_distance += torch.sum((weight_tensor - avg_weight) ** 2)
        
            variance += sum_squared_distance
            num_of_params += avg_weight.numel()
        # Calculate average variance
        average_variance = self.config.models[model_i].local_lr*self.config.models[model_i].global_lr*\
        variance / (total_norm * num_of_models) / \
        (self.config.models[model_i].local_lr*self.config.models[model_i].delay.local_multi)**2

        return float(average_variance)**1/2
    def howManyActiveRequests(self):
        i = 0 
        for c in self.clients:
            i += c.events_in_queue
        return i
    def howManyActiveRequestsModel(self,modelIdx):
        i = 0 
        for elem in self.qEvents.queue:
            if elem.modelIdx == modelIdx:
                i+=1
        for elem in self.qUpdates.queue:
            if elem.modelIdx == modelIdx:
                i+=1
        return i

    def load_data(self):        

        # Extract config for loaders
        config = self.config
        self.data_rand_states = []
        
        # Set up data generators
        generators = []
        for i in range(self.nb_of_models):            
            if self.config.models[i].data.data_seed is not None:
                random.seed(self.config.models[i].data.data_seed)
            else:
                random.seed()
            generators.append(self.fl_modules[i].Generator())
            self.data_rand_states.append(random.getstate())
        
        self.loaders = list()
        self.testloader_list = list()
        
        for i in range(self.nb_of_models):
            temp = self.doesSameTaskExistBefore(modelId = i) #Efficiency for identical tasks
            if temp is not None:
                self.data_rand_states[i] = self.data_rand_states[temp] #dnm
                logging.info('The same task with the same seed is repeated at task {0}!'.format(temp))
                self.loaders.append(self.loaders[temp])
                logging.info('Model '+str(i+1)+': '+self.config.models[i].name+' | Loader: {}, IID: {}'.format(
                    self.config.models[i].data.loader, self.config.models[i].data.IID))
                self.testloader_list.append(self.testloader_list[temp])
            else:
                random.setstate(self.data_rand_states[i])
                # Generate data
                data_path = self.config.models[i].data_path
                args_temp = (data_path, config.models[i].data.testset_device) if\
                self.config.models[i].data.loader != 'leaf' else (data_path, self.total_clients)
                data = generators[i].generate(*args_temp)
                labels = generators[i].labels
                if not self.config.models[i].data.leaf:
                    logging.info('Model '+str(i+1)+': '+self.config.models[i].name+' | Dataset size: {}'.format(
                        sum([len(x) for x in [data[label] for label in labels]])))
                    logging.debug('Model '+str(i+1)+': '+self.config.models[i].name+' | Labels ({}): {}'.format(
                        len(labels), labels))
                else:
                    logging.info('Model '+str(i+1)+': '+self.config.models[i].name+' (LEAF dataset)')
                # Set up data loader
                loader = {
                    'basic': load_data.Loader(config, generators[i],i),
                    'bias': load_data.BiasLoader(config, generators[i],i),
                    'leaf': load_data.LEAFLoader(config, generators[i],i),
                    'dirichlet': load_data.DirichletLoader(config, generators[i],i)
                }[self.config.models[i].data.loader]
                self.loaders.append(loader)
    
                logging.info('Model '+str(i+1)+': '+self.config.models[i].name+' | Loader: {}, IID: {}'.format(
                    self.config.models[i].data.loader, self.config.models[i].data.IID))
                temp_args = (loader.get_testset(), self.config.models[i].test_batch_size,\
                             config.models[i].data.testset_device) if self.config.models[i].data.loader == 'leaf' else\
                            (loader.get_testset(), self.config.models[i].test_batch_size)
                self.testloader_list.append(self.fl_modules[i].get_testloader(*temp_args))
                self.data_rand_states[i] = random.getstate()

    
    def load_model(self):
        # Set up global models
        self.models = list()
        for i in range(self.nb_of_models):
            logging.info('Model '+str(i+1)+': '+self.config.models[i].name)
            self.models.append(self.fl_modules[i].Net().to(self.config.models[i].model_device))
        if self.config.varBasedR:
            self.modelsPrevVersion = [copy.deepcopy(j) for j in self.models.copy()]


    def make_clients(self, num_clients):
        
        dist_list = list()
        for i in range(self.nb_of_models):
            random.setstate(self.data_rand_states[i])
            IID = self.config.models[i].data.IID
            loader = self.config.models[i].data.loader
            loading = self.config.models[i].data.loading
            
            labels = self.loaders[i].labels
            if not IID:  # Create distribution for label preferences if non-IID
                dist = {
                    "uniform": dists.uniform(num_clients, len(labels)),
                    "normal": dists.normal(num_clients, len(labels))
                }[self.config.models[i].data.label_distribution]
                random.shuffle(dist)  # Shuffle distribution
                dist_list.append(dist)
            self.data_rand_states[i] = random.getstate()
        # Make simulated clients
        clients = []
        for client_id in range(num_clients):
            IID = self.config.models[i].data.IID
            loader = self.config.models[i].data.loader
            loading = self.config.models[i].data.loading

            # Create new client
            new_client = client.Client(client_id)
            
            pref_list = list()
            bias_list = list()
            for i in range(self.nb_of_models):
                random.setstate(self.data_rand_states[i])
                if IID:
                    pref_list.append(None)
                    bias_list.append(None)
                else:  # Configure clients for non-IID data
                    if self.config.models[i].data.bias:
                        # Bias data partitions
                        # Choose weighted random preference
    
                        pref_list.append(random.choices(self.loaders[i].labels, dist_list[i])[0])
                        bias_list.append(self.config.models[i].data.bias)
                        
                    elif self.config.models[i].data.leaf or self.config.models[i].data.dirichlet:
                        pref_list.append(None)
                        bias_list.append(None)
                self.data_rand_states[i] = random.getstate()
            # Assign preference, bias config
            new_client.set_bias(pref_list, bias_list)

            clients.append(new_client)

        logging.info('Total clients: {}'.format(len(clients)))

        for i in range(self.nb_of_models):
            random.setstate(self.data_rand_states[i])
            IID = self.config.models[i].data.IID
            loader = self.config.models[i].data.loader
            loading = self.config.models[i].data.loading
            labels = self.loaders[i].labels
            if loader == 'bias':
                logging.info('Model '+str(i+1)+': Label distribution: {}'.format(
                    [[client.pref_list[i] for client in clients].count(label) for label in labels]))
              
            self.data_rand_states[i] = random.getstate()
        # Send data partition to all clients
        [self.set_client_data(client) for client in clients]
        self.clients = clients
        # Create delay generators of clients
        self.createClientDelays()
    def createClientDelays(self):
        random.seed(64) #To standardize experiments, note that this doesn't affect randomness as we have another seed for data dist.
        indices = np.arange(len(self.clients))
        random.shuffle(indices)
        for i in indices[:int(self.config.clients.slow_ratio*len(self.clients))]:
            self.clients[i].set_delay_generator(self.config, inv_speed = 1.3)
        for i in indices[int(self.config.clients.slow_ratio*len(self.clients)):\
                    int((self.config.clients.slow_ratio+self.config.clients.normal_ratio)*len(self.clients))]:
            self.clients[i].set_delay_generator(self.config, inv_speed = 1)
        for i in indices[int((self.config.clients.slow_ratio+self.config.clients.normal_ratio)*len(self.clients)):]:
            self.clients[i].set_delay_generator(self.config, inv_speed = 0.7)
        random.seed()
    
    def train(self):
        wandbSummary = bool(self.config.wandb)
        if self.config.algo == 'fedast':
            self.async_run_fedast()
        elif self.config.algo == 'sync':
            self.sync_run()
        elif self.config.algo == 'sync-bods':
            self.sync_bods_run()
        elif self.config.algo == 'sync-ucb':
            self.sync_ucb_run()
        else:
            if self.config.wandb:
                self.wandb.finish()
                wandbSummary = False
        if wandbSummary:
            self.wandb.log({"Summary_Acc" : self.wandb.plot.line_series(
                       xs=[self.test_accuracies[j][0] for j in range(self.nb_of_models)], 
                       ys=[self.test_accuracies[j][2]  for j in range(self.nb_of_models)],
                       keys=["Task {0} {1}".format(j, self.config.models[j].name) for j in range(self.nb_of_models)],
                       title="All tasks summary acc",
                       xname="time")})
            self.wandb.log({"Summary_Loss" : self.wandb.plot.line_series(
                       xs=[self.test_accuracies[j][0] for j in range(self.nb_of_models)], 
                       ys=[self.test_accuracies[j][3]  for j in range(self.nb_of_models)],
                       keys=["Task {0} {1}".format(j, self.config.models[j].name) for j in range(self.nb_of_models)],
                       title="All tasks summary loss",
                       xname="time")})

    def sync_bods_run(self):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern
        from utils.bodsUtils import expected_improvement
        self.round_list = [0 for _ in range(self.nb_of_models)]
        self.model_version_dicts_list = [{'0':self.fl_modules[i].extract_weights(self.models[i],
                                                                                 self.config.models[i].model_device)}
                                         for i in range(self.nb_of_models)]
        self.test_accuracies = [[[],[],[],[]] for _ in range(self.nb_of_models)]
        self.buffers = [[] for _ in range(self.nb_of_models)]
        self.T = 0 #Time
        self.is_finished_list = [False for _ in range(self.nb_of_models)]
        
        self.target_rounds_list = []
        self.target_accuracy_list = []
        
        for i in range(self.nb_of_models):
            self.target_rounds_list.append(self.config.models[i].rounds)
            self.target_accuracy_list.append(self.config.models[i].target_accuracy)
            if self.config.models[i].target_accuracy:
                logging.info('Model '+str(i+1)+': Training: {} rounds or {}% accuracy\n'.format(
                    self.config.models[i].rounds, 100 * self.config.models[i].target_accuracy))
            else:
                logging.info('Model '+str(i+1)+': Training: {} rounds\n'.format(self.config.models[i].rounds))

        self.qEvents = PriorityQueue()
        
        # BODS
        self.gp = GaussianProcessRegressor(kernel=Matern())
        self.alpha = self.config.bods_alpha
        self.beta = 1-self.alpha
        self.clientSelectionsList = [np.zeros(self.total_clients) for _ in range(self.nb_of_models)]
        def calcFairnessCost(A):
            return sum([(np.std(A[i]))**2 for i in range(self.nb_of_models)])

        #Normalize number of active clients
        self.activeClients =self.config.partialNbofClients if self.config.partialNbofClients else self.total_clients
        self.howManyClients = [max(min(self.config.models[i].nb_of_active_works,1),int(self.activeClients*self.config.models[i].nb_of_active_works/sum([self.config.models[ii].nb_of_active_works for ii in range(self.nb_of_models)]))) for i in range(self.nb_of_models)]
        
        for i in range(self.nb_of_models):
            self.config.models[i] = self.config.models[i]._replace(nb_of_active_works =self.howManyClients[i] )
        
        while not np.all(self.is_finished_list) and self.T < self.config.max_time:
            self.howManyClients = [max(min(self.config.models[i].nb_of_active_works,1),int(self.activeClients*self.config.models[i].nb_of_active_works/sum([self.config.models[ii].nb_of_active_works for ii in range(self.nb_of_models)]))) for i in range(self.nb_of_models)]

            if self.activeClients > sum(self.howManyClients):
                for i in np.random.choice([j for j in np.arange(self.nb_of_models, dtype=int) if self.config.models[j].nb_of_active_works != 0], self.activeClients - sum(self.howManyClients), replace=True):
                    self.howManyClients[i] += 1
            self.firstC_list = []
            for i in range(self.nb_of_models):
                if self.config.models[i].firstC:
                    self.firstC_list.append(min(self.config.models[i].firstC, self.howManyClients[i]))
                else:
                    self.firstC_list.append(self.howManyClients[i])

            if self.round_list[0] == 0: #First round is random.
                temp = np.random.permutation(self.total_clients) #uniform partitioning of clients
                selectionVector = np.zeros((1,self.total_clients*self.nb_of_models))
                for i in range(self.nb_of_models):
                    if self.config.models[i].nb_of_active_works != 0:
                        for clientId in temp[sum(self.howManyClients[:i]):sum(self.howManyClients[:i+1])]:
                            self.qEvents.put(self.createNewJob(i,client = self.clients[clientId]))
                            selectionVector[0,i*self.total_clients+clientId] = 1
                            self.clientSelectionsList[i][clientId] += 1
                # delete events after the firstC from queue:
                self.qEvents.queue.sort()
                for i in range(self.qEvents.qsize()-1,-1,-1):
                    j = self.qEvents.queue[i].modelIdx
                    if [1 for e in self.qEvents.queue if e.modelIdx == j].__len__() > self.firstC_list[j]:
                        self.clientSelectionsList[self.qEvents.queue[i].modelIdx][self.qEvents.queue[i].client.client_id] -= 1
                        self.qEvents.queue.pop(i)
                self.bodsX = selectionVector
                self.bodsY = [self.alpha*max([e.workTime for e in self.qEvents.queue]) + self.beta*calcFairnessCost(self.clientSelectionsList)]
                self.bodsY = [self.alpha*sum([max([e.workTime for e in self.qEvents.queue if e.modelIdx == kkk]) for kkk in range(self.nb_of_models)]) + self.beta*calcFairnessCost(self.clientSelectionsList)]
                
            else:
                self.gp.fit(self.bodsX, self.bodsY)
                randomSelections = []
                temp = np.random.permutation(self.total_clients)[:sum(self.howManyClients)+5] #here we limited the available clients, 5 is just extra 
                for _ in range(75): # Sample 75 random points in the search space
                    np.random.shuffle(temp) # Sample possible random search points
                    selectionVector = np.zeros((1,self.total_clients*self.nb_of_models))
                    for i in range(self.nb_of_models):
                        if self.config.models[i].nb_of_active_works != 0:
                            for clientId in temp[sum(self.howManyClients[:i]):sum(self.howManyClients[:i+1])]:
                                selectionVector[0,i*self.total_clients+clientId] = 1
                    randomSelections.append(selectionVector)
                ei = -1 * expected_improvement(randomSelections, self.gp, self.bodsY, greater_is_better=False)
                selectionVector = randomSelections[np.argmax(ei)]
                temp1, temp2 = np.where(np.array(selectionVector).reshape(self.nb_of_models,self.total_clients)==1)
                for i in range(len(temp1)):
                    self.clientSelectionsList[temp1[i]][temp2[i]] += 1
                    self.qEvents.put(self.createNewJob(temp1[i],client = self.clients[temp2[i]]))
                # delete events after the firstC from queue:
                self.qEvents.queue.sort()
                for i in range(self.qEvents.qsize()-1,-1,-1):
                    j = self.qEvents.queue[i].modelIdx
                    if [1 for e in self.qEvents.queue if e.modelIdx == j].__len__() > self.firstC_list[j]:
                        self.clientSelectionsList[self.qEvents.queue[i].modelIdx][self.qEvents.queue[i].client.client_id] -= 1
                        self.qEvents.queue.pop(i)
                self.bodsX = np.vstack([self.bodsX, selectionVector])
                self.bodsY.append(self.alpha*sum([max([e.workTime for e in self.qEvents.queue if e.modelIdx == kkk]) for kkk in range(self.nb_of_models)]) + self.beta*calcFairnessCost(self.clientSelectionsList))    
            self.stalenessTemp =[[self.howManyClients[i],self.firstC_list[i]] for i in range(self.nb_of_models)] #For developing purpose
            while self.qEvents.qsize() != 0:
                self.threadedEvents = [self.qEvents.get() for _ in range(min(self.config.num_of_threads, self.qEvents.qsize()))]
                self.Threads = [Thread(target=temp.run(self.config, 
                                                  self.model_version_dicts_list[temp.modelIdx][str(temp.modelVersion)],
                                                  self.fl_modules[temp.modelIdx])) for temp in self.threadedEvents]
                [t.start() for t in self.Threads]
                [t.join() for t in self.Threads]
                for upd in self.threadedEvents:
                    self.update(upd) 
                    self.T = max(self.T, upd.finish_time) #Set global time to the current work

            if self.varBasedR:
                tempBoolVarBasedR = (max(self.round_list)%max([j.round_to_test for j in self.config.models])==0)
                if tempBoolVarBasedR:
                    self.varInfo = [self.calculateUpdVariance(i) for i in range(self.nb_of_models)]
            for modelIdx in range(self.nb_of_models):
                if self.howManyClients[modelIdx] != 0:
                    self.aggregation(modelIdx)
                    del self.model_version_dicts_list[modelIdx][str(self.round_list[modelIdx]-1)]
            if self.varBasedR and tempBoolVarBasedR:
                newActiveWorks = self.redistributeActiveWorks(self.varInfo)
                logging.info('Act.Work dist. update: Old:'\
                        +str([self.config.models[i].nb_of_active_works for i in range(self.nb_of_models)])\
                        +' New:'+str(newActiveWorks)+' Var.s:'+str(self.varInfo))
                for i in range(self.nb_of_models):
                    self.config.models[i] = self.config.models[i]._replace(nb_of_active_works = newActiveWorks[i])

    def sync_ucb_run(self):
        tempOriginalModelDevices = [self.config.models[i].model_device for i in range(self.nb_of_models)]
        for i in range(self.nb_of_models):
            self.config.models[i] = self.config.models[i]._replace(model_device = 'cpu')
        self.gamma = self.config.ucb_gamma
        self.round_list = [0 for _ in range(self.nb_of_models)]
        self.model_version_dicts_list = [{'0':self.fl_modules[i].extract_weights(self.models[i],
                                                                                 self.config.models[i].model_device)}
                                         for i in range(self.nb_of_models)]
        self.test_accuracies = [[[],[],[],[]] for _ in range(self.nb_of_models)]
        self.buffers = [[] for _ in range(self.nb_of_models)]
        self.T = 0 #Time
        self.is_finished_list = [False for _ in range(self.nb_of_models)]
        
        self.target_rounds_list = []
        self.target_accuracy_list = []
        
        for i in range(self.nb_of_models):
            self.target_rounds_list.append(self.config.models[i].rounds)
            self.target_accuracy_list.append(self.config.models[i].target_accuracy)
            if self.config.models[i].target_accuracy:
                logging.info('Model '+str(i+1)+': Training: {} rounds or {}% accuracy\n'.format(
                    self.config.models[i].rounds, 100 * self.config.models[i].target_accuracy))
            else:
                logging.info('Model '+str(i+1)+': Training: {} rounds\n'.format(self.config.models[i].rounds))

        self.qEvents = PriorityQueue()
        self.activeClients =self.config.partialNbofClients if self.config.partialNbofClients else self.total_clients

        #Normalize number of active clients

        self.howManyClients = [max(min(self.config.models[i].nb_of_active_works,1),int(self.activeClients*self.config.models[i].nb_of_active_works/sum([self.config.models[ii].nb_of_active_works for ii in range(self.nb_of_models)]))) for i in range(self.nb_of_models)]

        for i in range(self.nb_of_models):
            self.config.models[i] = self.config.models[i]._replace(nb_of_active_works =self.howManyClients[i] )
        
        while not np.all(self.is_finished_list) and self.T < self.config.max_time:
            if self.round_list[0] == 0: #sample all clients in the first round
                for i in range(self.nb_of_models): 
                    for clientId in range(self.total_clients):
                        self.qEvents.put(self.createNewJob(i,client = self.clients[clientId]))
                        self.NtList = [np.zeros(self.total_clients) for _ in range(self.nb_of_models)]
                        self.LtList = [np.zeros(self.total_clients) for _ in range(self.nb_of_models)]
            else:
                self.NtList = [self.gamma*self.NtList[i]+self.thisRoundNtList[i] for i in range(self.nb_of_models)]
                self.LtList = [self.gamma*self.LtList[i]+self.thisRoundLtList[i] for i in range(self.nb_of_models)]
                self.UtList = [np.sqrt(2*np.log(sum([self.gamma**t for t in range(max(self.round_list))]))/self.NtList[i])\
                              for i in range(self.nb_of_models)]
                self.AtList = [self.LtList[i]/self.NtList[i]+self.UtList[i] for i in range(self.nb_of_models)] #Assumed equal data points
                
                self.howManyClients = [max(min(self.config.models[i].nb_of_active_works,1),int(self.activeClients*self.config.models[i].nb_of_active_works/sum([self.config.models[ii].nb_of_active_works for ii in range(self.nb_of_models)]))) for i in range(self.nb_of_models)]
    
                if self.activeClients > sum(self.howManyClients):
                    for i in np.random.choice([j for j in np.arange(self.nb_of_models, dtype=int) if self.config.models[j].nb_of_active_works != 0], self.activeClients - sum(self.howManyClients), replace=True):
                        self.howManyClients[i] += 1
                self.firstC_list = []
                for i in range(self.nb_of_models):
                    if self.config.models[i].firstC:
                        self.firstC_list.append(min(self.config.models[i].firstC, self.howManyClients[i]))
                    else:
                        self.firstC_list.append(self.howManyClients[i])
                usedClients = []
                activeClientsThisRound = np.random.permutation(self.total_clients)[:sum(self.howManyClients)+5] #here we limited the available clients, 5 is just extra 
                for i in range(self.nb_of_models):
                    modelIdx = (max(self.round_list)+i)%self.nb_of_models
                    temp = self.AtList[modelIdx]
                    temp = np.argsort(temp)[::-1]
                    tempClientCounter = 0
                    tempIterCounter = 0
                    s = 'Model: '
                    while tempClientCounter < self.howManyClients[modelIdx]:
                        if temp[tempIterCounter] in activeClientsThisRound and temp[tempIterCounter] not in usedClients:
                            usedClients.append(temp[tempIterCounter])
                            tempClientCounter += 1
                            self.qEvents.put(self.createNewJob(modelIdx,
                                                               client = self.clients[temp[tempIterCounter]]))
                            s += str(temp[tempIterCounter])+', '
                        tempIterCounter += 1
                    # logging.info(s)
                # delete events after the firstC from queue:
                self.qEvents.queue.sort()
                for i in range(self.qEvents.qsize()-1,-1,-1):
                    j = self.qEvents.queue[i].modelIdx
                    if [1 for e in self.qEvents.queue if e.modelIdx == j].__len__() > self.firstC_list[j]:
                        self.qEvents.queue.pop(i)
                self.stalenessTemp =[[self.howManyClients[i],self.firstC_list[i]] for i in range(self.nb_of_models)] #For developing purpose

            self.thisRoundLtList = [np.zeros(self.total_clients) for _ in range(self.nb_of_models)]
            self.thisRoundNtList = [np.zeros(self.total_clients) for _ in range(self.nb_of_models)]
            while self.qEvents.qsize() != 0:
                self.threadedEvents = [self.qEvents.get() for _ in range(min(self.config.num_of_threads, self.qEvents.qsize()))]
                self.Threads = [Thread(target=temp.run(self.config, 
                                                  self.model_version_dicts_list[temp.modelIdx][str(temp.modelVersion)],
                                                  self.fl_modules[temp.modelIdx])) for temp in self.threadedEvents]
                [t.start() for t in self.Threads]
                [t.join() for t in self.Threads]
                for upd in self.threadedEvents:
                    self.update(upd) 
                    self.T = max(self.T, upd.finish_time) #Set global time to the current work
                    self.thisRoundNtList[upd.modelIdx][upd.client.client_id] = 1
                    self.thisRoundLtList[upd.modelIdx][upd.client.client_id] = upd.trainingLoss
            if self.varBasedR:
                tempBoolVarBasedR = (max(self.round_list)%max([j.round_to_test for j in self.config.models])==0)
                if tempBoolVarBasedR:
                    self.varInfo = [self.calculateUpdVariance(i) for i in range(self.nb_of_models)]
            for modelIdx in range(self.nb_of_models):
                if self.howManyClients[modelIdx] != 0:
                    self.aggregation(modelIdx)
                    del self.model_version_dicts_list[modelIdx][str(self.round_list[modelIdx]-1)]
            if self.round_list[0] == 1: #Back to original setting
                for i in range(self.nb_of_models):
                    self.config.models[i] = self.config.models[i]._replace(model_device = tempOriginalModelDevices[i])
                    self.models[i] = self.models[i].to(tempOriginalModelDevices[i])
                    self.model_version_dicts_list[i]['1'] = self.fl_modules[i].extract_weights(self.models[i],
                                                                                 self.config.models[i].model_device)
            if self.varBasedR and tempBoolVarBasedR:
                newActiveWorks = self.redistributeActiveWorks(self.varInfo)
                logging.info('Act.Work dist. update: Old:'\
                        +str([self.config.models[i].nb_of_active_works for i in range(self.nb_of_models)])\
                        +' New:'+str(newActiveWorks)+' Var.s:'+str(self.varInfo))
                for i in range(self.nb_of_models):
                    self.config.models[i] = self.config.models[i]._replace(nb_of_active_works = newActiveWorks[i])

    
    def sync_run(self):
        self.round_list = [0 for _ in range(self.nb_of_models)]
        self.model_version_dicts_list = [{'0':self.fl_modules[i].extract_weights(self.models[i],
                                                                                 self.config.models[i].model_device)}
                                         for i in range(self.nb_of_models)]
        self.test_accuracies = [[[],[],[],[]] for _ in range(self.nb_of_models)]
        self.buffers = [[] for _ in range(self.nb_of_models)]
        self.T = 0 #Time
        self.is_finished_list = [False for _ in range(self.nb_of_models)]
        
        self.target_rounds_list = []
        self.target_accuracy_list = []
        
        for i in range(self.nb_of_models):
            self.target_rounds_list.append(self.config.models[i].rounds)
            self.target_accuracy_list.append(self.config.models[i].target_accuracy)
            if self.config.models[i].target_accuracy:
                logging.info('Model '+str(i+1)+': Training: {} rounds or {}% accuracy\n'.format(
                    self.config.models[i].rounds, 100 * self.config.models[i].target_accuracy))
            else:
                logging.info('Model '+str(i+1)+': Training: {} rounds\n'.format(self.config.models[i].rounds))

        self.qEvents = PriorityQueue()
        self.activeClients =self.config.partialNbofClients if self.config.partialNbofClients else self.total_clients

        #Normalize number of active clients

        self.howManyClients = [max(min(self.config.models[i].nb_of_active_works,1),int(self.activeClients*self.config.models[i].nb_of_active_works/sum([self.config.models[ii].nb_of_active_works for ii in range(self.nb_of_models)]))) for i in range(self.nb_of_models)]

        for i in range(self.nb_of_models):
            self.config.models[i] = self.config.models[i]._replace(nb_of_active_works =self.howManyClients[i] )
        
        while not np.all(self.is_finished_list) and self.T < self.config.max_time:
            self.howManyClients = [max(min(self.config.models[i].nb_of_active_works,1),int(self.activeClients*self.config.models[i].nb_of_active_works/sum([self.config.models[ii].nb_of_active_works for ii in range(self.nb_of_models)]))) for i in range(self.nb_of_models)]

            if self.activeClients > sum(self.howManyClients):
                for i in np.random.choice([j for j in np.arange(self.nb_of_models, dtype=int) if self.config.models[j].nb_of_active_works != 0], self.activeClients - sum(self.howManyClients), replace=True):
                    self.howManyClients[i] += 1
            self.firstC_list = []
            for i in range(self.nb_of_models):
                if self.config.models[i].firstC:
                    self.firstC_list.append(min(self.config.models[i].firstC, self.howManyClients[i]))
                else:
                    self.firstC_list.append(self.howManyClients[i])
            temp = np.random.permutation(self.total_clients) #uniform partitioning of clients
            for i in range(self.nb_of_models):
                if self.config.models[i].nb_of_active_works != 0:
                    for clientId in temp[sum(self.howManyClients[:i]):sum(self.howManyClients[:i+1])]:
                        self.qEvents.put(self.createNewJob(i,client = self.clients[clientId]))

            # delete events after the firstC from queue:
            self.qEvents.queue.sort()
            for i in range(self.qEvents.qsize()-1,-1,-1):
                j = self.qEvents.queue[i].modelIdx
                if [1 for e in self.qEvents.queue if e.modelIdx == j].__len__() > self.firstC_list[j]:
                    self.qEvents.queue.pop(i)
            self.stalenessTemp =[[self.howManyClients[i],self.firstC_list[i]] for i in range(self.nb_of_models)] #For developing purpose
            while self.qEvents.qsize() != 0:
                self.threadedEvents = [self.qEvents.get() for _ in range(min(self.config.num_of_threads, self.qEvents.qsize()))]
                self.Threads = [Thread(target=temp.run(self.config, 
                                                  self.model_version_dicts_list[temp.modelIdx][str(temp.modelVersion)],
                                                  self.fl_modules[temp.modelIdx])) for temp in self.threadedEvents]
                [t.start() for t in self.Threads]
                [t.join() for t in self.Threads]
                for upd in self.threadedEvents:
                    self.update(upd) 
                    self.T = max(self.T, upd.finish_time) #Set global time to the current work

            if self.varBasedR:
                tempBoolVarBasedR = (max(self.round_list)%max([j.round_to_test for j in self.config.models])==0)
                if tempBoolVarBasedR:
                    self.varInfo = [self.calculateUpdVariance(i) for i in range(self.nb_of_models)]
            for modelIdx in range(self.nb_of_models):
                if self.howManyClients[modelIdx] != 0:
                    self.aggregation(modelIdx)
                    del self.model_version_dicts_list[modelIdx][str(self.round_list[modelIdx]-1)]
            if self.varBasedR and tempBoolVarBasedR:
                newActiveWorks = self.redistributeActiveWorks(self.varInfo)
                logging.info('Act.Work dist. update: Old:'\
                        +str([self.config.models[i].nb_of_active_works for i in range(self.nb_of_models)])\
                        +' New:'+str(newActiveWorks)+' Var.s:'+str(self.varInfo))
                for i in range(self.nb_of_models):
                    self.config.models[i] = self.config.models[i]._replace(nb_of_active_works = newActiveWorks[i])

    def async_run_fedast(self):
        self.test_accuracies = [[[],[],[], []] for _ in range(self.nb_of_models)]
        self.model_version_dicts_list = [{'0':self.fl_modules[i].extract_weights(self.models[i],
                                                                                 self.config.models[i].model_device)}
                                         for i in range(self.nb_of_models)]
        self.buffers = [[] for _ in range(self.nb_of_models)]
        self.T = 0 #Time
        self.is_finished_list = [False for _ in range(self.nb_of_models)]
        self.round_list = [0 for _ in range(self.nb_of_models)]
        self.target_rounds_list = []
        self.target_accuracy_list = []

        for i in range(self.nb_of_models):
            self.target_rounds_list.append(self.config.models[i].rounds)
            self.target_accuracy_list.append(self.config.models[i].target_accuracy)
            if self.config.models[i].target_accuracy:
                logging.info('Model '+str(i+1)+': Training: {} rounds or {}% accuracy\n'.format(
                    self.config.models[i].rounds, 100 * self.config.models[i].target_accuracy))
            else:
                logging.info('Model '+str(i+1)+': Training: {} rounds\n'.format(self.config.models[i].rounds))
        
        # self.qEvents is a queue for all async events
        self.qEvents = PriorityQueue()
        self.qUpdates = PriorityQueue()
        #Initialize all models
        [[self.qEvents.put(temp) for temp in self.initialization(modelIdx)] for modelIdx in range(self.nb_of_models)]
        self.howManyActiveWorkPerModel = [self.config.models[modelIdx].nb_of_active_works for modelIdx in range(self.nb_of_models)]
        while not np.all(self.is_finished_list) and self.T < self.config.max_time:
            
            self.threadedEvents = [self.qEvents.get() for _ in range(min(self.config.num_of_threads, self.qEvents.qsize()))]
            self.Threads = [Thread(target=temp.run(self.config, 
                                              self.model_version_dicts_list[temp.modelIdx][str(temp.modelVersion)],
                                              self.fl_modules[temp.modelIdx])) for temp in self.threadedEvents]
            [t.start() for t in self.Threads]
            [t.join() for t in self.Threads]
            [self.qUpdates.put(temp) for temp in self.threadedEvents]
            
            while self.qUpdates.qsize() != 0:
                upd = self.qUpdates.get()
                self.T = upd.finish_time #Set global time to the current work
                self.stalenessTemp[upd.modelIdx].append(self.round_list[upd.modelIdx]-upd.modelVersion)
                self.update(upd)

                self.howManyActiveWorkPerModel[upd.modelIdx] = max(self.howManyActiveWorkPerModel[upd.modelIdx]-1,0)
                temp_how_many_new = 0 
                if self.howManyActiveWorkPerModel[upd.modelIdx]<self.config.models[upd.modelIdx].nb_of_active_works:
                    temp_how_many_new = min(min(self.config.models[upd.modelIdx].nb_of_active_works-self.howManyActiveWorkPerModel[upd.modelIdx],sum([j.nb_of_active_works for j in self.config.models])-sum(self.howManyActiveWorkPerModel)),2)

                for _ in range(temp_how_many_new):
                    c = self.select_a_client()
                    activeClientsTemp = set([e.client for e in (self.qEvents.queue + self.qUpdates.queue)])
                    if activeClientsTemp.__len__()>=self.config.partialNbofClients and  (not (c in activeClientsTemp)):
                        continue
                    newEvent = self.createNewJob(upd.modelIdx, client = c)
                    if newEvent is None:
                        continue
                    self.howManyActiveWorkPerModel[upd.modelIdx] += 1
                    if self.qUpdates.qsize() != 0:
                        if newEvent < sorted(self.qUpdates.queue)[-1]: #Interrupt already-done updates if a new job has to precede
                            newEvent.run(self.config, 
                                        self.model_version_dicts_list[newEvent.modelIdx][str(newEvent.modelVersion)],
                                        self.fl_modules[newEvent.modelIdx])
                            self.qUpdates.put(newEvent)
                        else:
                            self.qEvents.put(newEvent)
                    else:
                        self.qEvents.put(newEvent)
                
                if self.full_async:
                    self.config.models[upd.modelIdx] = self.config.models[upd.modelIdx]._replace(buffer_size = 1)
                elif self.varBasedR:
                    self.config.models[upd.modelIdx] = self.config.models[upd.modelIdx]._replace(buffer_size = \
                                max(1, self.howManyActiveWorkPerModel[upd.modelIdx]//self.R_bs_ratio+\
                                    int(bool(self.howManyActiveWorkPerModel[upd.modelIdx]%self.R_bs_ratio)) ) )#original
                
                # Check upd.modelVersion is still required; delete that version if not needed to release memory
                if self.round_list[upd.modelIdx] != upd.modelVersion and str(upd.modelVersion) in self.model_version_dicts_list[upd.modelIdx].keys():
                    if not self.requireModelVersion(upd.modelIdx, upd.modelVersion):
                        del self.model_version_dicts_list[upd.modelIdx][str(upd.modelVersion)]
                
    def requireModelVersion(self, modelIdx, version): 
        for elem in self.qEvents.queue:
            if elem.modelIdx == modelIdx and elem.modelVersion == version:
                return True
        return False

    # Perform one update and create new event if necessary
    def update(self,update):

        if self.varBasedR and self.config.algo == 'fedast':
            self.varEstModels[update.modelIdx].append(update.delta)
            if self.varEstWindowSizes[update.modelIdx] == len(self.varEstModels[update.modelIdx]):
                del self.varEstModels[update.modelIdx][0]            
                self.varVariances[update.modelIdx].append(self.calculateUpdVariance(update.modelIdx))
            self.varUpdatePeriodsCounter += 1
            if self.varUpdatePeriodsCounter >= self.varUpdatePeriods and (min(self.round_list)>max(self.varEstWindowSizes)) and\
                np.all([len(self.varVariances[j])>0 for j in range(self.nb_of_models) if self.config.models[j].nb_of_active_works>0]):
                self.varUpdatePeriodsCounter = 0
                self.varInfo = [sum(self.varVariances[j])/len(self.varVariances[j]) if self.config.models[j].nb_of_active_works>0 else 0\
                                for j in range(self.nb_of_models)]
                                
                self.varVariances = [[] for _ in range(self.nb_of_models)]
                newActiveWorks = self.redistributeActiveWorks(self.varInfo)
                logging.info('Act.Work dist. update: Old:'\
                        +str([self.config.models[i].nb_of_active_works for i in range(self.nb_of_models)])\
                        +' New:'+str(newActiveWorks)+' Var.s:'+str(self.varInfo))
                for i in range(self.nb_of_models):
                    self.config.models[i] = self.config.models[i]._replace(nb_of_active_works = newActiveWorks[i])

        self.buffers[update.modelIdx].append(update.delta)
        update.client.events_in_queue -= 1
        if self.config.algo == 'fedast':
            if len(self.buffers[update.modelIdx]) >= self.config.models[update.modelIdx].buffer_size: #aggregation happens
                self.aggregation(update.modelIdx)
            return None 
        elif self.config.algo == 'sync' or self.config.algo == 'sync-bods' or self.config.algo == 'sync-ucb':
            return None
    def aggregation(self, modelIdx):
        sumOfUpdates = [torch.zeros(x.size(),device = self.config.models[modelIdx].model_device)\
                        for _, x in self.buffers[modelIdx][0]]
        for i, update in enumerate(self.buffers[modelIdx]):
            for j, (_, weighted_delta) in enumerate(update):
                sumOfUpdates[j] += weighted_delta
        updated_weights = []

        for i, (name, weight) in enumerate(self.model_version_dicts_list[modelIdx][str(self.round_list[modelIdx])]):
            updated_weights.append((name, weight + \
                        self.config.models[modelIdx].global_lr/len(self.buffers[modelIdx]) * sumOfUpdates[i]))
        self.buffers[modelIdx] = []
        self.round_list[modelIdx] += 1
        if self.varBasedR:
            self.fl_modules[modelIdx].load_weights(self.modelsPrevVersion[modelIdx],\
                                                   self.model_version_dicts_list[modelIdx][str(self.round_list[modelIdx]-1)])
        self.model_version_dicts_list[modelIdx][str(self.round_list[modelIdx])] = updated_weights
        self.fl_modules[modelIdx].load_weights(self.models[modelIdx], updated_weights)   

        if self.round_list[modelIdx] == self.target_rounds_list[modelIdx]:
            self.is_finished_list[modelIdx] = True
        
        if not self.config.clients.do_test and \
        self.round_list[modelIdx]%self.config.models[modelIdx].round_to_test==0:
            accuracy, loss = self.testAccuracy(modelIdx)
            
            self.test_accuracies[modelIdx][0].append(self.T)
            self.test_accuracies[modelIdx][1].append(self.round_list[modelIdx])
            self.test_accuracies[modelIdx][2].append(accuracy)
            self.test_accuracies[modelIdx][3].append(loss)
            if self.config.wandb:
                self.wandb_dict['acc_'+self.model_run_names[modelIdx]] = accuracy
                self.wandb_dict['loss_'+self.model_run_names[modelIdx]] = loss
                self.wandb_dict['acc_'+str(modelIdx)] = accuracy
                self.wandb_dict['loss_'+str(modelIdx)] = loss  
                self.wandb_dict['time_'+self.model_run_names[modelIdx]] = self.wandb_dict['time'] = self.T
                self.wandb_dict['tasksAvgAcc'] = np.mean([self.test_accuracies[i][2][-1] for i in range(self.nb_of_models) if self.test_accuracies[i][2]])
                self.wandb_dict['tasksStdAcc'] = np.std([self.test_accuracies[i][2][-1] for i in range(self.nb_of_models) if self.test_accuracies[i][2]])
                self.wandb_dict['tasksAvgLoss'] = np.mean([self.test_accuracies[i][3][-1] for i in range(self.nb_of_models) if self.test_accuracies[i][3]])
                self.wandb_dict['tasksStdLoss'] = np.std([self.test_accuracies[i][3][-1] for i in range(self.nb_of_models) if self.test_accuracies[i][3]])
                self.wandb_dict['nbOfActiveClients'] = sum(self.howManyClients) if 'sync' in self.config.algo else set([e.client for e in (self.qEvents.queue + self.qUpdates.queue)]).__len__()
                self.wandb_dict['globalRound_m'+str(modelIdx)] = self.round_list[modelIdx]
                temp = self.config.models[modelIdx].buffer_size if 'sync' not in self.config.algo[:4] else self.howManyClients[modelIdx]
                self.wandb_dict['clientTrips_m'+str(modelIdx)] += temp
                self.wandb_dict['totalClientTrips'] += temp

                self.wandb_dict['avgStaleness_m'+str(modelIdx)] = sum(self.stalenessTemp[modelIdx]) / len(self.stalenessTemp[modelIdx]) if 'sync' not in self.config.algo[:4] else 0
                self.wandb.log(self.wandb_dict)

            temp =[self.howManyActiveRequestsModel(i) for i in range(self.nb_of_models)] if self.config.algo == 'fedast' else ''
            infoTemp = 't={:.2f}: Model-{} v{} acc={:.4f}%, test loss={:.4f}'.format(self.T,
            modelIdx+1, self.round_list[modelIdx]-1,100*accuracy,loss)+str(self.stalenessTemp[modelIdx])
            infoTemp = infoTemp+ ', Rs=' +str(temp)+'='+str(sum(temp))+', bs='+str([j.buffer_size for j in self.config.models])+'||otherRs='+str(self.howManyActiveWorkPerModel)  if self.config.algo == 'fedast' else infoTemp
            logging.info(infoTemp)
            # logging.info('L estimates: '+str(self.calculate_ratio_of_differences()))
            
            if self.target_accuracy_list[modelIdx] and accuracy >= self.target_accuracy_list[modelIdx]:
                logging.info('Model {0} ({1}) reached the target. Its {2} active works are distributed to remaining tasks.'.format(modelIdx+1, self.config.models[modelIdx].name, self.config.models[modelIdx].nb_of_active_works))
                self.is_finished_list[modelIdx] = True
                if np.any(self.is_finished_list):
                    if np.all(self.is_finished_list):
                        logging.info('All tasks are finished!')
                        return None
                    # Cancel planned updates for async
                    if 'sync' not in self.config.algo: 
                        for i in range(self.qEvents.qsize()-1,-1,-1):
                            if self.qEvents.queue[i].modelIdx == modelIdx:
                                self.qEvents.queue[i].client.events_in_queue -= 1
                                del self.qEvents.queue[i]
                                self.howManyActiveWorkPerModel[modelIdx] -= 1
                        for i in range(self.qUpdates.qsize()-1,-1,-1):
                            if self.qUpdates.queue[i].modelIdx == modelIdx:
                                self.qEvents.queue[i].client.events_in_queue -= 1
                                del self.qUpdates.queue[i]
                                self.howManyActiveWorkPerModel[modelIdx] -= 1
                    
                    # Redistribute the resources of the finished tasks to the other tasks
                    toDistribute = self.config.models[modelIdx].nb_of_active_works
                    self.config.models[modelIdx] = self.config.models[modelIdx]._replace(nb_of_active_works = 0)
                    if self.varBasedR and self.config.algo == 'fedast':
                        self.howManyActiveWorkPerModel[modelIdx] = 0
                        self.varUpdatePeriods = self.varUpdatePeriodsC * len([1 for j in self.config.models if j.nb_of_active_works>0])
                        logging.info("varUpdatePeriods: "+str(self.varUpdatePeriods))
    
                    if not np.all(self.is_finished_list): 
                        distributeList = [int(toDistribute*self.config.models[i].nb_of_active_works/sum([self.config.models[ii].nb_of_active_works for ii in range(self.nb_of_models)])) for i in range(self.nb_of_models)]
                        while toDistribute > sum(distributeList):
                            i = np.random.choice(np.arange(self.nb_of_models, dtype=int), 1)[0]
                            if self.config.models[i].nb_of_active_works != 0:
                                distributeList[i] += 1

                        self.config.models[modelIdx] = self.config.models[modelIdx]._replace(nb_of_active_works = 0)
                        for i, add in enumerate(distributeList):
                            oldActiveWork = self.config.models[i].nb_of_active_works
                            self.config.models[i] = self.config.models[i]._replace(nb_of_active_works = self.config.models[i].nb_of_active_works+add)
                            logging.info('Model {0} ({1}) gets {2} more active works!'.format(i+1, self.config.models[i].name, add))
                            if 'sync' not in self.config.algo:
                                if self.full_async:
                                    self.config.models[i] = self.config.models[i] if not bool(oldActiveWork) else self.config.models[i]._replace(buffer_size = 1)
                                else:
                                    self.config.models[i] = self.config.models[i] if not bool(oldActiveWork) else self.config.models[i]._replace(buffer_size = round(self.config.models[i].nb_of_active_works/self.R_bs_ratio))

                                if not self.varBasedR:
                                    [self.qEvents.put(temp) for temp in [self.createNewJob(i) for _ in range(add)]]
                                    self.howManyActiveWorkPerModel[i] += add

            if math.isnan(self.test_accuracies[modelIdx][3][-1]):
                logging.info('Model '+str(modelIdx+1)+' diverged. Finishing..')
                self.is_finished_list = [True for _ in range(self.nb_of_models)]
        self.stalenessTemp[modelIdx] = [] 
    def testAccuracy(self, modelIdx):
        return self.fl_modules[modelIdx].test(self.models[modelIdx],
                                                  self.testloader_list[modelIdx])

    def createNewJob(self,modelIdx,weight = None,client = None):
        if self.is_finished_list[modelIdx]:
            return None
        else:
            c = self.select_a_client(modelIdx = modelIdx) if not client else client
            generated_delay = c.generateDelay(self.config, modelIdx)
            if weight:
                return asyncEvent(c, modelIdx, self.round_list[modelIdx],
                              generated_delay,self.T, c.last_finish_time, weight)
            else:
                return asyncEvent(c, modelIdx, self.round_list[modelIdx],
                              generated_delay, self.T, c.last_finish_time,1)

    def initialization(self, modelIdx):
        if self.config.algo == 'fedast':
            return [self.createNewJob(modelIdx)
                for _ in range(self.config.models[modelIdx].nb_of_active_works)] 
    def select_a_client(self, modelIdx = None, initialization = False):
        if self.config.algo == 'fedast':
            return random.sample(self.clients,1)[0]
    def set_client_data(self, client):
        
        data_list = []
        for i in range(self.nb_of_models):
            temp = self.doesSameTaskExistBefore(modelId = i)
            if temp is not None:
                data_list.append(data_list[temp])
            else:
                tempState = random.getstate()
                random.setstate(self.data_rand_states[i])    
                
                loader = self.config.models[i].data.loader
        
                # Get data partition size
                if loader != 'shard' and loader != 'leaf':
                    if self.config.models[i].data.partition.get('size'):
                        partition_size = self.config.models[i].data.partition.get('size')
                    elif self.config.models[i].data.partition.get('range'):
                        start, stop = self.config.models[i].data.partition.get('range')
                        partition_size = random.randround(start, stop)
        
                # Extract data partition for client
                if loader == 'basic':
                    data = self.loaders[i].get_partition(partition_size)
                elif loader == 'bias':
                    data = self.loaders[i].get_partition(partition_size, client.pref_list[i])
                elif loader == 'dirichlet':
                    data = self.loaders[i].get_partition(partition_size, self.config.models[i].data.dirichlet['alpha'])
                elif loader == 'leaf':
                    data = self.loaders[i].extract(self.loaders[i].select_loader_client[client.client_id])
                else:
                    logging.critical('Unknown data loader type')
                data_list.append(data)
                self.data_rand_states[i] = random.getstate()
                random.setstate(tempState)
        # Send data to client
        client.set_data(data_list, self.config)

    def doesSameTaskExistBefore(self, modelId):
        if modelId == 0:
            return None
        else:
            for i,m in enumerate(self.config.models[:modelId]):
                if self.config.models[modelId].name == m.name or ('CIFAR' in m.name and 'CIFAR' in self.config.models[modelId].name):
                    if self.config.models[modelId].data == m.data:
                        return i
        return None
            
    def save_model(self, model, path):
        path += '/global'
        a = model.to('cpu')
        torch.save(a.state_dict(), path)


