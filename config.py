from collections import namedtuple
import json
import random

class Config(object):
    """Configuration module."""

    def __init__(self, config):
        self.path = config
        # Load config file
        with open(config, 'r') as config:
            self.config = json.load(config)
        # Extract configuration
        self.extract()

    def extract(self):
        config = self.config
        # --Multithreading--
        self.num_of_threads = config.get('num_of_threads', None)
        # --Algo--
        self.algo = config.get('algo', None)
        self.bods_alpha = config.get('bods_alpha', 0.25)
        self.ucb_gamma = config.get('ucb_gamma', 0.99)
        self.varBasedR = config.get('varBasedR', False)
        self.R_bs_ratio = config.get('R_bs_ratio', 32)
        self.full_async = config.get('full_async', None)
        # --Partial Client Participation for Sync and max # clients for async--
        self.partialNbofClients = config.get('partialNbofClients', None)
        # --Time--
        self.max_time = config.get('max_time', 9999999999)
        # --Track through WandB
        self.wandb = config.get('wandb', False)
        self.wandb_runname = config.get('wandb_runname', '')
        self.wandb_project_name = config.get('wandb_project_name','fedast')
        # -- Clients --
        fields = ['total',
                  'do_test', 'test_partition','slow_ratio','normal_ratio','fast_ratio']
        defaults = (0, False, None, 0., 1., 0.)
        params = [config['clients'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.clients = namedtuple('clients', fields)(*params)


        tempList = list()
        for iii in range(len(config['models'])):
            # -- Data --
            defaults = [('loading', ''),
             ('partition', 0),
             ('IID', False),
             ('bias', None),
             ('leaf', None),
             ('dirichlet', None),
             ('loader', None),
             ('testset_device', 'cpu'),
             ('trainset_device', 'cpu'),
             ('data_seed', None),
             ('label_distribution', 'uniform'),
             ('batch_size', 1)]
            params = [config['models']['model'+str(iii+1)]['data'].get(default[0], default[1])
                      for ii, default in enumerate(defaults)]
            
            fields = [temp[0] for temp in defaults]
            if params[fields.index('IID')]:
                params[fields.index('loader')] = 'basic'
            elif params[fields.index('bias')]:
                params[fields.index('loader')] = 'bias'
            elif params[fields.index('leaf')]:
                params[fields.index('loader')] = 'leaf' 
            elif params[fields.index('dirichlet')]:
                params[fields.index('loader')] = 'dirichlet'
            datatemp = namedtuple('data', fields)(*params)
            del config['models']['model'+str(iii+1)]['data']
            
            # -- Delay --
            defaults = [('name', None),
                        ('exp_mean', None),
                        ('shift', None),
                      ('mean', None), 
                      ('support', None),
                       ('local_multi', config['models']['model'+str(iii+1)].get('local_iter',None))]
            params = [config['models']['model'+str(iii+1)]['delay'].get(default[0], default[1])
                      for ii, default in enumerate(defaults)]
            fields = [temp[0] for temp in defaults]
            delaytemp = namedtuple('data', fields)(*params)
            del config['models']['model'+str(iii+1)]['delay']
            
            # -- Rest --
            defaults = [('rounds', 0),
             ('target_accuracy', 0.2),
             ('name', ''),
             ('model_path', ''),
             ('data_path', ''),
             ('model_device', 'cpu'),
             ('nb_of_active_works', 100),
             ('batch_size', None),
             ('local_iter', None),
             ('firstC', None),
             ('epochs', False),
             ('buffer_size', 1),
             ('test_batch_size',100),
             ('round_to_test',1),
             ('local_lr', 0.01),
             ('global_lr', 1),
             ('local_momentum', 0),
             ('weight_decay', 0)]
            params = [config['models']['model'+str(iii+1)].get(default[0], default[1])
                      for ii, default in enumerate(defaults)]
            fields = [temp[0] for temp in defaults]
            
            tempList.append(namedtuple('model'+str(iii+1),[*fields,"data","delay"])(*params, datatemp, delaytemp))
        self.models  = tempList
        self.models = [self.models[vv]._replace(local_momentum = 0) for vv in range(len(self.models))] # force momentum to 0 we haven't used local momentum
        
        for i in range(len(config['models'])):    
            # Determine correct data loader
            assert self.models[i].data.IID ^ bool(self.models[i].data.bias) ^  bool(self.models[i].data.leaf) ^ bool(self.models[i].data.dirichlet)


