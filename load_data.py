import logging
import random
from torchvision import datasets, transforms
import utils.dists as dists
import numpy as np
import random

class Generator(object):
    """Generate federated learning training and testing data."""
    # Abstract read function
    def read(self, path, testset_device = 'cpu'):
        # Read the dataset, set: trainset, testset, labels
        raise NotImplementedError

    # Group the data by label
    def group(self):
        # Create empty dict of labels
        grouped_data = {label: []
                        for label in self.labels}  # pylint: disable=no-member

        # Populate grouped data dict
        for datapoint in self.trainset:  # pylint: disable=all
            _, label = datapoint  # Extract label
            label = self.labels[label]

            grouped_data[label].append(  # pylint: disable=no-member
                datapoint)

        self.trainset = grouped_data  # Overwrite trainset with grouped data

    # Run data generation
    def generate(self, path, testset_device = 'cpu'):
        self.read(path,testset_device=testset_device)
        self.trainset_size = len(self.trainset)  # Extract trainset size
        self.group()

        return self.trainset


class Loader(object):
    """Load and pass IID data partitions."""

    def __init__(self, config, generator, modelIdx=0):
        # Get data from generator
        self.config = config
        self.trainset = generator.trainset
        self.testset = generator.testset
        self.labels = generator.labels
        self.trainset_size = generator.trainset_size
        self.modelIdx = modelIdx
        # Store used data seperately
        self.used = {label: [] for label in self.labels}
        self.used['testset'] = []
        self.npRandState = np.random.RandomState(seed=random.randint(0,100000000))

    def extract(self, label, n):
        if len(self.trainset[label]) > n:
            extracted = self.trainset[label][:n]  # Extract data
            self.used[label].extend(extracted)  # Move data to used
            del self.trainset[label][:n]  # Remove from trainset
            return extracted
        else:
            logging.warning('Insufficient data in label: {}'.format(label))
            logging.warning('Dumping used data for reuse')

            # Unmark all data as used
            for label in self.labels:
                self.trainset[label].extend(self.used[label])
                self.used[label] = []

            # Extract replenished data
            return self.extract(label, n)

    def get_partition(self, partition_size):
        # Get an partition uniform across all labels

        # Use uniform distribution
        dist = dists.uniform(partition_size, len(self.labels))

        partition = []  # Extract data according to distribution
        for i, label in enumerate(self.labels):
            partition.extend(self.extract(label, dist[i]))

        # Shuffle data partition
        random.shuffle(partition)

        return partition

    def get_testset(self):
        # Return the entire testset
        return self.testset

class DirichletLoader(Loader):
    def get_partition(self, partition_size, alpha):
        # Get a non-uniform partition with a drichlet distribution
        dist = np.round(self.npRandState.dirichlet([alpha]*len(self.labels))*partition_size).astype(int)
        
        partition = []  # Extract data according to distribution
        for i, label in enumerate(self.labels):
            partition.extend(self.extract(label, dist[i]))

        # Shuffle data partition
        random.shuffle(partition)

        return partition

class BiasLoader(Loader):
    """Load and pass 'preference bias' data partitions."""

    def get_partition(self, partition_size, pref):
        # Get a non-uniform partition with a preference bias

        # Extract bias configuration from config
        bias = self.config.models[self.modelIdx].data.bias['primary']
        secondary = self.config.models[self.modelIdx].data.bias['secondary']

       # Calculate sizes of majorty and minority portions
        majority = int(partition_size * bias)
        minority = partition_size - majority

        # Calculate number of minor labels
        len_minor_labels = len(self.labels) - 1

        if secondary:
                # Distribute to random secondary label
            dist = [0] * len_minor_labels
            dist[random.randint(0, len_minor_labels - 1)] = minority
        else:
            # Distribute among all minority labels
            dist = dists.uniform(minority, len_minor_labels)

        # Add majority data to distribution
        dist.insert(self.labels.index(pref), majority)

        partition = []  # Extract data according to distribution
        for i, label in enumerate(self.labels):
            partition.extend(self.extract(label, dist[i]))

        # Shuffle data partition
        random.shuffle(partition)

        return partition

class LEAFLoader(object):
    """Load and pass IID data partitions."""

    def __init__(self, config,generator,modelIdx):
        # Get data from generator
        self.npRandState = np.random.RandomState(seed=random.randint(0,100000000))
        self.config = config
        self.trainset = generator.trainset
        self.testset = generator.testset
        self.labels = generator.labels
        if config.models[modelIdx].data.loader == 'leaf':
            self.num_clients = len(generator.trainset['users'])
        self.select_loader_client = leaf_loader_client = self.npRandState.choice(np.arange(self.config.clients.total),
                                     self.config.clients.total, replace=False)
        if config.models[modelIdx].data.leaf:
            self.client_user_random_match = self.trainset['users'].copy()
            random.shuffle(self.client_user_random_match)

    def extract(self, client_id):
        # Given client_id, extract the training data of that user
        # The extracted data is in a dictionary format with keys of 'x' and 'y'
        user_name = self.client_user_random_match[client_id]
        return self.trainset['user_data'][user_name], self.testset['user_data'][user_name]

    def get_testset(self):
        # Extract the testing data of all users in the list of loader_id
        # The extracted data is in a dictionary format with keys of 'x' and 'y'
        testset = {'x': [], 'y': []}
        for user in self.testset['users']:
            testset['x'] += self.testset['user_data'][user]['x']
            testset['y'] += self.testset['user_data'][user]['y']
        # user_name = self.trainset['users'][client_id]
        return testset