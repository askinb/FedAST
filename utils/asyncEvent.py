class asyncEvent(object):
    """Basic async event."""
    def __init__(self, client, modelIdx, modelVersion, workTime, T, finish_time, agg_weight):

        self.workTime = workTime
        client.events_in_queue += 1
        client.last_finish_time = workTime + max(T, finish_time)
        self.client = client
        self.finish_time = workTime + max(T, finish_time)
        self.modelIdx = modelIdx
        self.modelVersion = modelVersion
        self.agg_weight = agg_weight
        self.workTime = workTime
    def run(self, config, weights, fl_module):

        # Configure and train
        self.client.configure(self.modelIdx, config, weights, fl_module)
        self.delta, self.trainingLoss = self.client.train(config, self.modelIdx,weights, fl_module, self.agg_weight)
            
    def __eq__(self, other):
        return self.finish_time == other.finish_time

    def __ne__(self, other):
        return self.finish_time != other.finish_time

    def __lt__(self, other):
        return self.finish_time < other.finish_time

    def __le__(self, other):
        return self.finish_time <= other.finish_time

    def __gt__(self, other):
        return self.finish_time > other.finish_time

    def __ge__(self, other):
        return self.finish_time >= other.finish_time
    
