from sklearn.pipeline import Pipeline


class MyPipeline(Pipeline):
    def __init__(self, steps):
        Pipeline.__init__(self, steps)
