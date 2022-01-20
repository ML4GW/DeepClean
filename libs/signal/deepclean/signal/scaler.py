from deepclean.signal.op import Op


class StandardScaler(Op):
    def fit(self, X):
        if X.ndim > 1:
            self.mean = X.mean(axis=1, keepdims=True)
            self.std = X.std(axis=1, keepdims=True)
        else:
            self.mean = X.mean()
            self.std = X.std()

    def __call__(self, x, inverse=False):
        if self.mean.ndim > 1 and x.shape[0] != self.mean.shape[0]:
            raise ValueError(
                "Expected input with {} channels, found {}".format(
                    self.mean.shape[0], x.shape[0]
                )
            )

        if inverse:
            return self.std * x + self.mean
        return (x - self.mean) / self.std
