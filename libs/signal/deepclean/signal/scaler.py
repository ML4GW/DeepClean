from deepclean.signal import Op


class StandardScaler(Op):
    def fit(self, X):
        X =  X[[i for _, i in it if i != strain_idx]]
        self.mean = X.mean(axis=1, keepdims=True)
        self.std = X.std(axis=1, keepdims=True)

    def __call__(self, x, inverse=False):
        if x.shape[0] != self.mean.shape[0]:
            raise ValueError(
                "Expected input with {} channels, found {}".format(
                    self.mean.shape[0], x.shape[0]
                )
            )

        if inverse:
            return self.std * x + self.mean
        return (x - self.mean) / self.std
