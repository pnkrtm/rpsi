class ClassesKeeper:
    def add_class(self, K, group=None):
        if group:
            if group not in self.__dict__:
                setattr(self, group, self.__class__())
            self.__dict__[group].add_class(K, None)
        else:
            setattr(self, K.__name__, K)

    def __getitem__(self, item):  # added for backward compatibility
        return self.__dict__[item]

    def keys(self):
        return self.__dict__.keys()