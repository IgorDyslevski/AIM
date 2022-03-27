import numpy as np


class Sample:
    def __init__(self, length: int, count: int, one_of_mut: int, one_of_rand: int) -> None:
        self.set_length(length)
        self.set_count(count)
        self.set_weights(self.generate_weights(self.get_length(), self.get_count()))
        self.set_one_of_mut(one_of_mut)
        self.set_one_of_rand(one_of_rand)

    def __str__(self) -> str:
        return f'Sample({ self.get_length() }, { self.get_count() })'

    def set_length(self, n: int) -> None:
        self.__length = n

    def get_length(self) -> int:
        return self.__length

    def set_one_of_mut(self, n: int) -> None:
        self.one_of_mut = n

    def get_one_of_mut(self) -> int:
        return self.one_of_mut

    def set_one_of_rand(self, n: int) -> None:
        self.one_of_rand = n

    def get_one_of_rand(self) -> int:
        return self.one_of_rand

    def set_count(self, n: int) -> None:
        self.__count = n

    def get_count(self) -> int:
        return self.__count

    def set_weights(self, weights: np.array) -> None:
        self.__weights = weights

    def get_weights(self) -> np.array:
        return self.__weights

    def set_loss(self, func) -> None:
        self.__func = func
        
    def get_loss(self):
        return self.__func

    def calc(self, high2low: bool = 0) -> tuple:
        result = []
        for i in self.get_weights():
            result.append((self.get_loss()(i), i))
        return tuple(map(lambda x: x[1], sorted(result, reverse=high2low, key=lambda x: x[0]))) 

    def print_loss(self) -> None:
        print('-------Losses--------')
        for i in self.get_weights():
            print('Loss:', self.get_loss()(i))

    def step(self, elem1: np.array, elem2: np.array) -> np.array:
        nelem = []
        rands = np.random.random_integers(0, self.get_one_of_rand(), size=(self.get_length(),))
        marges = np.random.random_integers(0, 2, size=(self.get_length(),))
        muts = np.random.random_integers(0, self.get_one_of_mut(), size=(self.get_length(),))
        for i in range(self.get_length()):
            if rands[i] == 0:
                j = np.random.randn(1)[0]
            else:
                if marges[i] in (0, 1):
                    j = elem1[i]
                else:
                    j = elem2[i]
                if muts[i] == 0:
                    j += np.random.rand(1)[0] * 2 - 1
            nelem.append(j)
        return np.array(nelem)

    def next_step(self, high2low: bool = 0) -> None:
        elem1, elem2, *_ = self.calc(high2low)
        nweights = []
        for _ in range(self.get_count()):
            nweights.append(self.step(elem1, elem2))
        self.set_weights(np.array(nweights))

    @staticmethod
    def generate_weights(length: int, count: int) -> np.array:
        return np.random.randn(count, length)

sample = Sample(10, 10, 3, 7)
@sample.set_loss
def sm(a):
    return np.abs(np.sum(a))

for i in range(1000):
    print(i)
    sample.print_loss()
    sample.next_step(1)
