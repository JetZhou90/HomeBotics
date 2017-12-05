import pickle
from sklearn.model_selection import train_test_split


class faceload:

    def __init__(self,path):
        read_file = open(path, 'rb')
        self.faces = pickle.load(read_file)
        self.label = pickle.load(read_file)
        read_file.close()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.faces,
                                                                                self.label,
                                                                                test_size = 0.2,
                                                                                random_state = 0)
        self.batch_index=0
        self.batch_size=0


    def train_next_batch(self,batch_size):

        self.batch_size=batch_size
        batch_x = self.X_train[self.batch_index * self.batch_size: self.batch_index * self.batch_size + self.batch_size]
        batch_y = self.y_train[self.batch_index * self.batch_size: self.batch_index * self.batch_size + self.batch_size]
        self.batch_index += 1
        if self.batch_index > self.y_train.shape[0] // self.batch_size:
            self.batch_index=0
        return batch_x, batch_y

    def test_x_y(self):
        return self.X_test,self.y_test
