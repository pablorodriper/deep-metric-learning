from torchvision.datasets import ImageFolder

import numpy as np


class SiameseDataset(ImageFolder):

    def __init__(self, root, transform):
        super(SiameseDataset, self).__init__(root, transform)
        print('Found {} images belonging to {} classes'.format(len(self), len(self.classes)))

        self.label_to_idxs = {label: np.where(np.array(self.targets) == self.class_to_idx[label])[0] for label in self.classes}

    def __getitem__(self, index):
        target = np.random.randint(0, 2)
        if target == 1:
            siamese_label = self.classes[self.targets[index]]
        else:
            siamese_label = np.random.choice(list(set(self.classes) - {self.classes[self.targets[index]]}))
        siamese_index = np.random.choice(self.label_to_idxs[siamese_label])

        return (self.samples[index][0], self.samples[siamese_index][0]), target
