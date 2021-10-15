import numpy as np
import matplotlib.pyplot as plt

def display_some_examples(examples, labels, num_images=25):
    plt.figure(figsize=(10,10))
    
    for i in range(num_images):
        idx = np.random.randint(0, examples.shape[0]-1)
        img = examples[idx]
        label = labels[idx]

        plt.subplot(5,5, i+1)
        plt.title(str(label))
        plt.tight_layout()
        plt.imshow(img, cmap='gray')

    plt.show()