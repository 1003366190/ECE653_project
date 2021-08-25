import matplotlib.pyplot as plt
import pickle
import numpy as np
def draw(model, image, label):
    plt.figure()
    plt.grid(False)
    plt.imshow(np.reshape(image, (image.shape[1],image.shape[2],3)))
    logits = model.predict(image)
    title = "prediction: "+str(np.argmax(logits)) + " confidence: "+str(np.max(logits))
    plt.title(title)
    plt.xlabel("Original Label: {}".format(label))
    plt.show()
    
def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)