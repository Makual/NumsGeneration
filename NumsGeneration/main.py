from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import keras


noise_size = 32

generator = keras.models.load_model('NUMgen_2000.h5')

def genNum(num):
  noise = np.random.normal(0, 1, (1, noise_size))
  setting = to_categorical([num],10)
  settingNoise = np.concatenate((noise,setting),1)
  generated_samples = generator(settingNoise)
  generated_samples = np.array(generated_samples)
  generated_samples = ((generated_samples * 0.5) + 0.5) * 255
  generated_samples = generated_samples.round()
  generated_samples = generated_samples.astype(np.uint8)
  return generated_samples

for i in range(100):
    ax = plt.subplot(10, 10, i + 1)
    plt.imshow(genNum(i%10)[0],cmap='gray')
    plt.xticks([])
    plt.yticks([])

plt.show()

    










