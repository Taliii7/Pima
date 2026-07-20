from stardist.models import StarDist2D
from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label
from csbdeep.utils import normalize
import matplotlib.pyplot as plt
import numpy as np 
# prints a list of available models
StarDist2D.from_pretrained()

# creates a pretrained model
model = StarDist2D.from_pretrained('2D_versatile_fluo')



img = test_image_nuclei_2d()

print('Image',img)
print('Img shape',img.shape)
labels, _ = model.predict_instances(normalize(img))
print(np.max(labels))
print('labels',labels)
print('labels shape', labels.shape)
print("uniques labels", np.unique(labels,return_counts=True))
"""
l=[0]*119
print(l)

for i in range(512):
    for y in range(512):
        #print((i,y))
        #print(labels[i,y])
        l[labels[i,y]]+=1

print(l)


plt.subplot(1,2,1)
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.title("input image")

plt.subplot(1,2,2)
plt.imshow(render_label(labels, img=img))
plt.axis("off")
plt.title("prediction + input overlay")
plt.show()
plt.close()
"""
