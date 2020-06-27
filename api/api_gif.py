import imageio
images = []
filenames = ['1.jpg','2.jpg','3.jpg']
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('api.gif', images, duration=5)