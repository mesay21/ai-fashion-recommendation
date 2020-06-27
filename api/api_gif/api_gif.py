import imageio
images = []
filenames = ['1.png','2.png','3.png']
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('api.gif', images, duration=2)