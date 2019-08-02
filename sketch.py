import imageio
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import os
import image

LENGTH=5
WIDTH=4
CROSSOVER_LEN = 8
CHILDREN_COUNT = 15

def grayscale(rgb):
 return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
def dodge(front,back):
 result=front*255/(255-back)
 result[result>255]=255
 result[back==255]=255

 return result.astype("uint8")


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = imageio.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def get_fitness(child_patch, sketch_patch):

    diff_patch = np.abs(child_patch - sketch_patch)

    energy_val = sum(sum(diff_patch))

    return energy_val

def train_grey_template(gr,template,sketch):
    i=0
    j=0

    final_child = np.zeros((250, 200))
    final_point = []


    for i in range(0,250,LENGTH):
        for j in range(0,200,WIDTH):
            # print(gr[i:i+LENGTH,j:j+WIDTH])
            children, points = crossover(gr[i:i+LENGTH,j:j+WIDTH],template[i:i+LENGTH,j:j+WIDTH])

            sketch_patch = sketch[i:i+LENGTH,j:j+WIDTH]

            best_fitness = get_fitness(children[0], sketch_patch)
            best_child = children[0]
            best_child_points = points[0]

            for k in range(1, len(children)):
                fit = get_fitness(children[k], sketch_patch)
                if fit < best_fitness:
                    best_fitness = fit
                    best_child = children[k]
                    best_child_points = points[int(k/2)]

            #bc = (gr[i:i+LENGTH,j:j+WIDTH] + template[i:i+LENGTH,j:j+WIDTH])/2
            # print(best_child_points)

            final_child[i:i+LENGTH, j:j+WIDTH] = best_child
            #final_child[i:i+LENGTH, j:j+WIDTH] = bc
            final_point.append(best_child_points)

    return np.array(final_child) , np.array(final_point)

def test_sketch(sketch_in, template, cross_over_points):
    z = 0
    for i in range(0, 250, LENGTH):
        for j in range(0, 200, WIDTH):

            sketch_patch = sketch_in[i:i+LENGTH, j:j+WIDTH]
            template_patch = template[i:i+LENGTH, j:j+WIDTH]

            points = cross_over_points[z]

            sketch_patch = np.reshape(sketch_patch, LENGTH*WIDTH)
            template_patch = np.reshape(template_patch, LENGTH*WIDTH)

            for k in points:
                tmp = sketch_patch[k]
                sketch_patch[k] = template_patch[k]
                template_patch[k] = tmp

            sketch_patch = np.reshape(sketch_patch, (LENGTH, WIDTH))

            sketch_in[i:i+LENGTH, j:j+WIDTH] = sketch_patch

            z = z + 1

    return sketch_in


def crossover(gr,template):
    q=[]
    tot_points = []

    for _ in range(CHILDREN_COUNT):
        points = []

        a=gr.copy()
        b=template.copy()

        a = np.reshape(a,LENGTH*WIDTH)
        b = np.reshape(b,LENGTH*WIDTH)

        for i in (np.random.rand(CROSSOVER_LEN)*LENGTH*WIDTH).astype(int):
            tmp = b[i]
            b[i] = a[i]
            a[i] = tmp

            points.append(i)

        a = np.reshape(a, (LENGTH, WIDTH))
        b = np.reshape(b, (LENGTH, WIDTH))

        q.append(a)
        q.append(b)

        tot_points.append(points)

    result= np.asarray(q)
    return result , np.array(tot_points)
    # print (len(result))




image = load_images_from_folder('photos')
template = load_images_from_folder('sketches')

# img="photo\\f-005-01.jpg"
#start_img = imageio.imread(image[0])
sketch=[]
grey= []
i=0
reshaped = []


while(i<88):
    gray_img = grayscale(image[i])
    inverted_img = 255-gray_img
    blur_img = scipy.ndimage.filters.gaussian_filter(inverted_img,sigma=5)
    final_img= dodge(blur_img,gray_img)
    # plt.imshow(final_img, cmap="gray")
    # plt.imsave('img2.jpg', final_img, cmap='gray' ,vmin=0, vmax=255)
    sketch.append(final_img)
    grey.append(gray_img)
    i+=1

#reshaped = image[0].shpae.reshape()
#print(reshaped)
grey=np.array(grey)
template=np.array(template)

epoch = 2

global_fitness = 255*200*250
w1 = 4
w2 = 1

plt.figure()

# sk = sketch[0]

temp = np.full((250, 200), 150)

x = 0

for gr in grey[:1]:
    sk = sketch[x]
    for g in range(epoch):
        grey_child, grey_points = train_grey_template(gr,temp,template[x])
        sketch_child, sketch_points = train_grey_template(sk,temp,template[x])

        if get_fitness(grey_child, template[0]) < get_fitness(sketch_child, template[0]):
            sk = grey_child
            if g == 0:
                best_cross_over_points = grey_points
                global_fitness = get_fitness(grey_child, template[0])
            elif get_fitness(grey_child, template[0]) < global_fitness:
                best_cross_over_points = ((w1*best_cross_over_points + w2*grey_points)/(w1+w2)).astype(int)
                global_fitness = get_fitness(grey_child, template[0])
        else:
            sk = sketch_child
            if g == 0:
                best_cross_over_points = sketch_points

            elif get_fitness(sketch_child, template[0]) < global_fitness:
                best_cross_over_points = ((w1*best_cross_over_points + w2*sketch_points)/(w1+w2)).astype(int)

            global_fitness = get_fitness(sketch_child, template[0])
    x+=1

#
# plt.imshow(sk)
# plt.show()

# plt.imsave('img4.jpg', sk, cmap='gray' ,vmin=0, vmax=255)

# print(best_cross_over_points.shape)
# f = open( 'content.txt', 'w' )
# f.write( best_cross_over_points )
# f.close()

test_image=load_images_from_folder('test_photos')

# plt.imshow(test_image[9])
# plt.show()

test_template=load_images_from_folder('test_sketches')


# plt.imshow(test_template[1])
# plt.show()
f=0
test_sketches = []
while f<99:
    gray_img = grayscale(test_image[f])
    inverted_img = 255-gray_img
    blur_img = scipy.ndimage.filters.gaussian_filter(inverted_img,sigma=5)
    final_img= dodge(blur_img,gray_img)
    test_sketches.append(final_img)
    f+=1


# plt.imshow(final_img)
# plt.show()
# plt.imsave('img10.jpg', final_img, cmap='gray' ,vmin=0, vmax=255)
final_output = []
# best_cross_over_points = np.load('content.txt')
for d in test_sketches:
    fin=test_sketch(d,temp,best_cross_over_points)
    final_output.append(fin)
plt.imshow(final_output[9])
plt.show()
plt.imsave('img11.jpg', final_output[9], cmap='gray', vmin=0, vmax=255)
x=0
acc=0
n=0
while x<99:
    tmp = test_template[x]
    # print("Accuracy: ", 1 - (get_fitness(final_output[0], tmp[..., 0]) / (8 * tmp.shape[0] * tmp.shape[1])))
    if test_template[x].shape==final_output[x].shape:
        cur_acc=1 - (get_fitness(final_output[x], tmp) / (8 * tmp.shape[0] * tmp.shape[1]))
        plt.imshow(final_output[x])
        plt.show()
        print(cur_acc)
        acc+=cur_acc
        n+=1
    x+=1
print(acc/n)
# tmp = test_template[0]
# print(test_template[0].shape,test_template[4].shape)
# print("Accuracy: ", 1 - (get_fitness(final_output[0], tmp[...,0]) / (8 * tmp.shape[0] * tmp.shape[1])))
