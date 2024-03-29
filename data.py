from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os, os.path
import glob
import skimage.io as io
import skimage.transform as trans
import tensorflow as tf

ng = [255, 255, 255]
bg = [0, 0, 0]

COLOR_DICT = np.array([ng, bg])

def get_ref(ref_dir, sub_dir1, sub_dir2, sub_dir3, size):
    print('----------------------reading refs from file-------------------------')
    path1 = ref_dir + sub_dir1
    path2 = ref_dir + sub_dir2
    path3 = ref_dir + sub_dir3
    collection1 = []
    collection2 = []
    collection3 = []
    for file in glob.glob(os.path.join(path1, '*.png')):
        img = io.imread(file)
        img = trans.resize(img, size)
        collection1.append(img)
    collection1 = np.array(collection1)
    # print("collection1 shape: ", collection1.shape)
    for file in glob.glob(os.path.join(path2, '*.png')):
        img = io.imread(file)
        img = trans.resize(img, size)
        collection2.append(img)
    collection2 = np.array(collection2)
    # print("collection2 shape: ", collection2.shape)
    for file in glob.glob(os.path.join(path3, '*.png')):
        img = io.imread(file)
        img = trans.resize(img, size)
        collection3.append(img)
    collection3 = np.array(collection3)
    # print("collection3 shape: ", collection3.shape)
    print("ref read finished")
    return np.array([collection1, collection2, collection3])

def which_mint(name):
    print("looking for : ", name)
    mint1 = [line.rstrip('\n') for line in open("data/RR_mint/mint1.txt")]
    # print("mint1 list: ", mint1)
    mint2 = [line.rstrip('\n') for line in open("data/RR_mint/mint2.txt")]
    # print("mint2 list: ", mint2) 
    mint3 = [line.rstrip('\n') for line in open("data/RR_mint/mint3.txt")]
    # print("mint3 list: ", mint3)
    for i,mints in enumerate([mint1, mint2, mint3]):
        for mint in mints:
            if mint.endswith(name):
                # print("!!! find it !!!")
                return i
            else:
                continue

            



def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)


# add ref to the train and test generator
def trainGenerator(batch_size,train_path,image_folder,mask_folder,ref_array,aug_dict,image_color_mode = "rgb",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1): # img color mode changed to rgb
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    print("------------------trainGenerator initiated-----------------------")
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    # ref_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    # print("file names read from the image generator: ", image_generator.filenames)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    # ref_generator = ref_datagen.flow_from_directory(
    #     train_path,
    #     classes = [ref_folder],
    #     class_mode = None,
    #     color_mode = ref_color_mode,
    #     target_size = target_size,
    #     batch_size = batch_size,
    #     save_to_dir = save_to_dir,
    #     save_prefix  = ref_save_prefix,
    #     seed = seed)
    train_generator = zip(image_generator, mask_generator)
    print("------------------------start concatenating---------------------------")
    for (img,mask) in train_generator:
        idx = (image_generator.batch_index - 1) * image_generator.batch_size
        img_name = image_generator.filenames[idx : (idx + image_generator.batch_size)]
        for i in range(image_generator.batch_size):
            name = img_name[i]
            # first get the index 
            name1 = name[:-4] #remove .png
            if name1[-2] == '_':
                index = name1[-1]
            else:
                index = name1[-2:]
            index = int(index)
        
            print("index: ", index)
            # then get which mint it is
            mint = which_mint(name) # the index in the ref_array (where to get the ref)
            print("mint: ", mint)
            print("ref_array_shape: ", ref_array.shape)
            ref = ref_array[mint,index,:,:,:]
            print("ref_shape: ", ref.shape)
            ref = np.array([ref])
            print("ref_shape: ", ref.shape)
            print("image shape: ", img[i].shape)

            # concatenate img and ref
            img = tf.concat(img,ref)
            
        # print(img_name)
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        print(mask.shape)
        yield (img,mask)



def testGenerator(test_path,num_image = 30,target_size = (256,256),flag_multi_class = False,as_gray = False): #as_gray was True
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img



def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr


def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255



def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)