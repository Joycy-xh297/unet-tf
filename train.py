from model import *
from data import *
from dataPrepare import *

def mkdir(path):
    path = path.strip()
    path= path.rstrip("/")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' created successfully')
        return True
    else:
        print(path + ' already exist')
        return False

def main():
    mint_dir = "data/RR_mint/train/aug/"
    mkdir(mint_dir)

    ref_dir = "data/RR_mint/ref/"
    sub_dir1 = "mint1/"
    sub_dir2 = "mint2/"
    sub_dir3 = "mint3/"

    data_gen_args = dict(rotation_range=0.2,
                        width_shift_range=0.05,
                        height_shift_range=0.05,
                        shear_range=0.05,
                        zoom_range=0.05,
                        horizontal_flip=True,
                        fill_mode='nearest')
   
    size = (256, 256)
    ref_array = get_ref(ref_dir, sub_dir1, sub_dir2, sub_dir3, size) # should be of shape (3,73,h,w,3)
    # print("the shape of ref_array: ", [len(ref_array), ref_array[0].shape])
    # try by setting the generator batch size to 1 ?
    trainGene = trainGenerator(20,'data/RR_mint/train','image','mask',ref_array,data_gen_args,save_to_dir = mint_dir, target_size=size)

    # to visualise the data augmentation result
    # num_batch = 3
    # for i,batch in enumerate(myGenerator):
    #     if(i >= num_batch):
    #         break

    size = (256, 256, 3) # not sure if here we should use 3 or 1. according to the original structure we should use 1 but the instruction told 3
    model = unet(input_size=size)

    #change the hdf5 name below for training different models
    hdf5 = 'mint.hdf5'

    model_checkpoint = ModelCheckpoint(hdf5, monitor='loss',verbose=1, save_best_only=True)
    model.fit_generator(trainGene,steps_per_epoch=20,epochs=1,callbacks=[model_checkpoint]) #steps per epoch used to be 300

    test_dir = "data/RR_mint/test/img"
    test_num, _ = scan_file(test_dir)
    testGene = testGenerator(test_dir)
    results = model.predict_generator(testGene,num_image = test_num,verbose=1)
    saveResult(test_dir,results)

if __name__ == "__main__":
    main()