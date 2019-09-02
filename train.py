from model import *
from data import *

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

    data_gen_args = dict(rotation_range=0.2,
                        width_shift_range=0.05,
                        height_shift_range=0.05,
                        shear_range=0.05,
                        zoom_range=0.05,
                        horizontal_flip=True,
                        fill_mode='nearest')
    myGenerator = trainGenerator(20,'data/RR_mint/train','image','mask',data_gen_args,save_to_dir = mint_dir)

    # to visualise the data augmentation result
    # num_batch = 3
    # for i,batch in enumerate(myGenerator):
    #     if(i >= num_batch):
    #         break

if __name__ == "__main__":
    main()