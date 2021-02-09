import os
import random
import shutil


cwd = os.getcwd()
folder_names = ['angelica', 'courtney', 'olive', 'tim']
train_folder = 'ss_train'
test_folder = 'ss_test'


def move_files(working_dir):
    os.chdir(working_dir)

    # get all image names in dir
    image_names = os.listdir(working_dir)
    n_elements = len(image_names)//len(folder_names)

    for i, fn in enumerate(folder_names):
        img_group = random.sample(image_names, n_elements)
        destination = os.path.join(os.getcwd(), fn)
        if not os.path.exists(fn):
            os.mkdir(fn)

        for img in img_group:
            image_names.remove(img)
            shutil.move(img, destination)

move_files(os.path.join(cwd, train_folder))
move_files(os.path.join(cwd, test_folder))


