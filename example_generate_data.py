import numpy as np
import nibabel as nib
import os

def create_square(x, y, z, size=10):
    return slice(x-size, x+size), slice(y-size, y+size), slice(z-size, z+size)

def create_ellipse(center_x, center_y, center_z, a=30, b=30, c=30, shape=(100, 100, 100)):
    mask = np.zeros(shape)
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                if ((x-center_x)**2 / a**2 + (y-center_y)**2 / b**2 + (z-center_z)**2 / c**2) <= 1:
                    mask[x, y, z] = 1
    return mask

def generate_fake_data(out_folder, n_subjects, train=False, name="fake_data", seed=0):
    np.random.seed(seed)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    for i in range(n_subjects):
        img = np.zeros((100, 100, 100))
        x, y, z = 45, 45, 45
        img[create_square(x, y, z, size=20)] = 1

        if train:
            a, b, c = np.random.randint(8, 12, size=3)
            img[create_ellipse(x, y, z, a=a, b=b, c=c, shape=img.shape).astype(bool)] = 0.6

        noise = np.random.normal(0, 0.1, size=img.shape)
        img += noise
        img = np.clip(img, 0, 1)

        img_nii = nib.Nifti1Image(img.astype(np.float32), np.eye(4))
        nib.save(img_nii, os.path.join(out_folder, f"{name}_{i}.nii"))
        print (f"Saved {name}_{i}.nii")

def make_mask(out_folder="fake_data", radius=15):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    mask = create_ellipse(50, 50, 50, radius, radius, radius, shape=(100, 100, 100)).astype(np.float32)
    nib.save(nib.Nifti1Image(mask, np.eye(4)), os.path.join(out_folder, f"mask.nii"))
        

if __name__ == '__main__':
    out_folder = 'example_data'

    make_mask(out_folder=out_folder, radius=25)

    os.makedirs(os.path.join(out_folder, 'training', 'train'), exist_ok=True)
    os.makedirs(os.path.join(out_folder, 'training', 'control'), exist_ok=True)
    os.makedirs(os.path.join(out_folder, 'validation', 'train_validation'), exist_ok=True)
    os.makedirs(os.path.join(out_folder, 'validation', 'control_validation'), exist_ok=True)

    generate_fake_data(os.path.join(out_folder, 'training', 'train'), 15, train=True, name="train")
    generate_fake_data(os.path.join(out_folder, 'training', 'control'), 10, train=False, name="control")
    generate_fake_data(os.path.join(out_folder, 'validation', 'train_validation'), 5, train=True, name="train_validation")
    generate_fake_data(os.path.join(out_folder, 'validation', 'control_validation'), 5, train=False, name="control_validation")

