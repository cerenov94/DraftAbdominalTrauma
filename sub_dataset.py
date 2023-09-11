import pydicom.filereader
import torch
import torchio as tio
import pandas as pd
from configs import Config
from torch.utils.data import Dataset
import numpy as np
import os
from skimage.exposure import rescale_intensity
import dicomsdl as dicom


x_train = pd.read_csv('/media/cerenov/C/x_train.csv')
x_valid = pd.read_csv('/media/cerenov/C/x_valid.csv')

def get_do_separate_z(spacing, anisotropy_threshold=3):
    do_separate_z = (np.max(spacing) / np.min(spacing)) > anisotropy_threshold
    return do_separate_z


def get_lowres_axis(new_spacing):
    axis = np.where(max(new_spacing) / np.array(new_spacing) == 1)[0]  # find which axis is anisotropic
    return axis




class AbdominalDataset(Dataset):
    def __init__(self,train_df,transformer = None):
        self.train_df = train_df
        self.transformer = transformer
        #self.rescale_intensity = tio.RescaleIntensity((0,1),in_min_max=(-100,500))



    def dicom_to_image(self,dicom_image):
        """
        Read the dicom file and preprocess appropriately.
        """
        pixel_array = dicom_image.pixelData(storedvalue = True)

        # if dicom_image.PixelRepresentation == 1:
        #     bit_shift = dicom_image.BitsAllocated - dicom_image.BitsStored
        #     dtype = pixel_array.dtype
        #     new_array = (pixel_array << bit_shift).astype(dtype) >> bit_shift
        #     pixel_array = pydicom.pixel_data_handlers.util.apply_modality_lut(new_array, dicom_image)
        #
        # if dicom_image.PhotometricInterpretation == "MONOCHROME1":
        #     pixel_array = 1 - pixel_array

        # transform to hounsfield units
        intercept = dicom_image.RescaleIntercept
        slope = dicom_image.RescaleSlope
        pixel_array = pixel_array * slope + intercept
        #window_center,window_width,intercept,slope = get_windowing(dicom_image)

        # # windowing
        window_center = int(dicom_image.WindowCenter)
        #window_width = int(dicom_image.WindowWidth)
        window_width = 350
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        pixel_array = pixel_array.copy()
        pixel_array[pixel_array < img_min] = img_min
        pixel_array[pixel_array > img_max] = img_max

        # # normalization
        pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min())


        return (pixel_array * 255).astype(np.uint8)

    def load_scan(self,path):
        slices = [dicom.open(path + '/' + s) for s in os.listdir(path)]
        slices.sort(key=lambda x: int(x.InstanceNumber))
        return slices

    def resample_img_cucim(self,img, zoom=0.5, order=0, nr_cpus=-1):
        """
        Completely speedup of resampling compare to non-gpu version not as big, because much time is lost in
        loading the file and then in copying to the GPU.

        For small image no significant speedup.
        For large images reducing resampling time by over 50%.

        On our slurm gpu cluster it is actually slower with cucim than without it.
        """
        import cupy as cp
        from skimage.transform import resize

        #img = cp.asarray(img)  # slow
        new_shape = (np.array(img.shape) * zoom).round().astype(np.int32)

        resampled_img = resize(img, output_shape=new_shape, order=order, mode="edge", anti_aliasing=False).astype(np.int16)


        #resampled_img = cp.asnumpy(resampled_img)
        return resampled_img


    def __getitem__(self, item):
        series_path = self.train_df.loc[item,'series_path']
        labels = self.train_df.loc[item, Config.target_columns].values.astype(np.float32)

        slices = self.load_scan(series_path)

        voxel = np.expand_dims(np.transpose(np.stack(self.dicom_to_image(x) for x in slices),(1,2,0)),axis=0)
        zoom = np.array((1,256,256,380)) / voxel.shape
        reshaped_final_data = []

        voxel = self.resample_img_cucim(voxel,zoom,order=0)
        vmin, vmax = np.percentile(voxel, q=(0.5, 99.5))
        voxel = rescale_intensity(voxel,in_range=(vmin,vmax),out_range=np.int16)
        voxel = (voxel-voxel.mean())/voxel.std()

        if self.transformer:
              return self.transformer(torch.from_numpy(voxel).to(torch.float32)),labels
        return torch.from_numpy(voxel).to(torch.float32),labels


    def __len__(self):
        return len(self.train_df)




training_transform = tio.Compose([
    tio.RandomBlur(p=0.3),
    tio.RandomNoise(p=0.3),
    tio.RandomFlip(axes=('RAS'),p=0.3),
    tio.OneOf({
        tio.RandomSpike():1,
        tio.RandomGhosting():1,
    },p=0.5),
    tio.OneOf({
        tio.RandomElasticDeformation():1,
        tio.RandomAffine():1,
    },p = 0.5)
])



train_ds = AbdominalDataset(x_train,training_transform)
valid_ds = AbdominalDataset(x_valid)

