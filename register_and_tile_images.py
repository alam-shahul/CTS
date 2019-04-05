from collections import defaultdict
from util import *

def find_all_shifts(ref_image, other_images):
    shifts = []
    for other_image in other_images:
        shift = find_shift(ref_image, other_image, scale=0.25)
        shifts.append(shift)
    return shifts

#def apply_shifts(im, shift):
#    offset_image = []
#    for i in range(im.shape[2]):
#        offset_im = fourier_shift(np.fft.fftn(im[:,:,i]), shift)
#        offset_image.append(np.fft.ifftn(offset_im))
#    offset_image = np.transpose(offset_image,axes=[1,2,0]).astype(np.float32)
#    return offset_image

#def crop_all_shifted(images, shifts):
#    min_s = np.min(shifts, axis=0).astype(np.int)
#    max_s = np.max(shifts, axis=0).astype(np.int)
#    if min_s[0] < 0:
#        images = images[:min_s[0]]
#    if max_s[0] > 0:
#        images = images[max_s[0]:]
#    if min_s[1] < 0:
#        images = images[:,:min_s[1]]
#    if max_s[1] > 0:
#        images = images[:,max_s[1]:]
#    return images

def crop_all_shifted(images, shifts):
    print(images.shape)
    min_y_shift, min_x_shift = np.min(shifts,axis=0).astype(np.int)
    max_y_shift, max_x_shift = np.max(shifts,axis=0).astype(np.int)
    if min_y_shift < 0:
        images = images[:, :min_y_shift]
    if max_y_shift > 0:
        images = images[:, max_y_shift:]
    if min_x_shift < 0:
        images = images[:, :, :min_x_shift]
    if max_x_shift > 0:
        images = images[:, :, max_x_shift:]
    return images

def normalize_image_scale(im,max_value,thresh_each=False):
    if thresh_each:
        for i in range(im.shape[-1]):
            thresh = np.percentile(im,max_value)
            im[:,:,i][im[:,:,i] > thresh] = thresh
            im[:,:,i] = im[:,:,i]/thresh
        return im.astype(np.float32)
    else:
        return (im/max_value).astype(np.float32)

def break_into_tiles(image, tile_size):
    rows, columns, _ = image.shape
    tiles = []
    tile_pattern = []
    tile_index = 0
    for row in range(0, rows, tile_size):
        tile_row = []
        for column in range(0, columns, tile_size):
            cutout = image[row:row+tile_size, column:column+tile_size]
            actual_height, actual_width = cutout.shape
            tile = np.pad(cutout, [(0, tile_size - actual_height), (0, tile_size - actual_width), (0,0)], 'constant')
            tiles.append(tile)
            tile_row.append(tile_index)
            tile_index += 1
        tile_pattern.append(tile_row)
    tiles = np.array(tiles)
    tile_pattern = np.array(tile_pattern)    

    return tiles, tile_pattern

def register_and_tile_images(tissue_directory_regex, tile_size, blank_round_number):
    round_subdirectory_regex = "round*/"
    blank_round_string = "round%d/" % blank_round_number
    corrected_filename = "flat_field_corrected.tiff"
    registered_filename = "registered.tiff"

    tissue_directories = glob.glob(tissue_directory_regex)

    Images = []
    for tissue_directory in tissue_directories:
        print(tissue_directory)
        round_directory_regex = tissue_directory + round_subdirectory_regex
        round_directories = glob.glob(round_directory_regex)
        blank_round_directory = os.path.join(tissue_directory, blank_round_string)

        for round_directory in round_directories:
            if round_directory == blank_round_directory:
                continue
            round_image_filepath = os.path.join(round_directory, corrected_filename)
            round_image = imageio.imread(round_image_filepath)
            Images.append(round_image)
        
        min_shape = (min(im.shape[0] for im in Images), min(im.shape[1] for im in Images))
        Images = np.array([im[:min_shape[0],:min_shape[1]] for im in Images])
       
        # TODO: How to calculate dapi index? It's zero for now
        dapi_slices = Images[:, :, :, 0]
        first_dapi = dapi_slices[0]
        shifts = find_all_shifts(first_dapi, dapi_slices)
        print("Shifts calculated.")
        #ImagesAligned = []

        #for i, im in enumerate(Images):
        #    if i == 0:
        #        ImagesAligned.append(im.astype(np.float32))
        #    else:
        #        ImagesAligned.append(apply_shifts(im,shifts[i-1]))

        ImagesAligned = np.array([apply_shifts(Images[index], shifts[index]) for index in range(len(Images))])

        
        #ImagesAligned = np.array([crop_all_shifted(images, shifts) for images in ImagesAligned])
        ImagesAligned = crop_all_shifted(ImagesAligned, shifts)

        for round_index, round_directory in enumerate(round_directories):
            print(round_directory)
            round_output_filepath = os.path.join(round_directory, registered_filename)
            round_output_image = ImagesAligned[round_index]
            imageio.imwrite(round_output_filepath, ImagesAligned[round_index])
            #max_value = np.iinfo(Images[0].dtype).max
            #round_output_image = normalize_image_scale(round_output_image, max_value)

        #for image, filepath in zip(ImagesAligned, filepaths):
        #    round = filepath.split('/')[-2]
        #    channels = Rounds2Channels[(tissue, round)]
        #    if (round == 'round1') and ('DAPI' not in channels):
        #        channels.insert(args.dapi_index, 'DAPI')
        #    for i,c in enumerate(channels):
        #        imageio.imwrite('%s/stitched_aligned_filtered/%s.tiff' % (tissue_directory, c), np.rint(image[:,:,i]).astype(Images[0].dtype))
        #        #imageio.imwrite('%s/%s/stitched_aligned_filtered/%s.tiff' % (args.basepath, tissue, c), np.rint(im[:,:,i]).astype(Images[0].dtype))

        #_=os.system('mkdir %s/%s/arrays_aligned_filtered' % (args.basepath,tissue))
        # for round_index, round_directory in enumerate(round_directories):
            os.makedirs(os.path.join(round_directory, "tiles"), exist_ok=True)
#        for image,filepath in zip(ImagesAligned,filepaths):
#            max_value = np.iinfo(Images[0].dtype).max
#            image = normalize_image_scale(image,max_value)
#            chan_idx = np.where(image.reshape([-1,image.shape[-1]]).sum(0) > 0)[0]
#            im = image[:,:,chan_idx]
#            round = filepath.split('/')[-2]
#            channels = Rounds2Channels[(tissue,round)]
#            if (round == 'round1') and ('DAPI' not in channels):
#                channels.insert(args.dapi_index,'DAPI')
            tiles, new_fov_pattern = break_into_tiles(ImagesAligned[round_index], tile_size)
            for tile, new_fov in zip(tiles, new_fov_pattern.flatten()):
                array_filepath = os.path.join(round_directory, "tiles", "fov_%d.npy" % new_fov)
                np.save(array_filepath, tile)
                image_filepath = os.path.join(round_directory, "tiles", "fov_%d.tiff" % new_fov)
                np.save(image_filepath, tile)
            np.save(os.path.join(round_directory, "tiles", "modified_fov_pattern.npy"), new_fov_pattern)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tile-size', help='Size of output images',type=int,default=1024)
    parser.add_argument('--tissue-directory-regex', help='Path to directory with subdirs for parsed images in each tissue')
    parser.add_argument('--blank-round-number', help='Path to directory with subdirs for parsed images in each tissue')
    parser.set_defaults(save_tiles=False)
    parser.set_defaults(save_stitched=False)
    args, _ = parser.parse_known_args()

    register_and_tile_images(args.tissue_directory_regex, args.tile_size, args.blank_round+number)
