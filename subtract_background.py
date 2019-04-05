from util import *
from ast import literal_eval
import pyfftw

def find_shift(first_image, second_image, scale = 0.25):
    rescaled_first_image = rescale(first_image, scale)
    rescaled_second_image = rescale(second_image, scale)
    print(rescaled_first_image.shape)
    print(rescaled_second_image.shape)
    shift, error, diffphase = register_translation(rescaled_first_image, rescaled_second_image)
    return shift / scale

def apply_shifts(image, shift):
    offset_image = []
    for channel_index in range(image.shape[2]):
        print(image[:, :, channel_index].shape)
        print(image[:, :, channel_index].dtype)
        print("Calculating Fourier transform...")
        #transformed_slice = np.fft.fftn(image[:, :, channel_index])
        transformed_slice = pyfftw.interfaces.numpy_fft.fftn(image[:, :, channel_index])
        print("Shifting transformed image...")
        offset_im = fourier_shift(transformed_slice, shift)
        print(offset_im.shape)
        print(offset_im.dtype)
        offset_image.append(pyfftw.interfaces.numpy_fft.ifftn(offset_im))
        #offset_image.append(np.fft.ifftn(offset_im))
    offset_image = np.transpose(offset_image, axes = [1, 2, 0]).astype(np.float32)
    return offset_image

def subtract_background(tissue_directory_regex, ordered_channels, background_scaling_factors, blank_round_number):
    round_subdirectory_regex = "round*/"
    blank_round_string = "round%d/" % blank_round_number
    stitched_filename = "stitched.tiff"
    background_subtracted_filename = "background_subtracted.tiff"

    ordered_factors = np.array([background_scaling_factors[channel] for channel in ordered_channels])

    tissue_directories = glob.glob(tissue_directory_regex)

    for tissue_directory in tissue_directories:
        print(tissue_directory)
        round_directory_regex = tissue_directory + round_subdirectory_regex
        round_directories = glob.glob(round_directory_regex)
        blank_round_directory = os.path.join(tissue_directory, blank_round_string)
        blank_round_image_filepath = os.path.join(blank_round_directory, stitched_filename)
        blank_round_image = imageio.imread(blank_round_image_filepath)
        blank_round_dapi = blank_round_image[:, :, 0]

        for round_directory in round_directories:
            print(round_directory)
            if round_directory == blank_round_directory:
                continue
            round_image_filepath = os.path.join(round_directory, stitched_filename)
            round_image = imageio.imread(round_image_filepath)
            round_dapi = round_image[:, :, 0]
            
            print(round_image.shape)
            print(round_image.dtype)
            print(blank_round_image.shape)
            print(blank_round_image.dtype)

            print("Finding shift...")
            shift = find_shift(round_dapi, blank_round_dapi, scale = 0.1)
            print("Applying shift...")
            shifted_blank_image = apply_shifts(blank_round_image, shift)

            background_subtracted_image = np.clip(round_image - shifted_blank_image * ordered_factors, 0, None)
            background_subtracted_image[:, :, 0] = round_dapi

            background_subtracted_image = background_subtracted_image.astype(np.uint16)

            print(background_subtracted_image.dtype)

            new_filepath = os.path.join(round_directory, background_subtracted_filename)
            imageio.imwrite(new_filepath, background_subtracted_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--basepath', help='Path to directory with subdirs for parsed images in each tissue')
    #parser.add_argument('--tissues', help='Comma-separated list of tissue numbers to include')
    #parser.add_argument('--blank-round-number', help='Round in which blank images were collected')
    #parser.add_argument('--dapi-index', help='Channel index for DAPI images',type=int,default=3)
    parser.add_argument('--tissue-directory-regex', help="Regex that matches tissue directories.")
    parser.add_argument('--ordered-channels', help='Channels, in order from lowest to highest frequency.', nargs='+')
    parser.add_argument('--background-scaling-factors', help='Factors to multiply blanks before subtraction (by channel)',default='1.5,1.15,1.35')
    parser.add_argument('--blank-round-number', help="Round that contains blank images.", type=int)
    #parser.set_defaults(save_tiles=False)
    #parser.set_defaults(save_stitched=False)
    args,_ = parser.parse_known_args()

    subtract_background(args.tissue_directory_regex, args.ordered_channels, args.background_scaling_factors, args.blank_round_number)

    #for blank_directory in blank_directories:
    #    tissue_directory = blank_directory[:round_directory.rfind("/round")]
    #    blank_directory_map = [round_directory for round_directory in round_directories if round_directory.startswith(tissue_directory)]

    #for t in args.tissues.split(','):
    #   tissue = 'tissue%s' % t
    #   FP_blanks = glob.glob(os.path.join('%s/%s/%s/stitched.tiff' % (args.basepath, tissue, blank_round_subdirectory)))
    #   blanks = [imageio.imread(fp) for fp in FP_blanks]
    #   filepaths = glob.glob(os.path.join('%s/%s/round*/stitched.tiff' % (args.basepath, tissue)))
    #   filepaths = [filepath for filepath in filepaths if filepath not in FP_blanks]
    #   images = [imageio.imread(filepath) for filepath in filepaths]
    #    # Stuff from here...
    #   min_shape = (min(image.shape[0] for image in images + blanks), min(image.shape[1] for image in images+blanks))
    #   images = [im[:min_shape[0], :min_shape[1]] for im in images]
    #   blanks = [im[:min_shape[0], :min_shape[1]] for im in blanks]
    #   # ... to here is unneeded with already cropped images
    #    dapi_images = [im[:, :, args.dapi_index] for im in images]
    #   dapi_blank_images = [im[:, :, args.dapi_index] for im in blanks]
    #   shifts = [find_shift(dapi_image, dapi_blank_images[0]) for dapi_image in dapi_images]
    #   max_value = np.iinfo(images[0].dtype).max
    #   print(blanks)
    #    blank = blanks[0]
    #    print(blanks.shape)

    #   # setting the 1% brightest spots to the max, so that they will certainly be set to zero
    #   thresh = np.percentile(blank, 99, axis=0)
    #   blank[blank > thresh] = max_value
    #   for i,im in enumerate(images):
    #       shifted_blanks = apply_shifts(blank, shifts[i])
    #       im_new = im - shifted_blanks*factors
    #       im_new[im_new < 0] = 0
    #       im_new = np.rint(im_new).astype(im.dtype)
    #       im_new[:, :, args.dapi_index] = im[:, :, args.dapi_index]
    #       imageio.imwrite(filepaths[i].replace('.tiff', '.background_subtract.tiff'), im_new)
    #   print(tissue)
