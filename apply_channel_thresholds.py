from util import *

def apply_threshold_and_rescale(channel_slice, lower_threshold, upper_threshold):
    datatype = channel_slice.dtype
    max_value = np.iinfo(datatype).max

    channel_slice = np.clip(channel_slice, lower_threshold, upper_threshold)

    channel_slice = channel_slice - lower_threshold
    thresholded_slice = np.rint(channel_slice/channel_slice.ptp() * max_value).astype(datatype)

    return thresholded_slice

def apply_channel_thresholds(tissue_directory_regex, ordered_channels, channel_threshold_filepath, blank_round_number): 
    round_subdirectory_regex = "round*/"
    blank_round_string = "round%d/" % blank_round_number
    background_subtracted_filename = "background_subtracted.tiff"
    thresholded_filename = "thresholded.tiff"

    with open(channel_threshold_filepath) as f:
        channel_thresholds = json.load(f)

    tissue_directories = glob.glob(tissue_directory_regex)

    for tissue_directory in tissue_directories:
        round_directory_regex = tissue_directory + round_subdirectory_regex
        round_directories = glob.glob(round_directory_regex)
        blank_round_directory = os.path.join(tissue_directory, blank_round_string)

        for round_directory in round_directories:
            if round_directory == blank_round_directory:
                continue
            print(round_directory)
            image_filepath = os.path.join(round_directory, background_subtracted_filename)
            
            image = imageio.imread(image_filepath)
            thresholded_image = np.zeros(image.shape, dtype=image.dtype)
            new_im = np.zeros(image.shape,dtype=image.dtype)
            for channel_index, channel in enumerate(ordered_channels):
                thresholded_slice = apply_threshold_and_rescale(image[:, :, channel_index], channel_thresholds[channel]["lower_threshold"], channel_thresholds[channel]["upper_threshold"])
                new_im[:,:,channel_index] = thresholded_slice
            image_outpath = os.path.join(round_directory, thresholded_filename)
            imageio.imwrite(image_outpath, new_im)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--basepath', help = 'Path to directory with subdirs for parsed images in each tissue')
    parser.add_argument('--ordered-channels', help='Channels, in order from lowest to highest frequency.', nargs='+')
    parser.add_argument('--channel-threshold-filepath', help = 'File containing channel min and max thresholds')
    parser.add_argument('--thresholds', help = 'File containing channel min and max thresholds')
    parser.add_argument('--blank-round-number', help="Round that contains blank images.", type=int)
    parser.set_defaults(keep_existing = False)
    args,_ = parser.parse_known_args()

    tissue_directory_regex = "../ProcessedImages/20181016/ThresholdingTestRawImages/tissue*/images/"
    channel_threshold_filepath = args.channel_threshold_filepath

    apply_channel_thresholds(args.tissue_directory_regex, args.ordered_channels, args.channel_threshold_filepathargs.blank_round_number)

    #tissue_directories = glob.glob(tissue_directory_regex)[:1]
    #f = open(args.thresholds)
    #thresholds = [line.strip().split() for line in f]
    #f.close()
    #thresholds = [[int(t[0]),int(t[1])] for t in thresholds]
    #FP = glob.glob(os.path.join(args.basepath,'*','*','stitched.tiff'))
    #for fp in FP:
    #    outfile = fp.replace('.tiff','.threshold.tiff')
    #    if (args.keep_existing and not os.path.isfile(outfile)) or (not args.keep_existing):
    #        im = imageio.imread(fp)
    #        new_im = np.zeros(im.shape,dtype=im.dtype)
    #        for i in range(im.shape[-1]):
    #            im_i = apply_threshold_and_scale(im[:,:,i],thresholds[i][0],thresholds[i][1])
    #            new_im[:,:,i] = im_i
    #        imageio.imwrite(outfile,new_im)
    #        print(fp)
