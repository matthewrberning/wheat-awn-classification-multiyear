#matthew berning - GWU, 2021
import os
import sys
import time
import argparse


from utils.inspection_functions import build_conditional_montages
from utils.input_validation import yesno

def collect_arguments():
    parser = argparse.ArgumentParser(description="choose visualization to construct (make montages, or inspect datasets, etc.")
    parser.add_argument('--vis', type=str, default='montage', required=True)


    parser.add_argument('--model_name', type=str, default='vgg16', required=False)
    parser.add_argument('--model_pth', type=str, required=False)
    parser.add_argument('--data_csv', type=str, default='./data/2020_test_awns.csv', required=False)
    parser.add_argument('--fig_title', type=str, required=False)
    parser.add_argument('--find_incorrects', type=bool, default=True, required=False)
    parser.add_argument('--end_after', type=int, default=None, required=False)
    parser.add_argument('--batch_size', type=int, default=32, required=False)
    parser.add_argument('--verbose', type=bool, default=False, required=False)
    parser.add_argument('--pkl_file_path', type=str, default='./data/2020_plot-id_to_num_dict.pkl', required=False)
    parser.add_argument('--save_dir', type=str, default='./data/montages', required=False)
    parser.add_argument('--collect_class', type=str, default=None, required=False, choices=['0','1'])

    return parser.parse_args()


if __name__ == '__main__':

    #find the correct GPU -and use it!
    deviceIDs = GPUtil.getAvailable(order = 'first', 
                                    limit = 1, 
                                    maxLoad = 0.3, 
                                    maxMemory = 0.3, 
                                    includeNan=False, 
                                    excludeID=[], 
                                    excludeUUID=[])

    os.environ["CUDA_VISIBLE_DEVICES"] = str(deviceIDs[0])

    print("\n\nGPU Chosen: ", str(deviceIDs[0]),"\n\n")

    args = collect_arguments()

    if args.vis == 'montage':
        build_conditional_montages(args.data_csv,
                                   args.batch_size,
                                   args.fig_title,
                                   args.model_pth,
                                   args.pkl_file_path,
                                   args.model_name,
                                   args.end_after,
                                   args.find_incorrects,
                                   args.save_dir,
                                   args.verbose,
                                   args.collect_class)

    else:
        sys.exit("\n\n\n...that option doesn't exist yet\n\n\n")

