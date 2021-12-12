#matthew berning - GWU, 2021
import os
import sys
import time
import argparse
import GPUtil


from utils.inspection_functions import build_conditional_montages, iterate_plotting
from utils.input_validation import yesno

def collect_arguments():
    parser = argparse.ArgumentParser(description="choose visualization to construct (make montages, montage_poll_mistakes, or, etc.")
    parser.add_argument('--vis', type=str, default='montage', required=True)


    parser.add_argument('--model_name', type=str, default='vgg16', required=False) #montages, montage_poll_mistakes
    parser.add_argument('--model_pth', type=str, required=False) #montages, montage_poll_mistakes
    parser.add_argument('--data_csv', type=str, default='./data/2020_test_awns.csv', required=False) #montages, montage_poll_mistakes
    parser.add_argument('--fig_title', type=str, required=False)
    parser.add_argument('--find_incorrects', type=bool, default=True, required=False)
    parser.add_argument('--end_after', type=int, default=None, required=False)
    parser.add_argument('--batch_size', type=int, default=32, required=False) #montages, montage_poll_mistakes
    parser.add_argument('--verbose', type=bool, default=False, required=False) #montages, montage_poll_mistakes
    parser.add_argument('--plot_id_pkl_file_path', type=str, default='./data/2020_plot-id_to_num_dict.pkl', required=False) #montages, montage_poll_mistakes
    parser.add_argument('--date_pkl_file_path', type=str, default='./data/2020_date_to_num_dict.pkl', required=False) #montage_poll_mistakes
    parser.add_argument('--save_dir', type=str, default='./data/montages', required=False) #montages, montage_poll_mistakes
    parser.add_argument('--collect_class', type=str, default=None, required=False, choices=['0','1'])

    parser.add_argument('--voting_method', type=str, default='plot', required=False) #montage_poll_mistakes

    return parser.parse_args()


if __name__ == '__main__':

    #find the correct GPU -and use it!
    deviceIDs = GPUtil.getAvailable(order = 'first', 
                                    limit = 1, 
                                    maxLoad = 0.1, 
                                    maxMemory = 0.1, 
                                    includeNan=False, 
                                    excludeID=[], 
                                    excludeUUID=[])

    os.environ["CUDA_VISIBLE_DEVICES"] = str(deviceIDs[0])

    print("\n\nGPU Chosen: ", str(deviceIDs[0]),"\n\n")

    args = collect_arguments()

    for arg in vars(args):
        print(arg, getattr(args, arg))

    if yesno("are the visualization conditions above correct?"):
        if args.vis == 'montage':
            build_conditional_montages(data_csv=args.data_csv,
                                       batch_size=args.batch_size,
                                       fig_title=args.fig_title,
                                       model_pth=args.model_pth,
                                       pkl_file_path=args.plot_id_pkl_file_path,
                                       model_name=args.model_name,
                                       end_after=args.end_after,
                                       find_incorrects=args.find_incorrects,
                                       save_dir=args.save_dir,
                                       verbose=args.verbose,
                                       collect_class=args.collect_class)

        if args.vis == 'montage_poll_mistakes':
            iterate_plotting(model_name=args.model_name, 
                             model_pth=args.model_pth, 
                             plot_id_dict_path=args.plot_id_pkl_file_path, 
                             date_dict_path=args.date_pkl_file_path,
                             data_csv=args.data_csv, 
                             voting_method=args.voting_method,
                             save_dir=args.save_dir,
                             batch_size=args.batch_size,
                             verbose=args.verbose)


        else:
            sys.exit("\n\n\n...that option doesn't exist yet\n\n\n")
    else:
        sys.exit("\n\n\n...\n\n\n")


