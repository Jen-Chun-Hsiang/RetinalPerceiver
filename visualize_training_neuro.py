import os
from datetime import datetime
import logging
import torch

from utils.training_procedure import CheckpointLoader, forward_model
# (0) identify the cell we have modeled their responses
# (1) show the receptive field with the white noise
# (2) show the response to the test set


def main():
    stimulus_type = '200ktl2011ks111sd'  # get the name from the check point folder
    epoch_end = 200  # the number of epoch in the check_point file
    is_full_figure_draw = True # determine whether draw for each neuro or just get stats
    saveprint_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/Prints/'
    checkpoint_folder = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RetinalPerceiver/Results/CheckPoints/'

    # Compile the regarding parameters
    checkpoint_filename = f'PerceiverIO_20tp{stimulus_type}_checkpoint_epoch_{epoch_end}'
    checkpoint_path = os.path.join(checkpoint_folder, f'{checkpoint_filename}.pth')

    timestr = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(saveprint_dir, f'{checkpoint_filename}_training_log_{timestr}.txt')
    # Setup logging
    logging.basicConfig(filename=log_filename,
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)s:%(message)s')
    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU and CUDA installation.")
    device = torch.device("cuda")

    # Load the training and model parameters
    checkpoint_loader = CheckpointLoader(checkpoint_path=checkpoint_path, device=device)
    args = checkpoint_loader.load_args()



if __name__ == "__main__":
    main()