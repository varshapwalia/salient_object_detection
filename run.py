import argparse
import os
from dataset import get_loader
from solver import Solver

def main(cfg):
    # Train mode
    if cfg.mode == 'train':
        print("Initiating training!")
        
        # Obtain training data loader and dataset
        train_data_loader, train_dataset = get_loader(cfg.batch_size, num_thread=cfg.num_thread)
        
        # Set up directory structure for storing results
        experiment_name = "nnet"
        if not os.path.exists("%s/run-%s" % (cfg.save_fold, experiment_name)):
            os.mkdir("%s/run-%s" % (cfg.save_fold, experiment_name))  # Create directory for results
            os.mkdir("%s/run-%s/logs" % (cfg.save_fold, experiment_name))  # Create logs subdirectory
            os.mkdir("%s/run-%s/models" % (cfg.save_fold, experiment_name))  # Create models subdirectory
        cfg.save_fold = "%s/run-%s" % (cfg.save_fold, experiment_name)  # Update save_fold path
        
        # Initialize and execute solver for training
        print("Initializing solver...")
        trainer = Solver(train_data_loader, None, cfg)  # Solver initialization for training
        print("Training in progress...")
        trainer.train()  # Start training process
    
    # Test mode
    elif cfg.mode == 'test':
        # Obtain test data loader and dataset
        test_data_loader, test_dataset = get_loader(cfg.test_batch_size, mode='test', num_thread=cfg.num_thread, test_mode=cfg.test_mode, sal_mode=cfg.sal_mode)

        # Initialize and perform testing
        tester = Solver(None, test_data_loader, cfg, test_dataset.save_folder())  # Solver initialization for testing
        tester.test(test_mode=cfg.test_mode)  # Execute test
    
    # Invalid mode
    else:
        raise IOError("Invalid mode specified!")  # Error for invalid mode

if __name__ == '__main__':
    print("Setting up model paths...")
    # Paths for pre-trained models
    resnet_model_path = './resnet50_caffe.pth'

    # Argument parsing
    arg_parser = argparse.ArgumentParser()

    # Hyper-parameters
    arg_parser.add_argument('--n_color', type=int, default=3)
    arg_parser.add_argument('--cuda', type=bool, default=True)

    # Training settings
    arg_parser.add_argument('--resnet', type=str, default=resnet_model_path)
    arg_parser.add_argument('--epoch', type=int, default=30) # 12, now x3
    arg_parser.add_argument('--batch_size', type=int, default=1)
    arg_parser.add_argument('--test_batch_size', type=int, default=1)
    arg_parser.add_argument('--num_thread', type=int, default=4)
    arg_parser.add_argument('--load_bone', type=str, default='')
    arg_parser.add_argument('--save_fold', type=str, default='./EGNet')
    arg_parser.add_argument('--epoch_save', type=int, default=1) # 2, now x3
    arg_parser.add_argument('--epoch_show', type=int, default=1)
    arg_parser.add_argument('--pre_trained', type=str, default=None)

    # Testing settings
    arg_parser.add_argument('--model', type=str, default='./epoch_resnet.pth')
    arg_parser.add_argument('--test_fold', type=str, default='./results/test')
    arg_parser.add_argument('--test_mode', type=int, default=1)
    arg_parser.add_argument('--sal_mode', type=str, default='t')

    # Miscellaneous
    arg_parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    arg_parser.add_argument('--visdom', type=bool, default=False)

    config = arg_parser.parse_args()
  
    # Create save directory if it doesn't exist
    if not os.path.exists(config.save_fold):
        os.mkdir(config.save_fold)
    print("Arguments parsed successfully!")
    # Execute main function with parsed config
    main(config)
