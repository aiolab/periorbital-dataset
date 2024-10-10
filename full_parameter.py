import argparse

def str2bool(v):
    return v.lower() in ('true')

def get_parameters():

    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--imsize', type=int, default=256)
    parser.add_argument('--version', type=str,choices = ['cfd', 'celeb', 'combined'],default='cfd')

    # Training setting
    parser.add_argument('--total_step', type=int, default=500, help='how many times to update the generator')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_decay', type=float, default=0.95)
    
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)

    # Testing setting
    parser.add_argument('--test_size', type=int, default=2824) 
    parser.add_argument('--model_name', type=str, default='5000_OPTIMAL_HP.pth') 

    parser.add_argument('--name', type=str,default='you-forgot-name', help='experiment name') 



    # using pretrainedALL_
    parser.add_argument('--pretrained_model', type=int, default=None)

    ### USING DEEP LAB V3
    parser.add_argument('--dlv3', action='store_true', help='train using deep lab v3')

    parser.add_argument('--gpu_id', type=int, help='which gpu to use')

    # Misc
    parser.add_argument('--train', action='store_true', help='Enable training mode')
    parser.add_argument('--test', action='store_true', help='Enable testing mode')

    parser.add_argument('--parallel', action='store_true', help='Enable parallel training/ testing')

    parser.add_argument('--hp_tune', action='store_true', help='hp tuning on off')


    # parser.add_argument('--parallel', type=str2bool, default=False)
    # parser.add_argument('--get_sam_iris', action='store_true', help='Use Segment Anything Model to find iris masks')

    # parser.add_argument('--finetune', action='store_true', help='Finetune model using cfd data')

    # Trainign data path (celeb data)
    parser.add_argument('--img_path', type=str, default='./data/fixed_small')
    parser.add_argument('--label_path', type=str, default='./data/fixed_annot_small') 
    
    # Whether or not to train with a subset of the iamges
    # parser.add_argument('--train_limit', type=int, default=None, help='Limit the number of images used for training')

    # Step size 
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=10)
    parser.add_argument('--model_save_step', type=int, default=100)

    
    # Paths for saving
    # parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--model_save_path', type=str, default='./models/')
    parser.add_argument('--csv_path', type=str, default='./dice_scores/')

    # parser.add_argument('--sample_path', type=str, default='./samples')

    #which dataset to test on
    # parser.add_argument('--dataset', type=str, default='celeb', choices=['celeb', 'ted_long', 'ted', 'md', 'cfd'], help='Choose dataset')

    args = parser.parse_args()
    
    if args.version == 'cfd':
        args.img_path = './data/cfd_output_images/train'
        args.label_path = './data/cfd_output_masks/train'

        args.test_image_path = './data/cfd_output_images/test'
        args.test_label_path = './data/cfd_output_masks/test'
        
        # args.test_label_path = './test_results_TED'
        # args.test_color_label_path = './test_color_visualize_TED'
        # args.csv_path = './csvs/2024_ALL_TED_key_SIZED.csv'
        
    elif args.version == 'celeb':
        
        
        args.img_path = './data/celeb_output_images/train'
        args.label_path = './data/celeb_output_masks/train'

        args.test_image_path = './data/celeb_output_images/test'
        args.test_label_path = './data/celeb_output_masks/test'
        

    elif args.version == 'combined':
    
        args.img_path = './data/combined_output_images/train'
        args.label_path = './data/combined_output_masks/train'

        args.test_image_path = './data/combined_output_images/test'
        args.test_label_path = './data/combined_output_masks/test'
        

    if not args.train:
        args.test = True



    return args
