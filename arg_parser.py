import argparse


def args_parser():
    parser = argparse.ArgumentParser()


    parser.add_argument('--num_epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    # model arguments

    parser.add_argument('--model', type=str, default='H-optimus-0',
                        help='name of the model')
    parser.add_argument('--dataset', type=str, default='ovarian')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--num_channels', type=int, default=3, help="number \
                        of channels of imgs")
    parser.add_argument('--wsi_dim', type=int, default=1024, help='dimension of your input WSI')


    # other arguments
    parser.add_argument('--saved_features', default='False', help="Train the FM or add extra layer")
    parser.add_argument('--num_classes', type=int, default=5, help="number \
                        of classes")
    parser.add_argument('--gpu', default=True, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='adam', help="type \
                        of optimizer")
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--version', type=int, default=1, help='version of the experiment')
    parser.add_argument('--save_dir', type=str, default='./results', help='path to save results')
    parser.add_argument('--add_layer', type=str, default='True', help='add extra layer to the model for fine-tuning')
    args = parser.parse_args()
    return args
