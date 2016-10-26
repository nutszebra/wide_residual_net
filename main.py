import sys
sys.path.append('./trainer')
import argparse
import nutszebra_cifar10
import wide_residual_net
import nutszebra_data_augmentation
import nutszebra_optimizer

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='cifar10')
    parser.add_argument('--load_model', '-m',
                        default=None,
                        help='trained model')
    parser.add_argument('--load_optimizer', '-o',
                        default=None,
                        help='optimizer for trained model')
    parser.add_argument('--load_log', '-l',
                        default=None,
                        help='optimizer for trained model')
    parser.add_argument('--num_of_category', '-noc', type=int,
                        default=2,
                        help='number of categories')
    parser.add_argument('--save_path', '-p',
                        default='./',
                        help='model and optimizer will be saved every epoch')
    parser.add_argument('--epoch', '-e', type=int,
                        default=200,
                        help='maximum epoch')
    parser.add_argument('--batch', '-b', type=int,
                        default=128,
                        help='mini batch number')
    parser.add_argument('--gpu', '-g', type=int,
                        default=-1,
                        help='-1 means cpu mode, put gpu id here')
    parser.add_argument('--start_epoch', '-s', type=int,
                        default=1,
                        help='start from this epoch')
    parser.add_argument('--train_batch_divide', '-trb', type=int,
                        default=4,
                        help='divid batch number by this')
    parser.add_argument('--test_batch_divide', '-teb', type=int,
                        default=4,
                        help='divid batch number by this')
    parser.add_argument('--lr', '-lr', type=float,
                        default=0.1,
                        help='leraning rate')
    parser.add_argument('--k', '-k', type=int,
                        default=8,
                        help='width hyperparameter')
    parser.add_argument('--N', '-n', type=int,
                        default=16,
                        help='width hyperparameter')

    args = parser.parse_args().__dict__
    lr = args.pop('lr')
    k = args.pop('k')
    N = args.pop('N')
    num_of_category = args.pop('num_of_category')

    print('generating model')
    model = wide_residual_net.WideResidualNetwork(num_of_category, block_num=3, out_channels=(16 * k, 32 * k, 64 * k), N=(int(N / 3.), int(N / 3.), int(N / 3.)))
    print('Done')
    optimizer = nutszebra_optimizer.OptimizerWideRes(model, lr=lr)
    args['model'] = model
    args['optimizer'] = optimizer
    args['da'] = nutszebra_data_augmentation.DataAugmentationCifar10NormalizeSmall
    main = nutszebra_cifar10.TrainCifar10(**args)
    main.run()
