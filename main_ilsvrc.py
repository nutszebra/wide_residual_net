import sys
sys.path.append('./trainer')
import nutszebra_ilsvrc_object_localization
import nutszebra_optimizer
import wide_residual_net
import argparse
import trainer.nutszebra_data_augmentation as da

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
    parser.add_argument('--load_data', '-ld',
                        default=None,
                        help='ilsvrc path')
    parser.add_argument('--save_path', '-p',
                        default='./',
                        help='model and optimizer will be saved at every epoch')
    parser.add_argument('--epoch', '-e', type=int,
                        default=100,
                        help='maximum epoch')
    parser.add_argument('--batch', '-b', type=int,
                        default=64,
                        help='mini batch number')
    parser.add_argument('--gpu', '-g', type=int,
                        default=-1,
                        help='-1 means cpu, put gpu id here')
    parser.add_argument('--start_epoch', '-s', type=int,
                        default=1,
                        help='start from this epoch')
    parser.add_argument('--train_batch_divide', '-trb', type=int,
                        default=1,
                        help='divid train batch number by this')
    parser.add_argument('--test_batch_divide', '-teb', type=int,
                        default=1,
                        help='divid test batch number by this')
    parser.add_argument('--lr', '-lr', type=float,
                        default=0.045,
                        help='leraning rate')
    parser.add_argument('--k', '-k', type=int,
                        default=10,
                        help='width hyperparameter')
    parser.add_argument('--N', '-n', type=int,
                        default=28,
                        help='total layers')

    args = parser.parse_args().__dict__
    print(args)
    lr = args.pop('lr')
    k = args.pop('k')
    N = args.pop('N')

    print('generating model')
    model = wide_residual_net.WideResidualNetwork(10, block_num=3, out_channels=(16 * k, 32 * k, 64 * k), N=(int(N / 3. / 2.), int(N / 3. / 2.), int(N / 3. / 2.)))
    print('Done')
    optimizer = nutszebra_optimizer.OptimizerGooglenetV3(model, lr=lr)
    args['model'] = model
    args['optimizer'] = optimizer
    args['da'] = da.DataAugmentationNormalizeBigger
    main = nutszebra_ilsvrc_object_localization.TrainIlsvrcObjectLocalizationClassification(**args)
    main.run()
