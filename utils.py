import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num-epochs', type=int, default=3000, help='number of epochs to train for')
    parser.add_argument('--batch-size', type=int, default=32, help='batch_size')
    parser.add_argument('--dataset', type=str, default='pinwheel', help='which dataset to work with') #earthquake #'covid19' #'citibike', 'pinwheel'
    parser.add_argument('--model-type', type=str, default='benchmark', help='which model to work with') #'benchmark', 'transformer', 'benchmark'
    parser.add_argument('--temporal-model', type=str, default='Hawkesppp', help='which temporal model to work with as benchmark')  #'Homoppp', 'Hawkesppp','Selfppp'
    parser.add_argument('--spatial-model', type=str, default='gmm', help='which spatial model to work with as benchmark') #'gmm', 'cond_gmm'
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--num-layers', type=int, default=6, help='number of layers of the encoder and decoder in the transformer architecture')
    parser.add_argument('--num-heads', type=int, default=6, help='number of heads for the attention model')
    parser.add_argument('--bijector-type', type=str, default='MAR', help='bijector type for the overall NF')
    parser.add_argument('--bij-layers', type=int, default=6, help='number of bijectors being chained')
    parser.add_argument('--NF', type=str, default=None, help='shows the overal density learned by NF')
    parser.add_argument('--NFtrain', type=str, default=True, help='Learns the overal density using NF')
    parser.add_argument('--gpu-num', type=int, default=0, help='GPU number')
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--time_layer_prob', type=str, default='exp')
    parser.add_argument('--loc_layer_prob', type=str, default='gauss')
    parser.add_argument('--remove-all', type=int, default=0, help='Remove the previous experiment')
    parser.add_argument('--seed', type=int, default=0, help='Have reproducible results')
    parser.add_argument('--event_num', type=int, default=500, help='Number of events in each sequence')
    parser.add_argument('--event_out', type=int, default=3, help='Number of events to predict')
    parser.add_argument('--class_num', type=int, default=15, help='Only used for pinwheel dataset')
    parser.add_argument('--event_num_per', type=int, default=150, help='Only used for pinwheel dataset')
    parser.add_argument('--desc', type=str, default='Default', help='add a small descriptor to folder name')
    parser.add_argument('--train', default=True, action='store_true')
    parser.add_argument('--lookaheadmaskin', default=True)
    parser.add_argument('--lookaheadmaskout', default=True)
    parser.add_argument('--seqs', default='fixed') #'variable
    parser.add_argument('--shauffled', default=True)
    parser.add_argument('--ablation-type', default=False, help='what type of ablation study is this?') #'histindept' , 'nodecoder', False
    parser.add_argument('--notrain', dest='train', action='store_false')

    args, unparsed = parser.parse_known_args()
    return args, unparsed
