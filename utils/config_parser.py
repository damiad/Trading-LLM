import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Time-LLM')

    # basic config
    parser.add_argument('--model_id', type=str, required=True,
                        default='test', help='model id')
    parser.add_argument('--model_comment', type=str, required=True,
                        default='none', help='prefix when saving test results')
    parser.add_argument('--model', type=str, required=True, default='TradingLLM',
                        help='model name, options: [TradingLLM]')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')

    # data loader
    parser.add_argument('--data', type=str, required=True,
                        default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str,
                        default='./dataset', help='root path of the data file')
    parser.add_argument('--data_path', type=str,
                        default='ETTh1.csv', help='data file')
    parser.add_argument('--target', type=str, default='OT',
                        help='target feature in S or MS task')
    parser.add_argument('--loader', type=str,
                        default='modal', help='dataset type')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, '
                        'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                        'you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str,
                        default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96,
                        help='input sequence length')
    parser.add_argument('--label_len', type=int,
                        default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96,
                        help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str,
                        default='Monthly', help='subset for M4')

    # model define
    parser.add_argument('--model_checkpoint_path', type=str,
                        required=False)  # pass when loading checkpoint
    parser.add_argument('--d_model', type=int, default=16,
                        help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2,
                        help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1,
                        help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=32,
                        help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25,
                        help='window size of moving average')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str,
                        default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true',
                        help='whether to output attention in encoder')
    parser.add_argument('--patch_len', type=int,
                        default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--prompt_domain', type=int, default=0, help='stride')

    # optimization
    parser.add_argument('--num_workers', type=int,
                        default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int,
                        default=10, help='train epochs')
    parser.add_argument('--align_epochs', type=int,
                        default=10, help='alignment epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size of train input data')
    parser.add_argument('--eval_batch_size', type=int,
                        default=8, help='batch size of model evaluation')
    parser.add_argument('--patience', type=int, default=10,
                        help='early stopping patience')
    parser.add_argument('--learning_rate', type=float,
                        default=0.0001, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='MSE',
                        help='loss function')
    parser.add_argument('--lradj', type=str, default='type1',
                        help='adjust learning rate')
    parser.add_argument('--pct_start', type=float,
                        default=0.2, help='pct_start')
    parser.add_argument('--llm_layers', type=int, default=6)

    #custom 
    parser.add_argument('--cg_value', type=int, default=1)
    parser.add_argument('--seq_step', type=int, default=1)

    return parser.parse_args()
