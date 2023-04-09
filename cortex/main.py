'''Main file for running experiments.

'''


import logging

from cortex._lib import (config, data, exp, optimizer, setup_cortex,
                         setup_experiment, train)
from cortex._lib.utils import print_section

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'


logger = logging.getLogger('cortex')

#定义 run 函数，接受一个名为 model 的参数，默认值为 None
def run(model=None):
    '''Main function.

    '''
    # Parse the command-line arguments

    try:
        #调用 setup_cortex 函数，并传递 model 参数。该函数负责解析命令行参数、设置环境变量和实例化模型。返回的 args 变量包含解析后的命令行参数。
        args = setup_cortex(model=model)
        if args.command == 'setup':
            # Performs setup only.
            config.setup()
            exit(0)
        else:
            config.set_config()
            print_section('EXPERIMENT')
            model, reload_nets = setup_experiment(args, model=model)
            print_section('DATA')
            data.setup(**exp.ARGS['data'])
            print_section('MODEL')
            model.reload_nets(reload_nets)
            model.build()
            print_section('OPTIMIZER')
            optimizer.setup(model, **exp.ARGS['optimizer'])

    except KeyboardInterrupt:
        print('Cancelled')
        exit(0)

    print_section('RUNNING')
    train.main_loop(model, **exp.ARGS['train'])
