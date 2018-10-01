import argparse
import itertools
import os.path
import time
from subprocess import Popen, DEVNULL, PIPE

import dynet as dy
import numpy as np

import evaluate
import parse
import trees
import vocabulary

def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string

def get_dependancies(fin, path_penn="src/pennconverter.jar"):
    """ Creates dependancy dictionary for each intput file"""

    command = 'java -jar {} < {} -splitSlash=false'.format(path_penn, fin)
    # proc = Popen(command, shell=True, stdout=PIPE, stderr=DEVNULL)
    proc = Popen(command, shell=True, stdout=PIPE)
    results = proc.stdout.readlines()
    dependancies = []
    dependancy = []
    for res in results:
        res = res.decode('utf8')
        if res == '\n':
            dependancies.append(dependancy)
            dependancy = []
        else:
            dependancy.append(int(res.split()[6]))
    return dependancies

def run_train(args):
    if args.numpy_seed is not None:
        print("Setting numpy random seed to {}...".format(args.numpy_seed))
        np.random.seed(args.numpy_seed)

    print("Loading training trees from {}...".format(args.train_path))
    train_treebank = trees.load_trees(args.train_path)
    print("Loaded {:,} training examples.".format(len(train_treebank)))

    print("Loading development trees from {}...".format(args.dev_path))
    dev_treebank = trees.load_trees(args.dev_path)
    print("Loaded {:,} development examples.".format(len(dev_treebank)))

    print("Processing trees for training...")
    if args.parser_type != 'my':
        train_parse = [tree.convert() for tree in train_treebank]
    else:
        dependancies = get_dependancies(args.train_path)
        train_parse = [tree.myconvert(dep)(args.keep_valence_value)
                            for tree, dep in zip(train_treebank, dependancies)]
        print("Processing trees for development...")
        dependancies = get_dependancies(args.dev_path)
        dev_parse = [tree.myconvert(dep)(args.keep_valence_value)
                            for tree, dep in zip(dev_treebank, dependancies)]

    print("Constructing vocabularies...")

    tag_vocab = vocabulary.Vocabulary()
    tag_vocab.index(parse.START)
    tag_vocab.index(parse.STOP)

    word_vocab = vocabulary.Vocabulary()
    word_vocab.index(parse.START)
    word_vocab.index(parse.STOP)
    word_vocab.index(parse.UNK)

    if args.parser_type == 'my':
        char_vocab = vocabulary.Vocabulary()
        char_vocab.index(parse.START)
        char_vocab.index(parse.STOP)
        for c in parse.START+parse.STOP+parse.UNK:
            char_vocab.index(c)

    label_vocab = vocabulary.Vocabulary()
    if args.parser_type != 'my':
        label_vocab.index(())
    else:
        label_vocab.index(parse.START)
        label_vocab.index(parse.STOP)

    for tree in train_parse:
        nodes = [tree]
        while nodes:
            node = nodes.pop()
            if args.parser_type != 'my':
                if isinstance(node, trees.InternalParseNode):
                    label_vocab.index(node.label)
                    nodes.extend(reversed(node.children))
                else:
                    tag_vocab.index(node.tag)
                    word_vocab.index(node.word)
            else:
                if isinstance(node, trees.InternalMyParseNode):
                    nodes.extend(reversed(node.children))
                else:
                    for l in node.labels:
                        label_vocab.index(l)
                    for c in node.word:
                        char_vocab.index(c)
                    tag_vocab.index(node.tag)
                    word_vocab.index(node.word)


    tag_vocab.freeze()
    word_vocab.freeze()
    label_vocab.freeze()
    if args.parser_type == 'my':
        char_vocab.freeze()

    def print_vocabulary(name, vocab):
        special = {parse.START, parse.STOP, parse.UNK}
        print("{} ({:,}): {}".format(
            name, vocab.size,
            sorted(value for value in vocab.values if value in special) +
            sorted(value for value in vocab.values if value not in special)))

    if args.print_vocabs:
        print_vocabulary("Tag", tag_vocab)
        print_vocabulary("Word", word_vocab)
        print_vocabulary("Label", label_vocab)

    print("Initializing model...")
    model = dy.ParameterCollection()
    if args.parser_type == "my":
        parser = parse.MyParser(
            model,
            tag_vocab,
            word_vocab,
            char_vocab,
            label_vocab,
            args.tag_embedding_dim,
            args.word_embedding_dim,
            args.char_embedding_dim,
            args.label_embedding_dim,
            args.lstm_layers,
            args.lstm_dim,
            args.char_lstm_dim,
            args.dec_lstm_dim,
            args.attention_dim,
            args.label_hidden_dim,
            args.keep_valence_value,
            args.dropouts
        )
    elif args.parser_type == "top-down":
        parser = parse.TopDownParser(
            model,
            tag_vocab,
            word_vocab,
            label_vocab,
            args.tag_embedding_dim,
            args.word_embedding_dim,
            args.lstm_layers,
            args.lstm_dim,
            args.label_hidden_dim,
            args.split_hidden_dim,
            args.dropout,
        )
    else:
        parser = parse.ChartParser(
            model,
            tag_vocab,
            word_vocab,
            label_vocab,
            args.tag_embedding_dim,
            args.word_embedding_dim,
            args.lstm_layers,
            args.lstm_dim,
            args.label_hidden_dim,
            args.dropout,
        )
    trainer = dy.AdamTrainer(model)

    total_processed = 0
    current_processed = 0
    check_every = len(train_parse) / args.checks_per_epoch
    best_dev_fscore = -np.inf
    best_dev_model_path = None
    best_dev_loss = np.inf

    start_time = time.time()

    def my_check_dev():
        nonlocal best_dev_loss
        nonlocal best_dev_model_path

        dev_start_time = time.time()

        total_losses = []
        for start_index in range(0, len(dev_parse), args.batch_size):
            dy.renew_cg()
            batch_losses = []
            for tree in dev_parse[start_index:start_index + args.batch_size]:
                sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves()]
                _, losses = parser.parse(sentence, tree, True)
                batch_losses.extend(losses)
            batch_loss = dy.average(batch_losses)
            total_losses.append(batch_loss.scalar_value())

            print(
                "batch {:,}/{:,} "
                "batch-loss {:.4f} "
                "dev-elapsed {} "
                "total-elapsed {}".format(
                    start_index // args.batch_size + 1,
                    int(np.ceil(len(dev_parse) / args.batch_size)),
                    total_losses[-1],
                    format_elapsed(dev_start_time),
                    format_elapsed(start_time),
                )
            )

        dev_loss = np.mean(total_losses)
        print(
            "dev-loss {} "
            "dev-elapsed {} "
            "total-elapsed {}".format(
                dev_loss,
                format_elapsed(dev_start_time),
                format_elapsed(start_time),
            )
        )

        if dev_loss < best_dev_loss:
            if best_dev_model_path is not None:
                for ext in [".data", ".meta"]:
                    path = best_dev_model_path + ext
                    if os.path.exists(path):
                        print("Removing previous model file {}...".format(path))
                        os.remove(path)

            best_dev_loss = dev_loss
            best_dev_model_path = "{}_dev={:.4f}".format(
                args.model_path_base, dev_loss)
            print("Saving new best model to {}...".format(best_dev_model_path))
            dy.save(best_dev_model_path, [parser])


    def check_dev():
        nonlocal best_dev_fscore
        nonlocal best_dev_model_path

        dev_start_time = time.time()

        dev_predicted = []
        for tree in dev_treebank:
            dy.renew_cg()
            sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves()]
            predicted, _ = parser.parse(sentence)
            dev_predicted.append(predicted.convert())

        dev_fscore = evaluate.evalb(args.evalb_dir, dev_treebank, dev_predicted)

        print(
            "dev-fscore {} "
            "dev-elapsed {} "
            "total-elapsed {}".format(
                dev_fscore,
                format_elapsed(dev_start_time),
                format_elapsed(start_time),
            )
        )

        if dev_fscore.fscore > best_dev_fscore:
            if best_dev_model_path is not None:
                for ext in [".data", ".meta"]:
                    path = best_dev_model_path + ext
                    if os.path.exists(path):
                        print("Removing previous model file {}...".format(path))
                        os.remove(path)

            best_dev_fscore = dev_fscore.fscore
            best_dev_model_path = "{}_dev={:.2f}".format(
                args.model_path_base, dev_fscore.fscore)
            print("Saving new best model to {}...".format(best_dev_model_path))
            dy.save(best_dev_model_path, [parser])

    for epoch in itertools.count(start=1):
        if args.epochs is not None and epoch > args.epochs:
            break

        np.random.shuffle(train_parse)
        epoch_start_time = time.time()

        for start_index in range(0, len(train_parse), args.batch_size):
            dy.renew_cg()
            batch_losses = []
            for tree in train_parse[start_index:start_index + args.batch_size]:
                sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves()]
                if args.parser_type == "my":
                    _, losses = parser.parse(sentence, tree)
                    batch_losses.extend(losses)
                elif args.parser_type == "top-down":
                    _, loss = parser.parse(sentence, tree, args.explore)
                    batch_losses.append(loss)
                else:
                    _, loss = parser.parse(sentence, tree)
                    batch_losses.append(loss)
                total_processed += 1
                current_processed += 1

            batch_loss = dy.average(batch_losses)
            batch_loss_value = batch_loss.scalar_value()
            batch_loss.backward()
            trainer.update()

            print(
                "epoch {:,} "
                "batch {:,}/{:,} "
                "processed {:,} "
                "batch-loss {:.4f} "
                "epoch-elapsed {} "
                "total-elapsed {}".format(
                    epoch,
                    start_index // args.batch_size + 1,
                    int(np.ceil(len(train_parse) / args.batch_size)),
                    total_processed,
                    batch_loss_value,
                    format_elapsed(epoch_start_time),
                    format_elapsed(start_time),
                )
            )

            if current_processed >= check_every:
                current_processed -= check_every
                if args.parser_type == "my":
                    my_check_dev()
                else:
                    check_dev()

def run_test(args):
    print("Loading test trees from {}...".format(args.test_path))
    test_treebank = trees.load_trees(args.test_path)
    print("Loaded {:,} test examples.".format(len(test_treebank)))

    print("Loading model from {}...".format(args.model_path_base))
    model = dy.ParameterCollection()
    [parser] = dy.load(args.model_path_base, model)

    print("Parsing test sentences...")

    start_time = time.time()

    test_predicted = []
    if args.parser_type == "my":
        predict_parms = {'astar_parms': args.astar_parms, 'beam_parms':args.beam_size}
    for i, tree in  enumerate(test_treebank):
        dy.renew_cg()
        sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves()]
        if args.parser_type == "my":
            prediction_start_time = time.time()
            predicted, _ = parser.parse(sentence, predict_parms=predict_parms)
            print(
                "processed {:,}/{:,} "
                "prediction-elapsed {} "
                "total-elapsed {}".format(
                    i+1,
                    len(test_treebank),
                    format_elapsed(prediction_start_time),
                    format_elapsed(start_time),
                )
            )
        else:
            predicted, _ = parser.parse(sentence)
        test_predicted.append(predicted.convert())

    test_fscore = evaluate.evalb(args.evalb_dir, test_treebank, test_predicted)

    print(
        "test-fscore {} "
        "test-elapsed {}".format(
            test_fscore,
            format_elapsed(start_time),
        )
    )

def main():
    dynet_args = [
        "--dynet-mem",
        "--dynet-weight-decay",
        "--dynet-autobatch",
        "--dynet-gpus",
        "--dynet-gpu",
        "--dynet-devices",
        "--dynet-seed",
    ]

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    subparser = subparsers.add_parser("train")
    subparser.set_defaults(callback=run_train)
    for arg in dynet_args:
        subparser.add_argument(arg)
    subparser.add_argument("--numpy-seed", type=int)
    subparser.add_argument("--parser-type", choices=["top-down", "chart", "my"], required=True)
    subparser.add_argument("--tag-embedding-dim", type=int, default=50)
    subparser.add_argument("--word-embedding-dim", type=int, default=100)
    subparser.add_argument("--char-embedding-dim", type=int, default=50)
    subparser.add_argument("--label-embedding-dim", type=int, default=100)
    subparser.add_argument("--lstm-layers", type=int, default=2)
    subparser.add_argument("--lstm-dim", type=int, default=250)
    subparser.add_argument("--char-lstm-dim", type=int, default=100)
    subparser.add_argument("--dec-lstm-dim", type=int, default=600)
    subparser.add_argument("--attention-dim", type=int, default=250)
    subparser.add_argument("--label-hidden-dim", type=int, default=250)
    subparser.add_argument("--split-hidden-dim", type=int, default=250)
    subparser.add_argument("--dropout", type=float, default=0.4)
    subparser.add_argument("--dropouts", nargs='+', type=float, default=[0.4, 0.2])
    subparser.add_argument("--explore", action="store_true")
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--train-path", default="data/02-21.10way.clean")
    subparser.add_argument("--dev-path", default="data/22.auto.clean")
    subparser.add_argument("--batch-size", type=int, default=10)
    subparser.add_argument("--epochs", type=int)
    subparser.add_argument("--checks-per-epoch", type=int, default=4)
    subparser.add_argument("--print-vocabs", action="store_true")
    subparser.add_argument("--keep-valence-value", action="store_true")

    subparser = subparsers.add_parser("test")
    subparser.set_defaults(callback=run_test)
    for arg in dynet_args:
        subparser.add_argument(arg)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--test-path", default="data/23.auto.clean")
    subparser.add_argument("--parser-type", choices=["top-down", "chart", "my"], required=True)
    subparser.add_argument("--astar-parms", nargs=4, default=[1, 60., 2, 0.2], type=float)
    subparser.add_argument("--beam-size", nargs='+', default=[5], type=int)


    args = parser.parse_args()
    args.callback(args)

if __name__ == "__main__":
    main()
