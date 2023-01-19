import pandas as pd
import torch
from transformers import BertModel, BertConfig, DNATokenizer
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import os
import argparse
import sys
from fasta import fasta_iter
from torch.cuda.amp import GradScaler, autocast
from captum.attr import FeatureAblation
import matplotlib.pyplot as plt
import logging

input_length = 4096
kmer_len = 6

def parse_args(args):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='DANBERT')
    parser.add_argument('-i', '--input',
                        required=True,
                        help='Path to the input genome FASTA file.',
                        dest='genome_fasta',
                        default = None)

    parser.add_argument('--model',
                        required=True,
                        help='Path to the model file.',
                        dest='model',
                        default = None)

    parser.add_argument('--batch-size',
                        required=False,
                        help='batch size used in inference.',
                        dest='batch_size',
                        default = 16)

    parser.add_argument('-o', '--output',
                        required=True,
                        help='Output directory (will be created if non-existent)',
                        dest='output',
                        default = None)

    parser.add_argument('--visualize',
                           required=False,
                           help='output the visualization.',
                           dest='visualize',
                           action='store_true',)

    return parser.parse_args()

def generate_reversed(seq):
    reverse_seq = ''
    seq = seq[::-1]
    mapping = {'A':'T', 'T':'A', 'G':'C', 'C':'G','N':'N'}
    for temp in seq:
        reverse_seq += mapping[temp]
    return reverse_seq

def generate_bert_embeddings(seq, tokenizer):
    seq_list = []
    for i in range(8):
        seq_list.append(seq[i * 512 : (i + 1) * 512])

    model_input = []

    for seq1 in seq_list:
        seq1_kmer = ''
        for i in range(len(seq1) - kmer_len + 1):
            if i != len(seq1) - kmer_len:
                seq1_kmer += seq1[i:i + kmer_len] + ' '
            else:
                seq1_kmer += seq1[i:i + kmer_len]

        model_input1 = tokenizer.encode_plus(seq1_kmer, add_special_tokens=True, max_length=512)["input_ids"]
        model_input.extend(model_input1)

    return model_input

class BaseBERTmodel(torch.nn.Module):
    def __init__(self, dir_to_pretrained_model, config):
        super(BaseBERTmodel, self).__init__()
        self.bert = BertModel(config = config)
        self.linear1 = torch.nn.Linear(768 * 8, 128)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(128, 1)

    def forward(self, sequence):
        output1 = self.bert(sequence[:, 0:sequence.shape[1] // 8])[1]
        for i in range(8):
            if i == 0:
                continue
            output = self.bert(sequence[:, i * 509 : (i + 1) * 509])[1]
            output1 = torch.cat((output1, output), dim=1)
        output1 = self.relu(self.linear1(output1))
        output1 = self.linear2(output1)
        return output1

def validate_args(args):
    def expect_file(f):
        if f is not None:
            if not os.path.exists(f):
                sys.stderr.write(f"Error: Expected file '{f}' does not exist\n")
                sys.exit(1)
    expect_file(args.genome_fasta)

    for h, seq in fasta_iter(args.genome_fasta):
        if len(seq) != input_length:
            sys.stderr.write(f"Error: input sequence length != 4096bp\n")
            sys.exit(1)

class feature_Dataset(Dataset):
    def __init__(self, embedding_lists, rever_embedding_lists):
        self.embedding_lists = embedding_lists
        self.rever_embedding_lists = rever_embedding_lists

    def __getitem__(self, item):
        return self.embedding_lists[item], self.rever_embedding_lists[item]

    def __len__(self):
        return len(self.embedding_lists)

def main(args=None):
    args = parse_args(args)
    validate_args(args)

    logger = logging.getLogger('DNABERT_aging')
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s'))
    logger.addHandler(sh)

    dir_to_model = args.model
    output_path = args.output
    os.makedirs(output_path, exist_ok=True)

    config = BertConfig.from_pretrained('config.json')
    tokenizer = DNATokenizer.from_pretrained('dna6')

    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    model = BaseBERTmodel(None, config).to(device)
    checkpoint = torch.load(dir_to_model)
    model.load_state_dict(checkpoint['model_dict'])

    features = []
    reversed_features = []
    seq_list = []
    for h, seq in fasta_iter(args.genome_fasta):
        seq = str(seq).upper()
        features.append(generate_bert_embeddings(seq, tokenizer))
        rev_seq = generate_reversed(seq)
        reversed_features.append(generate_bert_embeddings(rev_seq, tokenizer))
        seq_list.append(h)

    features = np.array(features)
    reversed_features = np.array(reversed_features)

    dataset = feature_Dataset(features, reversed_features)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8)
    logger.info('Running prediction.')
    output_list = []
    with torch.no_grad():
        model.eval()
        for input, input1 in dataloader:
            input = input.long().to(device)
            input1 = input1.long().to(device)

            with autocast(dtype=torch.float16):
                out = model(input)
                out = out.cpu().detach().numpy()

            with autocast(dtype=torch.float16):
                out1 = model(input1)
                out1 = out1.cpu().detach().numpy()

            ave_output = [(temp[0] + temp1[0]) / 2 for temp, temp1 in zip(out, out1)]
            output_list.extend(ave_output)
    logger.info('Prediction finished.')

    results = pd.DataFrame({'sequence':seq_list,'predict':output_list})
    results.to_csv(os.path.join(output_path, 'predictions.csv'))

    dataloader_viz = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8)

    if args.visualize:
        logger.info('Running visualization.')
        with torch.no_grad():
            model.eval()
            for h, (input, input1 ) in zip(seq_list, dataloader_viz):
                input = input.long().to(device)
                input1 = input1.long().to(device)
                lig_f = FeatureAblation(model)
                attributions_score = lig_f.attribute(inputs=input, perturbations_per_eval=10)
                attributions = attributions_score.cpu().numpy().reshape(4072)
                plt.figure(figsize=(12, 4))
                plt.plot(list(range(4072)), attributions, label='attribution score', linewidth=0.5)
                plt.savefig(os.path.join(output_path, f'{h}_forward.jpg'))
                plt.close()
                att_forward = pd.DataFrame({'attribution score': attributions})
                att_forward.to_csv(os.path.join(output_path, f'{h}_forward.csv'))

                lig_r = FeatureAblation(model)
                attributions_score = lig_r.attribute(inputs=input1, perturbations_per_eval=10)
                attributions = attributions_score.cpu().numpy().reshape(4072)
                plt.figure(figsize=(12, 4))
                plt.plot(list(range(4072)), attributions, label='attribution score', linewidth=0.5)
                plt.savefig(os.path.join(output_path, f'{h}_reverse.jpg'))
                plt.close()
                att_reverse = pd.DataFrame({'attribution score': attributions})
                att_reverse.to_csv(os.path.join(output_path, f'{h}_reverse.csv'))

            logger.info('Visualization finished.')

if __name__ == '__main__':
    main(sys.argv)
