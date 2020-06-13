import glob, json, os, argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--segment_len', type=int, default=254,
                        help='the length of each example')
    # we set this to be 254 instead of 256 because we want the input to be like: <control_code> input_ids <eos>
    parser.add_argument('--stride', type=int, default=10,
                        help='stride to split training examples')
    parser.add_argument('--dev_size', type=float, default=0.1,
                        help='split ratio of development set for each language')
    args = parser.parse_args()

    gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2", do_lower_case=False)
    paths = ['Python', 'Java']
    segments = {}

    for path in paths:
        source_files = glob.glob(f'{path}/**/*.py' if path == "Python" else f'{path}/**/*.java', recursive=True)
        for each_src in tqdm(source_files):
            with open(each_src, "r", encoding="utf-8") as f:
                code_content = f.read()
                encoded = gpt2_tok.encode(code_content)
                for i in range(len(encoded) // args.stride):
                    seg = encoded[i * args.stride:i * args.stride + args.segment_len]
                    if path not in segments:
                        segments[path] = []
                    segments[path].append(json.dumps({"token_ids": seg, "label": path}))

    train, dev = [], []
    for key in segments:
        # we don't shuffle before splitting because we want the train and dev to be very different (less overlapping)
        tr, de = train_test_split(segments[key], test_size=args.dev_size)
        train += tr
        dev += de

    to_path = "source_code/json"
    if not os.path.isdir(to_path):
        os.makedirs(to_path)

    with open(os.path.join(to_path, "train.jsonl"), "w") as f:
        f.write("\n".join(train))

    with open(os.path.join(to_path, "dev.jsonl"), "w") as f:
        f.write("\n".join(dev))
