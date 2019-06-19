from typing import List
from collections import defaultdict
import json
import argparse
import logging
from itertools import zip_longest

# from joblib import Parallel, delayed
import spacy
import neuralcoref
from spacy.tokens import Doc

from fftqdm import make_tqdm
from coref_replace_dirk import batch_coref_replace

# from tqdm import tqdm
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
tqdm = make_tqdm(logger)

using_gpu = spacy.prefer_gpu()
print(f"SpaCy using GPU: {using_gpu}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Splits hotpotqa paragraphs to sentences and applies coref replacement.')
    parser.add_argument('paragraphs_file', type=str, help='path to hotpotqa input paragraphs file.')
    parser.add_argument('sentences_file', type=str, help='path to hotpotqa output sentences file.')
    parser.add_argument('--data-type', type=str, choices=("wiki", "hotpotqa"), default="hotpotqa")

    args = parser.parse_args()

    # If it's hotpotqa-wikipedia file
    if args.data_type == "wiki":
        logger.info(f"Set {args.data_type} type.")
        with open(args.paragraphs_file, 'r') as read_file, open(args.sentences_file, 'w') as write_file:
            instances, paragraphs_sentences = [], []

            logger.info("Reading file...")
            for line in tqdm(read_file.readlines()):
                if not line.strip():
                    continue
                instance = json.loads(line.strip())
                paragraph = instance.pop("paragraph")
                paragraph_sentences = [sent.strip() for sent in paragraph.split("[SBD]")]
                instances.append(instance)
                paragraphs_sentences.append(paragraph_sentences)

            logger.info("Processing file...")
            def _chunks(items, size):
                """Yield successive n-sized chunks from items."""
                for i in range(0, len(items), size):
                    yield items[i:i + size]

            batch_size = 1000
            for batch_paragraphs_sentences, batch_instances in tqdm(zip(_chunks(paragraphs_sentences, batch_size),
                                                                        _chunks(instances, batch_size))):
                batch_corefed_paragraphs_sentences = batch_coref_replace(batch_paragraphs_sentences)
                for corefed_paragraph_sentences, instance in zip(batch_corefed_paragraphs_sentences, batch_instances):
                    instance["paragraph"] = " [SBD] ".join(corefed_paragraph_sentences) # updated later. It should be inplace.
                    write_file.write(json.dumps(instance) + "\n")

    # If it's hotpotqa-processed file
    else:
        logger.info(f"Set {args.data_type} type.")
        with open(args.paragraphs_file, 'r') as read_file, open(args.sentences_file, 'w') as write_file:
            instances, paragraphs = [], []
            for line in tqdm(read_file.readlines()):
                if not line.strip():
                    continue
                instance = json.loads(line.strip())
                answer = instance["answer"]
                paragraphs = instance["context"]
                paragraphs_sentences = [paragraph["text"] for paragraph in paragraphs]
                corefed_paragraphs_sentences = batch_coref_replace(paragraphs_sentences,
                                                                   exclude_texts=[answer]*len(paragraphs_sentences))
                for paragraph, corefed_paragraph_sentences in zip(paragraphs, corefed_paragraphs_sentences):
                    if answer in "".join(paragraph["text"]) and answer not in "".join(corefed_paragraph_sentences):
                        logger.warning("Found an instance for which answer text was replaced.")
                    paragraph["text"] = corefed_paragraph_sentences
                write_file.write(json.dumps(instance) + "\n")
