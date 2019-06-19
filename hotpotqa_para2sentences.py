from typing import List
from collections import defaultdict
import json
import argparse
import logging

# from joblib import Parallel, delayed
import spacy
import neuralcoref
from spacy.tokens import Doc

from fftqdm import make_tqdm


# from tqdm import tqdm
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
tqdm = make_tqdm(logger)

using_gpu = spacy.prefer_gpu()
print(f"SpaCy using GPU: {using_gpu}")

def custom_sentencizer(doc: Doc):
    for i, token in enumerate(doc[:-2]):
        # Define sentence start if 2 blank spaces found
        if token.text == " ":
            doc[i+1].is_sent_start = True
        else:
            # Explicitly set sentence start to False otherwise, to tell
            # the parser to leave those tokens alone
            doc[i+1].is_sent_start = False
    return doc

def coref_replace(doc: Doc, exclude_text: str = "") -> List[str]:
    # Make sure following two lines have been pre-run
    # paragraph = "  ".join([" ".join(sentence.split()) for sentence in paragraph_sentences])
    # doc = coref_nlp(paragraph)
    if not doc._.coref_clusters:
        sentences = [sent.text for sent in  doc.sents]
        # assert len(sentences) == len(paragraph_sentences)
        return sentences
    # Get sentence boundaris first by pointer and cumsum
    pointer = 0
    sentence_markers = []
    for sentence in doc.sents:
        sentence_len = len(sentence)
        sentence_markers.append((pointer, pointer+sentence_len-1))
        pointer += (sentence_len)

    updated_sentences = ["" for index in range(len(sentence_markers))]
    sent_idx_to_replacements = defaultdict(list)
    for token in doc:
        if not token.text.strip():
            continue
        token_sent_idx = [(marker[0] <= token.i <= marker[1])
                          for marker in sentence_markers].index(True)
        associated_nouns = []
        clusters = token._.coref_clusters
        for cluster in clusters:
            for mention in cluster.mentions:
                mention_sent_idx = [(marker[0] <= mention[0].i <= marker[1])
                                    for marker in sentence_markers].index(True)
                # Don't make coref replacement from the same sentence.
                if token_sent_idx == mention_sent_idx:
                    continue

                if len(mention.text) > 40:
                    continue

                # Essentially, exclude text corresponds to answer_text. We don't want
                # to remove it since it can break answer span identification.
                if mention.text.strip() in exclude_text:
                    continue
                if exclude_text.strip() and exclude_text.strip() in mention.text.strip():
                    continue

                # Constrained (Only allow Nouns and Pronouns)
                if token.pos_ == "PRON":
                    # same replacement shouldn't have been already been made in this sentence
                    already_replaced = sent_idx_to_replacements[token_sent_idx]
                    if any(replaced_mention.text == mention.text
                           for replaced_mention in already_replaced):
                        continue
                    if any(subtoken.pos_ == "NOUN" or subtoken.pos_ == "PROPN"
                           for subtoken in mention):
                        associated_nouns.append(mention)
                        sent_idx_to_replacements[token_sent_idx].append(mention)

                # # Unconstrained (Allow everything)
                # # same replacement shouldn't have been already been made in this sentence
                # already_replaced = sent_idx_to_replacements[token_sent_idx]
                # if any(replaced_mention.text == mention.text
                #        for replaced_mention in already_replaced):
                #     continue
                # associated_nouns.append(mention)
                # sent_idx_to_replacements[token_sent_idx].append(mention)

        replacement_span = associated_nouns[0] if associated_nouns else token
        updated_sentences[token_sent_idx] += replacement_span.text_with_ws
    # assert len(updated_sentences) == len(paragraph_sentences)
    return updated_sentences

coref_nlp = spacy.load("en_core_web_md")
coref_nlp.add_pipe(custom_sentencizer, before="parser")  # Insert before the parser.
neuralcoref.add_to_pipe(coref_nlp) # Insert at the last.


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
            instances, paragraphs = [], []

            logger.info("Reading file...")
            for line in tqdm(read_file.readlines()):
                if not line.strip():
                    continue
                instance = json.loads(line.strip())
                paragraph = instance.pop("paragraph")
                paragraph_sentences = [sent.strip() for sent in paragraph.split("[SBD]")]
                paragraph = "  ".join([" ".join(sentence.split()) for sentence in paragraph_sentences])

                instances.append(instance)
                paragraphs.append(paragraph)

            logger.info("Applying coref replacements...")
            for paragraph, instance in tqdm(zip(paragraphs, instances)):
                doc = coref_nlp(paragraph)
                answer = instance["answer"]
                corefed_paragraph_sentences = coref_replace(doc, exclude_text=answer)
                instance["sentences"] = corefed_paragraph_sentences
                write_file.write(json.dumps(instance) + "\n")

    # If it's hotpotqa-processed file
    else:
        logger.info(f"Set {args.data_type} type.")
        logger.info("Processing file...")
        with open(args.paragraphs_file, 'r') as read_file, open(args.sentences_file, 'w') as write_file:
            instances, paragraphs = [], []
            for line in tqdm(read_file.readlines()):
                if not line.strip():
                    continue
                instance = json.loads(line.strip())
                paragraphs = instance["context"]

                for paragraph in paragraphs:
                    paragraph_sentences = paragraph["text"]
                    # NOTE: 2 blanks and 1 blank are intentional. Do not remove them.
                    paragraph_text = "  ".join([" ".join(sentence.split()) for sentence in paragraph_sentences])
                    doc = coref_nlp(paragraph_text)
                    corefed_paragraph_sentences = coref_replace(doc)
                    if [e.strip() for e in corefed_paragraph_sentences] != paragraph_sentences:
                        import pdb; pdb.set_trace()
                    assert len(corefed_paragraph_sentences) == len(corefed_paragraph_sentences)
                    paragraph["text"] = corefed_paragraph_sentences
                write_file.write(json.dumps(instance) + "\n")
