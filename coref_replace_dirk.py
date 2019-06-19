from typing import List, Iterable
from collections import deque
import spacy
import neuralcoref
import re
import string
from collections import defaultdict

coref_nlp = spacy.load("en_core_web_md")
neuralcoref.add_to_pipe(coref_nlp)

_parenthetical_re = re.compile(r'\([^()]*\)')
def _remove_parentheticals(s: str) -> str:
    return _parenthetical_re.sub("", s) # does not work for nested parentheticals

def _window(seq: Iterable, n:int = 2) -> Iterable[List]:
    win = deque(maxlen=n)
    for e in seq:
        win.append(e)
        if len(win) == n:
            yield list(win)

def _replace_range(s: str, start: int, end: int, replacement: str):
    return s[:start] + replacement + s[end:]

def batch_coref_replace(paragraphs_sentences: List[List[str]], exclude_texts: List[str] = None) -> List[List[str]]:
    docs = coref_nlp.pipe("".join(sentences) for sentences in paragraphs_sentences)
    corefed_paragraphs_sentences = []

    exclude_texts = exclude_texts or [""]*len(paragraphs_sentences)
    for paragraph_sentences, doc, exclude_text in zip(paragraphs_sentences, docs, exclude_texts):
        if doc._.has_coref:
            replacements = []
            for cluster in doc._.coref_clusters:
                replacement = str(cluster.main).strip()
                if len(replacement) > 40:
                    continue
                if replacement.lower() in {"he", "him", "his", "she", "her", "it", "its", "they", "their"}:
                    continue

                # We don't want to make a replacement for the same cluster in the same
                # sentence more than once. Otherwise we end up with sentences like
                # 'Alice wrote Alice's first book when Alice was 10 years old.'.
                sentence2replacements = defaultdict(list)

                for mention in cluster.mentions:
                    if mention == cluster.main:
                        continue
                    if str(mention) == replacement:
                        continue
                    if mention.sent == cluster.main.sent:
                    # Since this information is present in same sentence, leave it on BERT.
                        continue

                    replacement = _remove_parentheticals(replacement).strip()
                    replacement = replacement.strip(string.punctuation)
                    if len(replacement) <= 0:
                        continue
                    if str(mention) == replacement:
                        continue

                    # Following check should happen before doing applying possessive rules.
                    if replacement.strip() in sentence2replacements[mention.sent]:
                    # To avoid the Alice case.
                        continue
                    sentence2replacements[mention.sent].append(replacement.strip())

                    if str(mention) in {"his", "her", "its", "their"}:
                        if replacement.endswith("'") or replacement.endswith("'s"):
                            pass
                        elif replacement.endswith("s"):
                            replacement += "'"
                        else:
                            replacement += "'s"
                    if str(mention).endswith("'s") and not replacement.endswith("'s"):
                        replacement += "'s"


                    # Essentially, exclude text corresponding to answer_text. We don't want
                    # to remove it since it can break answer span identification.
                    if mention.text.strip() in exclude_text:
                        continue
                    if exclude_text.strip() and exclude_text.strip() in mention.text.strip():
                        continue

                    replacements.append((mention.start_char, mention.end_char, replacement))

            replacements.sort(reverse=True)

            # delete replacements that overlap
            overlapping_replacement_indices = set()
            for (i1, (start1, end1, _)), (i2, (start2, end2, _)) in _window(enumerate(replacements), 2):
                if max(end1, end2) - min(start1, start2) < (end1 - start1) + (end2 - start2):
                    overlapping_replacement_indices.add(i1)
                    overlapping_replacement_indices.add(i2)
            overlapping_replacement_indices = list(overlapping_replacement_indices)
            overlapping_replacement_indices.sort(reverse=True)
            for i in overlapping_replacement_indices:
                del replacements[i]

            # apply the replacements
            # This is way more complicated than it should be, because spacy and hotpot have a different
            # way of breaking text into sentences. Hotpot refers to sentences by index, so we have to
            # preserve the hotpot sentence breaks.
            while len(replacements) > 0:
                start_char, end_char, replacement = replacements.pop()
                assert start_char < end_char
                sentence_index = 0
                sentence_start_char = start_char
                sentence_end_char = end_char
                while sentence_end_char > len(paragraph_sentences[sentence_index]):
                    skipped_sentence_length = len(paragraph_sentences[sentence_index])
                    sentence_start_char -= skipped_sentence_length
                    sentence_end_char -= skipped_sentence_length
                    sentence_index += 1
                if sentence_start_char < 0:
                    # _logger.warning("Coref replacement spanning sentence boundary. Ignoring.")
                    continue

                # apply the replacement
                paragraph_sentences[sentence_index] = _replace_range(
                    paragraph_sentences[sentence_index],
                    sentence_start_char,
                    sentence_end_char,
                    replacement)

                # adjust all the other replacements accordingly
                adjustment = len(replacement) - (end_char - start_char)
                new_replacements = []
                for replacement in replacements:
                    rep_start_char, rep_end_char, rep_string = replacement
                    if rep_start_char > start_char:
                        new_replacements.append((
                            rep_start_char + adjustment,
                            rep_end_char + adjustment,
                            rep_string))
                    else:
                        new_replacements.append(replacement)
                replacements = new_replacements

            # Sometimes our logic for possessives duplicates the 's suffix.
            corefed_paragraphs_sentences.append([s.replace("'s's", "'s") for s in paragraph_sentences])
        else:
            corefed_paragraphs_sentences.append(paragraph_sentences)

    return corefed_paragraphs_sentences

if __name__ == "__main__":
    inputs = [["This is a sentence with no coreferences."],
              ["Julie wants to buy fruit.", " That is what she loves."],
              ["I love my father and my mother.", " They work hard.", " She is always nice but he is sometimes rude."],
              ["Alice wrote her first book when she was 10 years old."],
              ["Alice is a smart girl.", "She wrote her first book when she was 10 years old."],
              ["My sister has a dog.", "She loves him."]]

    for output in batch_coref_replace(inputs):
        print(output)
        ['This is a sentence with no coreferences.']
        ['Julie wants to buy fruit.', ' That is what Julie loves.']
        ['I love my father and my mother.', ' my father and my mother work hard.', ' my mother is always nice but he is sometimes rude.']
        ['Alice wrote her first book when she was 10 years old.']
        ['Alice is a smart girl.', 'Alice wrote her first book when she was 10 years old.']

    print("----------")

    inputs = [["I love my father and my mother.", " They work hard.", " She is always nice but he is sometimes rude."]]
    print(batch_coref_replace(inputs, exclude_texts=["She"]))
    # [['I love my father and my mother.', ' my father and my mother work hard.', ' She is always nice but he is sometimes rude.']]
