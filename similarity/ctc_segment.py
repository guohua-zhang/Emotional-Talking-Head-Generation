"""
Forced Alignment with Wav2Vec2
==============================

**Author** `Moto Hira <moto@fb.com>`__

This tutorial shows how to align transcript to speech with
``torchaudio``, using CTC segmentation algorithm described in
`CTC-Segmentation of Large Corpora for German End-to-end Speech
Recognition <https://arxiv.org/abs/2007.09127>`__.

"""

import torch
import torchaudio

print(torch.__version__)
print(torchaudio.__version__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Overview
# --------
#
# The process of alignment looks like the following.
#
# 1. Estimate the frame-wise label probability from audio waveform
# 2. Generate the trellis matrix which represents the probability of
#    labels aligned at time step.
# 3. Find the most likely path from the trellis matrix.
#
# In this example, we use ``torchaudio``\ ’s ``Wav2Vec2`` model for
# acoustic feature extraction.
#


from dataclasses import dataclass

torch.random.manual_seed(0)

#SPEECH_FILE = torchaudio.utils.download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")
SPEECH_FILE = 'MEAD-sim/M003/audio/neutral/level_1/001.wav'

# Generate frame-wise label probability
# -------------------------------------
#
# The first step is to generate the label class porbability of each aduio
# frame. We can use a Wav2Vec2 model that is trained for ASR. Here we use
# :py:func:`torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H`.
#
# ``torchaudio`` provides easy access to pretrained models with associated
# labels.
#
# .. note::
#
#    In the subsequent sections, we will compute the probability in
#    log-domain to avoid numerical instability. For this purpose, we
#    normalize the ``emission`` with :py:func:`torch.log_softmax`.
#

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

model = bundle.get_model().to(device)
labels = bundle.get_labels()
with torch.inference_mode():
    waveform, _ = torchaudio.load(SPEECH_FILE)
    emissions, _ = model(waveform.to(device))
    emissions = torch.log_softmax(emissions, dim=-1)

emission = emissions[0].cpu().detach()


# Generate alignment probability (trellis)
# ----------------------------------------
#
# From the emission matrix, next we generate the trellis which represents
# the probability of transcript labels occur at each time frame.
#
# Trellis is 2D matrix with time axis and label axis. The label axis
# represents the transcript that we are aligning. In the following, we use
# :math:`t` to denote the index in time axis and :math:`j` to denote the
# index in label axis. :math:`c_j` represents the label at label index
# :math:`j`.
#
# To generate, the probability of time step :math:`t+1`, we look at the
# trellis from time step :math:`t` and emission at time step :math:`t+1`.
# There are two path to reach to time step :math:`t+1` with label
# :math:`c_{j+1}`. The first one is the case where the label was
# :math:`c_{j+1}` at :math:`t` and there was no label change from
# :math:`t` to :math:`t+1`. The other case is where the label was
# :math:`c_j` at :math:`t` and it transitioned to the next label
# :math:`c_{j+1}` at :math:`t+1`.
#
# The follwoing diagram illustrates this transition.
#
# .. image:: https://download.pytorch.org/torchaudio/tutorial-assets/ctc-forward.png
#
# Since we are looking for the most likely transitions, we take the more
# likely path for the value of :math:`k_{(t+1, j+1)}`, that is
#
# :math:`k_{(t+1, j+1)} = max( k_{(t, j)} p(t+1, c_{j+1}), k_{(t, j+1)} p(t+1, repeat) )`
#
# where :math:`k` represents is trellis matrix, and :math:`p(t, c_j)`
# represents the probability of label :math:`c_j` at time step :math:`t`.
# :math:`repeat` represents the blank token from CTC formulation. (For the
# detail of CTC algorithm, please refer to the *Sequence Modeling with CTC*
# [`distill.pub <https://distill.pub/2017/ctc/>`__])
#

transcript = 'She|had|your|dark|suit|in|greasy|wash|water|all|year'.upper()
dictionary = {c: i for i, c in enumerate(labels)}

tokens = [dictionary[c] for c in transcript]
print(list(zip(transcript, tokens)))


def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra diemsions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, 0], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")

    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis


trellis = get_trellis(emission, tokens)


# Find the most likely path (backtracking)
# ----------------------------------------
#
# Once the trellis is generated, we will traverse it following the
# elements with high probability.
#
# We will start from the last label index with the time step of highest
# probability, then, we traverse back in time, picking stay
# (:math:`c_j \rightarrow c_j`) or transition
# (:math:`c_j \rightarrow c_{j+1}`), based on the post-transition
# probability :math:`k_{t, j} p(t+1, c_{j+1})` or
# :math:`k_{t, j+1} p(t+1, repeat)`.
#
# Transition is done once the label reaches the beginning.
#
# The trellis matrix is used for path-finding, but for the final
# probability of each segment, we take the frame-wise probability from
# emission matrix.
#


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


def backtrack(trellis, emission, tokens, blank_id=0):
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        raise ValueError("Failed to align")
    return path[::-1]


path = backtrack(trellis, emission, tokens)
# for p in path:
#     print(p)


# Merge the labels
@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def merge_repeats(path):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments


segments = merge_repeats(path)
print('####### segments #######')
for seg in segments:
    print(seg)
print()


# Merge words
def merge_words(segments, separator="|"):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words


word_segments = merge_words(segments)
print('####### words #######')
for word in word_segments:
    print(word)
