# Model card

## Intended use

Education, architecture inspection, sampling comparison and causal-attention testing. Not a general-purpose assistant or production text service.

## Data and model

Character model trained on the complete 1,115,394-character public-domain Shakespeare source. Decoder-only Transformer: block 96, width 96, four heads, three layers, 357,473 parameters.

## Evaluation

On 55,675 unique held-out targets: CE 2.1991, character perplexity 9.0173, BPC 3.1727 and next-character accuracy 35.65%. Count Bigram alpha 0.5 reaches CE 2.4904 and PPL 12.0655.

## Limitations

Narrow historical English domain; no factuality, alignment, toxicity or memorization audit; stochastic generations may be incoherent; no production latency/SLA claim. Character perplexity is not comparable with subword-model perplexity.
