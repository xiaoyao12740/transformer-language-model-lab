# Model card

## Intended use

Education, architecture inspection, sampling comparison and causal-attention testing. Not a general-purpose assistant or production text service.

## Data and model

Character model trained on a 600,000-character public-domain Shakespeare excerpt. Decoder-only Transformer: block 96, width 96, four heads, three layers, 357,280 parameters.

## Evaluation

On the held-out contiguous test suffix: CE 2.1271, character perplexity 8.3907, BPC 3.0688 and next-character accuracy 38.01%. These are single-seed character-tokenizer results.

## Limitations

Narrow historical English domain; no factuality, alignment, toxicity or memorization audit; stochastic generations may be incoherent; no production latency/SLA claim. Character perplexity is not comparable with subword-model perplexity.

