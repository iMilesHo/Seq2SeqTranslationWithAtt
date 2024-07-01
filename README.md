# Seq2Seq Machine Translation with Attention

Seq2Seq Machine Translation with Attention

### Objectives:

Implement a sequence-to-sequence model with attention to perform machine translation between two languages (e.g., English to French). This task will help you understand how conditioned generation works in the context of translating sequences from one domain to another, leveraging the power of LSTMs and attention mechanisms.

1. Model Architecture (10 Points)
   •Construct a seq2seq model comprising an encoder and a decoder. Both the encoder
   and decoder should use LSTM layers to effectively capture the temporal dependencies
   in the input and target sequences.
   •Incorporate an attention mechanism between the encoder and decoder to improve the
   model’s ability to focus on relevant parts of the input sequence during translation,
   as discussed in the "Attention Mechanisms"section of the lectures.
2. Dataset and Preprocessing (10 Points)
   •Choose a bilingual corpus as your dataset (e.g., a collection of English-French sen-
   tence pairs). Perform necessary preprocessing steps, including tokenization, conver-
   ting text to sequences of integers, and padding sequences to a uniform length.
3. Training (10 Points)
   •Train your seq2seq model on the preprocessed dataset. The goal is to minimize the
   difference between the predicted translation output by the decoder and the actual
   target sentence in the dataset. Experiment with different hyper-parameters, such as
   the number of LSTM units, learning rate, and batch size, to optimize your model’s
   performance.
4. Translation and Evaluation (10 Points)
   •Use your trained model to translate a set of sentences from the source language
   to the target language. Evaluate the quality of your translations using a suitable
   metric, such as BLEU (Bilingual Evaluation Understudy) score, to quantitatively
   measure how your translations compare to a set of reference translations.
5. Analysis (10 Points)
   •Discuss the impact of the attention mechanism on the translation quality. Compare
   the performance of your model with and without attention using both qualitative
   examples and quantitative metrics. Reflect on how different architectural choices
   and hyper-parameters affected the outcomes.
