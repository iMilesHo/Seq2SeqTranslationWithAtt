#import "template.typ": *

// Take a look at the file `template.typ` in the file panel
// to customize this template and discover how it works.
#show: project.with(
  title: "Report",
  authors: (
    (name:"Lyle He", studentNumber: "", email:""),
  ),
  subTitle: "Seq2Seq Machine Translation with Attention",
  date: "Feb 25, 2024",
)

= Seq2Seq Machine Translation with Attention

== Introduction
In this part, I have implemented a seq2seq model with attention for machine translation between English and French and analyzed the beneficial of the attention mechanism and compared the performance of the seq2seq model with attention and the seq2seq model without attention.

In addition, my project code is organized as follows, the first two python notebook files are used to run the model in kaggle to utilize the GPU resources and the last one is the main project code for the seq2seq model.
#rect[
1. python notebook file: `neuralmachinetranslation_hyperparameter_grid_search.ipynb`
2. python notebook file: `neuralmachinetranslation_final_training.ipynb`
3. pure python project: (this is the main project code for the seq2seq model with attention for machine translation between English and French)
    - ./Seq2Seq-Attention-Model
      - data_preprocessing.py
      - eng\_-french.csv (you can download from kaggle[3])
      - model.py
      - train_test_the_model.py
      - utils.py
      - run.py
]


The model architecture, dataset, preprocessing, training, translation, and evaluation are shown in the below.

== Model Architecture
According to the assignment requirement, I have implemented a seq2seq model with attention for machine translation between English and French. The model architecture is shown in the below figure and text description.

```python
NMTModel(
  (embeddings): WordEmbeddingForTranlationTask(
    (source): Embedding(14516, 512, padding_idx=0)
    (target): Embedding(28336, 512, padding_idx=0)
  )
  (encoder): LSTM(512, 512, bidirectional=True)
  (decoder): LSTMCell(1024, 512)
  (encoder_hidden_to_initial_decoder_hidden): 
    Linear(in_features=1024, out_features=512, bias=False)
  (encoder_cell_to_initial_decoder_cell): 
    Linear(in_features=1024, out_features=512, bias=False)
  (encoder_hidden_to_decoder_hidden_for_attention): 
    Linear(in_features=1024, out_features=512, bias=False)
  (combined_output_and_hidden_to_hidden): 
    Linear(in_features=1536, out_features=512, bias=False)
  (target_vocab_projection): 
    Linear(in_features=512, out_features=28336, bias=False)
  (dropout): Dropout(p=0.5, inplace=False)
)
```

As we can see from the above code, the model consists of an embedding layer (for both source and target languages), an encoder, a decoder, and several linear layers for dimension prejection. The encoder is a bidirectional LSTM layer, and the decoder is a LSTMCell layer. The attention mechanism is implemented according to the lectures as shown in the Figure 1.

#figure(
  image("./images/image1.png", width: 50%),
  caption: [
    Seq2Seq Model with Attention [1]
  ],
)

The math equation for this model is from the lectures and the attention mechanism is according to Global Attention Model (Luong, et al. 2015)[2]. Below are the key python code implementation for this model.

#figure(
  image("./images/image13.png", width: 100%),
  caption: [
    Seq2Seq Model Forward Processes
  ],
)
#figure(
  image("./images/image14.png", width: 100%),
  caption: [
    Seq2Seq Model Forward Processes
  ],
)
#figure(
  image("./images/image15.png", width: 100%),
  caption: [
    Seq2Seq Model Forward Processes
  ],
)

In the above code, the key part is the `decode` function. We need to loop through the target sequence and compute the attention mechanism at each time step. Before we compute the attention mechanism, we need to concatenate the word embedding at time t and the previous output and input it into the LSTM cell. Then we can compute the attention mechanism and the combined output at time t. The combined output will be the input for the next time step. The `step` function is the implementation of the attention mechanism. 

The `test` function is the function to generate the translated text which is similar to the `decode` function but with a different loop condition and we use the generated word at time t as the input for the next time step instead of the word embedding at time t.

== Dataset and Preprocessing

I choose a medium-sized dataset for this task from kaggle[3]. There are 175621 pairs of English and French sentences. The vocabulary size for English is 14516 and for French is 28336. 

1. I remove the non-meaningful characters (e.g. non-breaking space, etc.)
2. remove accents of the French sentences, and convert all the characters to lowercase. 
3. Then I tokenize the sentences, get the vocabularies of the source and target languages respectively, add 'start', 'end', 'unknown' , and 'padding' tokens to the vocabularies
4. convert the sentences to sequences of integers. 
5. I pad the sequences to the same length and split the dataset into training and validation sets. 
5. The result is shown in the python notebook file.

=== Data Preprocessing Results
```python
number of samples 175621
characters in English sentences
 {'N', '"', 'e', '0', 'p', 'b', '%', 'P', 'é', '—', 'а', '-', 'r', 'T', '7', 'M', '2', 'y', 'W', 'Z', 'c', 'ç', 's', 'h', '8', 'O', 'd', 'Q', 'k', 't', 'ö', ',', 'L', 'C', '4', '&', '\xa0', 'a', '€', 'D', 'I', 'u', 'Y', 'U', 'l', 'z', 'x', 'v', '’', 'n', '.', 'K', 'E', 'ú', 'R', 'F', '1', '/', ' ', 'j', 'i', '!', '6', '?', 'G', 'g', "'", 'B', '5', 'f', '–', 'w', '+', 'V', 'X', 'A', '3', 'º', 'o', 'm', ':', 'q', 'S', '₂', '$', ';', 'H', '‘', '\xad', 'J', '9'}
characters in French sentences
 {'è', 'N', '"', 'e', 'À', '0', 'p', 'b', '%', 'P', 'é', '-', 'r', 'T', '7', 'M', '2', 'y', 'W', 'Z', 'c', 'ç', 'ê', 's', 'h', '8', 'd', 'O', 'Q', 'k', 't', 'ö', '\u202f', ',', 'L', 'C', 'É', '4', '&', '\xa0', 'a', 'D', 'ô', 'I', 'î', 'u', 'U', 'Y', 'l', 'z', '«', 'x', 'v', '’', 'n', 'С', '…', '.', 'ù', 'Ê', ')', 'K', 'E', '\u2009', 'R', '»', '1', 'F', '/', '(', ' ', 'j', 'i', 'Ô', '!', '\u200b', '6', '?', 'Ç', 'G', 'g', "'", 'â', 'B', '5', 'œ', 'ë', 'f', '–', 'w', '+', 'V', 'X', 'A', '3', 'û', 'o', 'á', 'm', ':', 'à', '‽', 'q', 'S', '₂', 'ï', '$', 'Â', ';', 'H', '‘', 'J', '9'}
length of Engish characters in English sentences 91
length of French characters in French sentences 113
Remove the unmeaningful character and convert to uppercase, and keep any meaningful characters
length of en character set after cleaning 51
length of fr character set after cleaning 55
Tokenize the sentences
English-French pairs Examples
Src: hi.	Tgt: salut!
Src: run!	Tgt: cours !
Src: run!	Tgt: courez !
Src: who?	Tgt: qui ?
Src: wow!	Tgt: ca alors !
Src: fire!	Tgt: au feu !
Build the vocabulary
English vocabulary Examples
<SOS>	: 2
hi	: 3
.	: 4
...
length of fr_vocab, 28336
number of samples - en_sequence:  175621
number of samples - fr_sequence:  175621
data preprocessing done
```
#figure(
  image("./images/image.png", width: 60%),
  caption: [
    Data Preprocessing Results
  ],
)

== Training

=== Hyperparameters Search

- Different Batch size: 32, 64, 128, 256
// A function to represent a virtual image
#let vimg(body) = {
    rect(width: 10mm, height: 5mm)[
        #text(body)
    ]
}

#figure(
    grid(
        columns: 2, 
        gutter: 0mm,
        image("./images/image3.png", width: 100%),
        image("./images/image4.png", width: 100%),
    ),
    caption: "Different Batch size"
)
As we can see the difference between different batch sizes is not significant. The batch size of 128 is chosen for the final model.

- embedding_dim: 128, 256, 512
#figure(
    grid(
        columns: 2, 
        gutter: 0mm,
        image("./images/image5.png", width: 100%),
        image("./images/image6.png", width: 100%),
    ),
    caption: "Different Batch size"
)

As we also could see the difference between different embedding dimensions is not significant. The embedding dimension of 512 is chosen for the final model.

- hidden_dim: 128, 256, 512
#figure(
    grid(
        columns: 2, 
        gutter: 0mm,
        image("./images/image7.png", width: 100%),
        image("./images/image8.png", width: 100%),
    ),
    caption: "Different Batch size"
)
As we can see the 512 seems to have a better performance than the other two. So I choose the hidden dimension of 512 for the final model.

- dropout_rate: 0.1, 0.2, 0.5
#figure(
    grid(
        columns: 2, 
        gutter: 0mm,
        image("./images/image9.png", width: 100%),
        image("./images/image10.png", width: 100%),
    ),
    caption: "Different Batch size"
)

As we can see the dropout rate of 0.5 seems to have a better performance than the other two. So I choose the dropout rate of 0.5 for the final model.

=== Final Trainning Results 

After tested different hyperparameters and the best hyperparameters are shown in the below table. The training process is shown in the python notebook file.

#let table_3_1=text("Batch size")
#let table_3_2=text("128")
#let table_4_1=text("Learning rate")
#let table_4_2=text("0.001")
#let table_5_1=text("Epochs")
#let table_5_2=text("10")
#let table_6_1=text("Optimizer")
#let table_6_2=text("Adam")
#let table_7_1=text("Loss function")
#let table_7_2=text("negative log probability loss")

#align(center, 
    table(
    columns: 2,
    align: left,
    [Hyperparameters], [Value],
    [#table_3_1], [#table_3_2],
    [embedding_dim], [512],
    [hidden_dim], [512],
    [#table_4_1], [#table_4_2],
    [dropout], [0.5],
    [#table_5_1], [#table_5_2],
    [#table_6_1], [#table_6_2],
    [#table_7_1], [#table_7_2],
    )
)

Here is the loss curve of the training process.

#figure(
  image("./images/image11.png", width: 60%),
  caption: [
    Loss Curve
  ],
)


== Translation and Evaluation

According to the assignment requirement, I tested the model in my test set and the result is shown in the below table.

#align(center, 
    table(
    columns: 2,
    align: left,
    [Model], [BLEU Score],
    [Seq2Seq with Attention], [0.29],
    )
)

=== Translation Example

I also show an example of the translation result in the below.

#rect[
Src: What is your name ?\
Tgt: comment vous nom ?

Src: I am doing great\
Tgt: je suis bon en train de faire .

Src: What are you doing today ?\
Tgt: que faites-vous aujourd'hui ?

Src: Can you help me with my homework ?\
Tgt: pouvez-vous m'aider a mes devoirs ?

Src: I am a student at the university and I am studying computer science\
Tgt: je suis etudiant a l'universite et j'ai etudier le ordinateur .

Src: In my opinion , the best way to learn a new language is to practice speaking with native speakers\
Tgt: selon mon avis , le meilleur moyen est de pratiquer avec ce qui est a parler de parler un truc de monde a apprendre le pays de parler avec ce qui est a parler de parler .

Src: I usually wake up at 6 am and then I go for a run in the park before I start working\
Tgt: je me leve habituellement au restaurant et puis je commence a travailler dans le parc avant .
]

- In order to let you know the translation result, I also show the inverted translation result in the below.

#rect[
Src: What is your name?\
Tgt: what is your name?

Src: I am doing great\
Tgt: I'm good at doing.

Src: What are you doing today?\
Tgt: what are you doing today?

Src: Can you help me with my homework?\
Tgt: can you help me with my homework?

Src: I am a student at the university and I am studying computer science\
Tgt: I am a university student and I studied computer science.

Src: In my opinion, the best way to learn a new language is to practice speaking with native speakers\
Tgt: in my opinion, the best way is to practice with what is to speak of speaking a world thing to learn the country of speaking with what is to speak of speaking.

Src: I usually wake up at 6 am and then I go for a run in the park before I start working\
Tgt: I usually get up at the restaurant and then I start working in the front park.
]


As we can see from the translation result, the translated French sentences are almost the same as the target sentences. So, basically, the model can translate the simple English sentences to French sentences correctly.


== Analysis
In this section, I will analyze the beneficial of the attention mechanism and compare the performance of the seq2seq model with attention and the seq2seq model without attention. The result is shown in the below.

=== Beneficial of Attention Mechanism
I recorded the attention result of the some of the translated sentences (shown above) and the result is shown in the below.

#figure(
    grid(
        columns: 2, 
        gutter: 0mm,
        image("./images/output1.png", width: 100%),
        image("./images/output2.png", width: 100%),
        image("./images/output3.png", width: 100%),
        image("./images/output4.png", width: 100%),
    ),
    caption: "Different Batch size"
)

#figure(
    grid(
        columns: 2, 
        gutter: 0mm,
        image("./images/output5.png", width: 100%),
        image("./images/output6.png", width: 100%),
        image("./images/output7.png", width: 100%),
    ),
    caption: "Different Batch size"
)

Obviously, as we can see the attention mechanism can help the model to focus on the related parts of the input sequence and generate the translated text. 

For example, in the first sentence, the attention mechanism can help the model to extract the corresponding French words for the English words. 


=== Comparison with Seq2Seq without Attention
I implemented another seq2seq model without attention and compared the performance of the two models. Here is the result.

#rect[
Src: What is your name?\
Tgt: si nom ?

Src: I am doing great\
Tgt: je ne suis pas un comme un .

Src: What are you doing today?\
Tgt: ce que est-ce que tu manges ?

Src: Can you help me with my homework?\
Tgt: si quelqu'un avec moi d'acquerir .

Src: I am a student at the university and I am studying computer science\
Tgt: un jour , je ne suis pas sur de chouette et de meme .

Src: In my opinion, the best way to learn a new language is to practice speaking with native speakers\
Tgt: si que ce qui sache , c'est quelqu'un de tres causer des conseils , que l'un de mes parents reviennent .

Src: I usually wake up at 6 am and then I go for a run in the park before I start working\
Tgt: je me fiche que je vais m'allonger au tennis avant qu'ils ne soient pas bien sous une heure .
]

And the in order to let you know the translation result, I also show the inverted translation result in the below.

#rect[
Src: What is your name?\
Tgt: if name?

Src: I am doing great\
Tgt: I'm not one like one.

Src: What are you doing today?\
Tgt: what are you eating?

Src: Can you help me with my homework?\
Tgt: if anyone with me to acquire .

Src: I am a student at the university and I am studying computer science\
Tgt: one day, I'm not sure if it's cool or the same.

Src: In my opinion, the best way to learn a new language is to practice speaking with native speakers\
Tgt: if anyone knows, it's someone who can give advice, that one of my parents comes back.

Src: I usually wake up at 6 am and then I go for a run in the park before I start working\
Tgt: I don't care if I'm going to lie down at tennis before they're not well under an hour.
]

And the translating loss curve is shown in the below.

#figure(
  image("./images/image12.png", width: 60%),
  caption: [
    Loss Curve
  ],
)

The result of model without attention is aboviouly worse than the model with attention. The translated French sentences are not that coherent.

=== Final Comparison Result
Here is the final comparison result of the two models.
#align(center, 
    table(
    columns: 2,
    align: left,
    [Model], [BLEU Score],
    [Seq2Seq with Attention], [0.29],
    [Seq2Seq without Attention], [0.21],
    )
)

As we can see from the above table, the seq2seq model with attention has a much better performance than the model without attention. The attention mechanism can help the model to focus on the important parts of the input sequence and improve the translation performance.

= References
[1] https://arxiv.org/pdf/1409.0473.pdf
[2] https://arxiv.org/pdf/1508.04025.pdf
[3] https://www.kaggle.com/tilii7/englishfrench
[4] https://www.gutenberg.org/cache/epub/84/pg84-images.html