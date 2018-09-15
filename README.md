# SCI2S at TASS 2018: Emotion Classification with Recurrent Neural Networks

When you read news about natural disasters, you usually feel negative emotions, and when you read news about the last championship won by your favourite football team, you usually feel positive emotions. If you want to promote your brand, you desire that the ads of your brand will be close to news that arouse positive emotions. Therefore, the identification of the emotions that can arouse news are very important for the reputations of brands. In this repository we have developed systems that can classify wheter a news is positive (it can arouse positive emotions) or negative (it can arouse negative emotions) for Task-4 of TASS-2018: Workshop on Semantic Analysis at SEPLN.

Please use the following citation:

```
@InProceedings{rodriguezBarroso:2018,
  author    = {Rodr{\'i}guez Barroso, Nuria and Mart{\'i}nez C{\'a}mara, Eugenio and Herrera Triguero, Francisco},
  title     = {SCI2S at TASS 2018: Emotion Classification with Recurrent Neural Networks},
  booktitle = {Proceedings of the 7th Workshop on Semantic Analysis at SEPLN (TASS-2018)},
  month     = September,
  year      = {2018},
  pages     = {(to appear)}
}
```

> **Abstract:** In this paper, we describe the participation of the team $SCI^2S$ in all the Subtasks of the Task 4 of TASS 2018. We claim that the use of external emotional
knowledge is not required for the development of an emotional classification system. Accordingly, we propose three Deep Learning models that are based on a sequence encoding layer built on a Long Short-Term Memory gated-architecture of Recurrent Neural Network. The results reached by the systems are over the average in the two Subtasks, which shows that our claim holds.


Contact person: Eugenio Martínez Cámara, emcamara@decsai.ugr.es and Nuria Rodríguez Barroso, rbnuria@correo.ugr.es

https://sci2s.ugr.es/


Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 

## Project structure
**(change this as needed!)**

* `folder/src` -- The code of the experiments
* `folder/src/tass_2018_task_4_subtask1_train_dev` -- Train data set subtask 1
* `folder/src/tass_2018_task_4_subtask2_train_dev` -- Train data set subtask 2
* `folder/src/tass_2018_task_4_subtask1_test_l1_l2` -- Test data set subtask 1

## Requirements

* python 3.6.3
* tensorflow 1.9.0
* numpy 1.14.5
* scikit-learn 0.19.1
* Word embeddings: You have to use ...

## Installation

In order to run the code used for the experiments of the paper, you have to know the following:

You can evaluate the three neural arquitectures of the paper:
  * Single LSTM: slstm.py
  * biLSTM: bilstm.py
  * Sequentia LSTM: selstm.py


Configuration:
------------------

All the parameters are fixed.


Run
------------------

To run the experimenst you have to go to the folder code and run the following command:

python3 [slstm/bilstm/selstm].py


### Expected results

After running the experiments, you should expect the results that are in the paper (The link will be published soon).


  
