This repository presents the code for the experiments in the paper called: Less Forgetting for Better Generalization: Exploring Continual-learning Fine-tuning Methods for Speech Self-supervised Representations. 

For every presented a script is present, to run an experiment you need to run the script with the according param file, for instance:

```
python replay.py hparams/auto_replay.yaml
```


The three csv files in the "danish\_splits" folder contain the exact split done for training, validation and testing concerning the Danish ASR task.
