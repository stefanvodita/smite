# SMITE: Sheet Music Interpretation using Transformers (End-to-end)

* Have a computer read your sheet music for you!
* First usage of transformers for optical music recognition
* 92% accuracy
* 3 times as fast as closest alternative
* Presented as graduation project in July 2021
* A tad unpolished

# Demo
There is a [demo on YouTube](https://youtu.be/VDbCB1P4h18?si=gH963CnpEqOAvpOi).
The voiceover is in Romanian, but the video alone should still be useful.

# Corpus & Conversor
At the foundation of this project is the work of Jorge Calvo-Zaragoza and David Rizo,
[End-to-End Neural Optical Music Recognition of Monophonic Scores](https://www.mdpi.com/2076-3417/8/4/606/htm).
Their code can be found [here](https://github.com/OMR-Research/tf-end-to-end), and the corpus [here](https://grfia.dlsi.ua.es/primus/).
I use the same corpus and part of the sample reading and initial processing are inspired by the original code.
I recommend merging the 2 corpus directories in the archive, or removing entries corresponding to one of the directories
from `Data/train.txt`.

Converting the custom music encoding to a playable midi file can be done with the conversor provided together with the corpus.

# Requirements

## Software
I'll list what I used, but is should be possible to use newer versions as long as they are compatible amongst each other.

* Python 3.6.9
* Numpy 1.18.5
* PyTorch 1.8.1
* Transformers 4.5.1
* Cuda 10.1
* OpenCV-Python 4.4.0.44
* Polyleven 0.7

## Hardware
You should use a GPU for training.
I mostly used a Standard NC6_Promo virtual machine, which Azure is deprecating. It would (usually) be assigned a Tesla K80,
and take about 24h to reach 92% accuracy. I never kept it going much longer due to budget constraints.

If you only want to run the pretrained model, you don't need a GPU. Inference is very fast even on CPU.
You can download the pretrained model [here](https://drive.google.com/file/d/1yWt-O4HVtgJ3BYbWb5EUYb0UrhvFis2A/view?usp=sharing).

# Run it
The run command is a legacy from [End-to-End Neural Optical Music Recognition of Monophonic Scores](https://www.mdpi.com/2076-3417/8/4/606/htm),
but it could easily be adapted if need be.
```
python3 train.py -semantic -corpus <path_to_corpus_base_dir> -set Data/train.txt -vocabulary Data/vocabulary_semantic.txt  -save_model <ignored>
```

# Files
* `presentation/  `
    * `poveste.txt` - Presentation speech in Romanian  
    * `TOPIC.pptx` - Presentation in Romanian (but numeric information and diagrams are still useful)  
* `pt-end-to-end-vvit/  `
    * `Data/`  
        * `Example/` - An image and corresponding interpretation
        * `test.txt` - Test set; because of a mistake, this was not used during development, a fraction of the train set was used instead  
        * `train.txt` - Train set  
        * `vocabulary_agnostic.txt` - Agnostic musical encoging, not used  
        * `vocabulary_semantic.txt` - Semantic musical encoging  
    * `model.py` - The model class  
    * `primus.py` - Loads and preprocesses data  
    * `train.py` - Train or run the model and gather various statistics  
* `paper.pdf` - The diploma paper in English documenting the project in much more detail than this README  
* `README.md` - this  
