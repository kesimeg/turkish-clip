# turkish-clip

This project replicates OpenAi's CLIP in for Turkish language. 
CLIP model consists of an image and a text encoder. Encoders are not trained from scratch. For image encoder pre-trained Resnet18 is used.
For text encoder pre-trained DistillBert for Turkish language is used.

Flicker8 dataset is used for training. For Turkish captions TasvirEt dataset is used which is a dataset for Turkish Flicker8 captions.


# What is in the notebooks?

CLIP\_training\_Tasviret.ipynb -> Training code of CLIP model with TasvirEt dataset

Inference.ipynb -> Some samples from the dataset are given to the model with somewhat confusing captions to see its behaviour.
The captions are in Turkish since the model expects them in that way. However I have given their English translations as well for non-turkish speakers. 

# Resources

I want to thank to sources below which I have used to make this project:

https://towardsdatascience.com/simple-implementation-of-openai-clip-model-a-tutorial-ace6ff01d9f2?gi=f69bd3da2189

TasvirEt (Turkish Flicker8 captions):
https://www.kaggle.com/datasets/begum302553/tasviret-flickr8k-turkish 

TasvirEt paper:
https://ieeexplore.ieee.org/document/7496155

Turkish DistilBERT:
https://huggingface.co/dbmdz/distilbert-base-turkish-cased

Original CLIP paper:
https://arxiv.org/abs/2103.00020
