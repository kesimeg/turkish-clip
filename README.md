# turkish-clip

# What is this project? 

This project *replicates OpenAi's CLIP in for Turkish language. 
CLIP model consists of an image and a text encoder. Encoders are not trained from scratch. For image encoder pre-trained Resnet18 is used.
For text encoder pre-trained DistillBert for Turkish language is used.

Flicker8 dataset is used for training. For Turkish captions TasvirEt dataset is used which is a dataset for Turkish Flicker8 captions.

The notebooks are inteded to run on Colab although it is not necessary. Necessary library installations are done in notebooks in order to run on Colab.

*The original paper use Vision transformer. I used Resnet18 to ease the training.


# What is CLIP?

![alt text](./Images/overview-a.png?raw=true "Title")

CLIP is a vision model introduced by OpenAI. What makes it unique is that it learns visual concepts rather than detecting specific objects.
This is done by using an image and a caption describing the image. As an example lets assume that we have an image and two captions. Lets assume that one caption suits to the image while the other does not. We feed the image to a vision model and get a feature vector (like Resnet).
We do the same thing to text with a text based model (like BERT). Then we do multiply the image vector with these two caption vectors. This operation will give us a matrix. Ideally the diagonal of the matrix should be higher then the rest. If you remember from linear algebra product of two vectors divided by their legnths gives something called cosine similarity. This shows how similar two vectors are. Cosine similarity of a vector with it self will be 1. So going back to image and caption vectors we want to make our caption vector and image vector similar. We can do that by increasing the similarity for correct image,caption pairs and decreasing for wrong image,caption pairs. This corresponds to increasing elements in the diagonal and decreasing the rest. After this our model will not just check for specific objects in an image but rather have a visual understanding.

To give a solid example lets say that you have a set of images and you want to find images of a beach during sunset. Normally you would have to have two classifiers one for sunset and one for beach.
With clip what you can do is write "Photo of a beach during sunset" and feed it to the text encoder. This will give you a caption vector. Now you can check how "similar" each image you have to this caption vector because the text encoder knows how things are supposed to look. With this you can instantly find your images. This representation if also much more robust compared to traiditional image classifiers.

# What is in the notebooks?

CLIP\_training\_Tasviret.ipynb -> Training code of CLIP model with TasvirEt dataset

Inference.ipynb -> Some samples from the dataset are given to the model with somewhat confusing captions to see its behaviour.
The captions are in Turkish since the model expects them in that way. However I have given their English translations as well for non-turkish speakers. 

# How to use the model?

In order to use the model use can use the class in model.py like the example below:

```Python
from model import Net
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer, AutoModel

model = Net()
# If you use model on cpu you need the map_location part
model.load_state_dict(torch.load("clip_model.pt", map_location=torch.device('cpu'))) 
model.eval()

tokenizer = AutoTokenizer.from_pretrained("dbmdz/distilbert-base-turkish-cased")

transform=transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
],
)

def predict(img,text_vec):
  input = transform(img).unsqueeze(0)
  token_list = tokenizer(text_vec,padding = True)

  text = torch.Tensor(token_list["input_ids"]).long()
  mask = torch.Tensor(token_list["attention_mask"]).long()


  image_vec, text_vec = model(input, text , mask) 
  print(F.softmax(torch.matmul(image_vec,text_vec.T),dim=1))

img = Image.open("dog.png") # A dog image

text_vec = ["Çimenler içinde bir köpek.","Bir köpek.","Çimenler içinde bir kuş."] # Descriptions
predict(img,text_vec) # Probabilities for each description

```

# What can be done more?
If you want to play with the model you can try different pretrained models for image encoder. I chose Resnet18 for computation and memory reasons. You can check Vision transformer or a bigger Resnet.
You can also play with hyper parameters, add dropout, add augmentation. Again for computation reasons I couldn't try many variations. You can try to improve the performance. 

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

Image is taken from:
https://openai.com/blog/clip/

