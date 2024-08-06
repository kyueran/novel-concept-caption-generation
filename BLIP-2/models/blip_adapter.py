'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import warnings
warnings.filterwarnings("ignore")

from models.vit import VisionTransformer, interpolate_pos_embed
from models.med import BertConfig, BertModel, BertLMHeadModel
from transformers import BertTokenizer, BitsAndBytesConfig, pipeline
from sentence_transformers import SentenceTransformer, util

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
import math

import os
from urllib.parse import urlparse
from timm.models.hub import download_cached_file
import spacy
from nltk.corpus import wordnet as wn

from PIL import Image
import torchvision.transforms as transforms

nlp = spacy.load("en_core_web_sm")
    
class BLIP_Base(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 224,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,                 
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)

        
    def forward(self, image, caption, mode):
        
        assert mode in ['image', 'text', 'multimodal'], "mode parameter must be image, text, or multimodal"
        text = self.tokenizer(caption, return_tensors="pt").to(image.device) 
        
        if mode=='image':    
            # return image features
            image_embeds = self.visual_encoder(image)             
            return image_embeds
        
        elif mode=='text':
            # return text features
            text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                            return_dict = True, mode = 'text')  
            return text_output.last_hidden_state
        
        elif mode=='multimodal':
            # return multimodel features
            image_embeds = self.visual_encoder(image)    
            image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)      
            
            text.input_ids[:,0] = self.tokenizer.enc_token_id
            output = self.text_encoder(text.input_ids,
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,      
                                       return_dict = True,
                                      )              
            return output.last_hidden_state
        

class Adapter(nn.Module):
    def __init__(self, input_dim, adapter_dim):
        super(Adapter, self).__init__()
        self.down_project = nn.Linear(input_dim, adapter_dim)
        self.activation = nn.ReLU()
        self.up_project = nn.Linear(adapter_dim, input_dim)
        self.init_weights()  # Initialize weights

    def init_weights(self):
        init.kaiming_uniform_(self.down_project.weight, a=math.sqrt(5))
        init.zeros_(self.down_project.bias)
        init.kaiming_uniform_(self.up_project.weight, a=math.sqrt(5))
        init.zeros_(self.up_project.bias)

    def forward(self, x):
        down = self.down_project(x)
        activated = self.activation(down)
        up = self.up_project(activated)
        return up + x  # Add residual connection

class BertOutputWithAdapter(nn.Module):
    def __init__(self, config, adapter_dim=64):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.adapter = Adapter(config.hidden_size, adapter_dim)
        self.init_weights()  # Initialize weights

    def init_weights(self):
        init.kaiming_uniform_(self.dense.weight, a=math.sqrt(5))
        init.zeros_(self.dense.bias)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        hidden_states = self.adapter(hidden_states)  # Apply adapter
        return hidden_states

        
class BLIP_Decoder(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 384,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 prompt = 'a picture of ',
                 adapter_dim=64,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """            
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_decoder = BertLMHeadModel(config=med_config)
        self.device = 'cuda'  
        self.kl_loss_weight = 0.01
        
        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids)-1

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        self.pipe_image_to_text = pipeline("image-to-text", 
                                           model="llava-hf/llava-v1.6-mistral-7b-hf", 
                                           model_kwargs={"quantization_config": quantization_config})
        
        self.semantic_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        #self.summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")
        
        # Freeze BLIP model weights
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
        for param in self.text_decoder.parameters():
            param.requires_grad = False
        for param in self.semantic_model.parameters():
            param.requires_grad = False

        self.adapter = Adapter(med_config.hidden_size, adapter_dim)
        self.text_decoder.bert.encoder.layer[-1].output = BertOutputWithAdapter(med_config, adapter_dim)

        self.transform = transforms.ToPILImage()
  
    def forward(self, image, caption):
        
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        
        text = self.tokenizer(caption, padding='longest', truncation=True, max_length=40, return_tensors="pt").to(image.device) 
        
        text.input_ids[:,0] = self.tokenizer.bos_token_id
        
        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)         
        decoder_targets[:,:self.prompt_length] = -100
     
        decoder_output = self.text_decoder(text.input_ids, 
                                           attention_mask = text.attention_mask, 
                                           encoder_hidden_states = image_embeds,
                                           encoder_attention_mask = image_atts,                  
                                           labels = decoder_targets,
                                           return_dict = True,   
                                          )   
        

        teacher_caption_objects = self.teacher_caption_objects(image)

        teacher_caption_desc = self.teacher_caption_desc(image)

        student_caption = self.generate(image)
        
        loss_lm = decoder_output.loss

        object_matching_loss = torch.tensor(self.compute_f1_score_loss(student_caption, teacher_caption_objects), requires_grad=True).to(self.device)

        semantic_similarity_loss = torch.tensor(self.compute_semantic_similarity_loss(student_caption, teacher_caption_desc), requires_grad=True).to(self.device)
        
        kl_loss = self.compute_kl_loss(image, student_caption, teacher_caption_desc)
        
        total_loss = loss_lm + object_matching_loss + semantic_similarity_loss + self.kl_loss_weight * kl_loss
        
        print("LM LOSS: ", loss_lm)
        print("OM LOSS: ", object_matching_loss)
        print("SM LOSS: ", semantic_similarity_loss)
        print("KL LOSS: ", kl_loss)
        print("Total LOSS ", total_loss)
        return total_loss

    def teacher_caption_desc(self, image):
        # Convert each image in the batch to PIL format
        pil_images = [self.transform(img) for img in image]
        prompt = "USER: <image>\nPlease describe the image briefly.\nASSISTANT:"
        responses = [self.pipe_image_to_text(img, prompt=prompt, generate_kwargs={"max_new_tokens": 200})[0]["generated_text"] for img in pil_images]
        teacher_captions = []
        for response in responses:
            assistant_index = response.find("ASSISTANT:")
            if assistant_index != -1:
                response = response[assistant_index + len("ASSISTANT:"):].strip()
            teacher_captions.append(response)
        return teacher_captions
    
    def teacher_caption_objects(self, image):
        # Convert each image in the batch to PIL format
        pil_images = [self.transform(img) for img in image]
        prompt = "USER: <image>\nWhat are the objects present in the image.\nASSISTANT:"
        responses = [self.pipe_image_to_text(img, prompt=prompt, generate_kwargs={"max_new_tokens": 200})[0]["generated_text"] for img in pil_images]
        teacher_captions = []
        for response in responses:
            assistant_index = response.find("ASSISTANT:")
            if assistant_index != -1:
                response = response[assistant_index + len("ASSISTANT:"):].strip()
            teacher_captions.append(response)
        return teacher_captions
    
    def compute_semantic_similarity_loss(self, student_caption, teacher_caption):
        student_embeddings = self.semantic_model.encode(student_caption, convert_to_tensor=True)
        teacher_embeddings = self.semantic_model.encode(teacher_caption, convert_to_tensor=True)
        
        similarity_matrix = util.pytorch_cos_sim(student_embeddings, teacher_embeddings)
        diagonal_similarities = similarity_matrix.diag().tolist()
        average_similarity = sum(diagonal_similarities) / len(diagonal_similarities)
        semantic_similarity_loss = 1 - average_similarity
        return semantic_similarity_loss
    
    '''
    def summarize(self, caption):
        summary = self.summarizer(caption, max_length=50, do_sample=False)
        summarized = summary[0]['summary_text']
        return summarized
    '''

    def extract_objects_features(self, text):
        doc = nlp(text)
        objects_features = set()
        for token in doc:
            if token.pos_ in {'NOUN'}:
                objects_features.add(token.lemma_.lower())

        objects_features.discard('object')
        objects_features.discard('image')
        return objects_features
    
    def create_synonym_dict(self, objects_features):
        synonym_dict = {}
        for obj in objects_features:
            synonyms = set([obj])
            for syn in wn.synsets(obj):
                for lemma in syn.lemmas():
                    synonyms.add(lemma.name().lower())
            synonym_dict[obj] = synonyms
        return synonym_dict
    
    def match_with_synonyms(self, objects_features1, synonym_dict2):
        matched_objects = set()
        for obj1 in objects_features1:
            for obj2, synonyms in synonym_dict2.items():
                if obj1 in synonyms:
                    matched_objects.add(obj1)
                    break
        return matched_objects

    def calculate_precision_recall_f1(self, set1, set2, matched_objects):
        true_positives = len(matched_objects)
        possible_positives = len(set1)
        predicted_positives = len(set2)

        precision = true_positives / predicted_positives if predicted_positives > 0 else 0
        recall = true_positives / possible_positives if possible_positives > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return precision, recall, f1

    def compute_f1_score_loss(self, student_captions, teacher_captions):

        assert(len(student_captions) == len(teacher_captions))

        list_of_f1_scores = []
        for i in range(len(student_captions)):
            teacher_caption = teacher_captions[i]
            student_caption = student_captions[i]
            objects_features_student = self.extract_objects_features(student_caption)
            objects_features_teacher = self.extract_objects_features(teacher_caption)
            
            synonym_dict_teacher = self.create_synonym_dict(objects_features_teacher)
            
            matched_objects = self.match_with_synonyms(objects_features_student, synonym_dict_teacher)
            
            _, _, f1_score = self.calculate_precision_recall_f1(objects_features_student, objects_features_teacher, matched_objects)
            list_of_f1_scores.append(f1_score)
        
        mean_f1_score = sum(list_of_f1_scores) / len(list_of_f1_scores)
        return 1 - mean_f1_score

    def compute_kl_loss(self, image, student_caption, teacher_caption):
    # Tokenize the captions using the shared tokenizer
        student_ids = self.tokenizer(student_caption, return_tensors="pt", padding=True, truncation=True).input_ids
        teacher_ids = self.tokenizer(teacher_caption, return_tensors="pt", padding=True, truncation=True).input_ids
        
        # Move token ids to the appropriate device
        student_ids = student_ids.to(self.device)
        teacher_ids = teacher_ids.to(self.device)
        
        # Compute the logits (probability distributions) from the decoder
        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts}
        
        student_logits = self.text_decoder(student_ids, **model_kwargs).logits
        teacher_logits = self.text_decoder(teacher_ids, **model_kwargs).logits
        
        # Determine the maximum length
        max_length = max(student_logits.size(1), teacher_logits.size(1))
        
        # Pad student logits to match teacher's length if necessary
        if student_logits.size(1) < max_length:
            pad_size = max_length - student_logits.size(1)
            padding = torch.zeros((student_logits.size(0), pad_size, student_logits.size(2)), device=student_logits.device)
            student_logits = torch.cat([student_logits, padding], dim=1)
        
        # Pad teacher logits to match student's length if necessary
        if teacher_logits.size(1) < max_length:
            pad_size = max_length - teacher_logits.size(1)
            padding = torch.zeros((teacher_logits.size(0), pad_size, teacher_logits.size(2)), device=teacher_logits.device)
            teacher_logits = torch.cat([teacher_logits, padding], dim=1)
        
        # Ensure the logits tensors match in size
        student_logits = student_logits[:, :max_length, :]
        teacher_logits = teacher_logits[:, :max_length, :]
        
        # Convert logits to log-probabilities and probabilities
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        
        # Compute the KL divergence
        kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
        kl_loss = kl_loss_fn(student_log_probs, teacher_probs)
        
        return kl_loss



    def generate(self, image, sample=False, num_beams=1, max_length=30, min_length=10, top_p=0.9, repetition_penalty=1.0):
        image_embeds = self.visual_encoder(image)

        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams,dim=0)
            
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask":image_atts}
        
        prompt = [self.prompt] * image.size(0)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(image.device) 
        input_ids[:,0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1] 

        if sample:
            #nucleus sampling
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  do_sample=True,
                                                  top_p=top_p,
                                                  num_return_sequences=1,
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id, 
                                                  repetition_penalty=1.1,                                            
                                                  **model_kwargs)
        else:
            #beam search
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  num_beams=num_beams,
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id,     
                                                  repetition_penalty=repetition_penalty,
                                                  **model_kwargs)            
            
        captions = []    
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)    
            captions.append(caption[len(self.prompt):])
        return captions
    

def blip_decoder(pretrained='',**kwargs):
    model = BLIP_Decoder(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        if len(msg.missing_keys) > 0:
            print(f"Missing keys: {msg.missing_keys}")
        #assert(len(msg.missing_keys)==0)
        print_trainable_parameters(model)
    return model    
    
def blip_feature_extractor(pretrained='',**kwargs):
    model = BLIP_Base(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        assert(len(msg.missing_keys)==0)
    return model        

def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    return tokenizer


def create_vit(vit, image_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0):
        
    assert vit in ['base', 'large'], "vit parameter must be base or large"
    if vit=='base':
        vision_width = 768
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=12, 
                                           num_heads=12, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0 or drop_path_rate
                                          )   
    elif vit=='large':
        vision_width = 1024
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=24, 
                                           num_heads=16, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0.1 or drop_path_rate
                                          )   
    return visual_encoder, vision_width

def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")

def load_checkpoint(model,url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu') 
    elif os.path.isfile(url_or_filename):        
        checkpoint = torch.load(url_or_filename, map_location='cpu') 
    else:
        raise RuntimeError('checkpoint url or path is invalid')
        
    state_dict = checkpoint['model']
    
    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder) 
    if 'visual_encoder_m.pos_embed' in model.state_dict().keys():
        state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                                         model.visual_encoder_m)    
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape!=model.state_dict()[key].shape:
                del state_dict[key]
    
    msg = model.load_state_dict(state_dict,strict=False)
    print('load checkpoint from %s'%url_or_filename)  
    return model,msg
    
def print_trainable_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Trainable: {name}")
        else:
            print(f"Frozen: {name}")