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
from transformers import BertTokenizer, AdapterConfig, BitsAndBytesConfig, pipeline
from sentence_transformers import SentenceTransformer, util

import torch
from torch import nn
import torch.nn.functional as F

import os
from urllib.parse import urlparse
from timm.models.hub import download_cached_file
import spacy
from nltk.corpus import wordnet as wn

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
        
        
        
class BLIP_Decoder(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 384,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 prompt = 'a picture of ',
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

        self.summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")
        
        # Freeze BLIP model weights
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
        for param in self.text_decoder.parameters():
            param.requires_grad = False

        adapter_config = AdapterConfig.load("pfeiffer")
        self.text_decoder.add_adapter("student_adapter", config=adapter_config)
        self.text_decoder.train_adapter("student_adapter")

        self.log_vars = nn.Parameter(torch.zeros(4))
  
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
        loss_lm = decoder_output.loss

        teacher_caption = self.convert_image_to_text(image)
        
        # Generate caption from current model
        student_caption = self.generate(image)
        
        # Compute semantic similarity
        semantic_similarity_loss = self.compute_semantic_similarity_loss(student_caption, teacher_caption)
        
        # Compute F1 score for object matches
        f1_score_loss = self.compute_f1_score_loss(student_caption, teacher_caption)
        
        # Compute KL divergence loss
        kl_loss = self.compute_kl_loss(student_caption, teacher_caption)
        
        loss_lm_weighted = 0.5 * torch.exp(-self.log_vars[0]) * loss_lm + 0.5 * self.log_vars[0]
        semantic_similarity_loss_weighted = 0.5 * torch.exp(-self.log_vars[1]) * semantic_similarity_loss + 0.5 * self.log_vars[1]
        f1_score_loss_weighted = 0.5 * torch.exp(-self.log_vars[2]) * f1_score_loss + 0.5 * self.log_vars[2]
        kl_loss_weighted = 0.5 * torch.exp(-self.log_vars[3]) * kl_loss + 0.5 * self.log_vars[3]

        # Total loss with weighted components
        total_loss = loss_lm_weighted + semantic_similarity_loss_weighted + f1_score_loss_weighted + kl_loss_weighted
        
        # Regularization term to prevent log_vars from growing too large
        reg_loss = 0.01 * torch.sum(self.log_vars ** 2)
        
        total_loss = total_loss + reg_loss
        
        return total_loss
    
    def convert_image_to_text(self, image):
        prompt = "USER: <image>\nPlease describe this image.\nASSISTANT:"
        outputs = self.pipe_image_to_text(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
        response = outputs[0]["generated_text"]
        # Extract text after "ASSISTANT:"
        assistant_index = response.find("ASSISTANT:")
        if assistant_index != -1:
            response = response[assistant_index + len("ASSISTANT:"):].strip()
        return response
    
    def compute_semantic_similarity_loss(self, student_caption, teacher_caption):
        student_embeddings = self.semantic_model.encode(student_caption, convert_to_tensor=True)
        teacher_embeddings = self.semantic_model.encode(teacher_caption, convert_to_tensor=True)

        semantic_similarity_loss = 1 - util.pytorch_cos_sim(student_embeddings, teacher_embeddings).item()
        return semantic_similarity_loss
    
    def summarize(self, caption):
        summary = self.summarizer(caption, max_length=50, do_sample=False)
        summarized = summary[0]['summary_text']
        return summarized
    
    def extract_objects_features(self, text):
        doc = nlp(text)
        objects_features = set()
        for token in doc:
            if token.pos_ in {'NOUN'}:
                objects_features.add(token.lemma_.lower())
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

    def calculate_precision_recall_f1(set1, set2, matched_objects):
        true_positives = len(matched_objects)
        possible_positives = len(set1)
        predicted_positives = len(set2)

        precision = true_positives / predicted_positives if predicted_positives > 0 else 0
        recall = true_positives / possible_positives if possible_positives > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return precision, recall, f1

    def compute_f1_score_loss(self, student_caption, teacher_caption):
        teacher_caption = self.summarize(teacher_caption)
        objects_features_student = self.extract_objects_features(student_caption)
        objects_features_teacher = self.extract_objects_features(teacher_caption)
        
        synonym_dict_teacher = self.create_synonym_dict(objects_features_teacher)
        
        matched_objects = self.match_with_synonyms(objects_features_student, synonym_dict_teacher)
        
        _, _, f1_score = self.calculate_precision_recall_f1(objects_features_student, objects_features_teacher, matched_objects)
        f1_score_loss = 1 - f1_score  # Loss should decrease as F1 score increases
        return f1_score_loss

    def generate(self, image, sample=False, num_beams=3, max_length=30, min_length=10, top_p=0.9, repetition_penalty=1.0):
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
        assert(len(msg.missing_keys)==0)
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
    
