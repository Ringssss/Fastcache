from fastcache_paths import ensure_sys_paths, CKPT_DIR, DATASETS_DIR, RESULTS_DIR

ensure_sys_paths()

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, AutoProcessor, LlavaForConditionalGeneration, Cache, DynamicCache
from transformers.generation import (
    SampleDecoderOnlyOutput,
    GreedySearchDecoderOnlyOutput
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from typing import List, Dict, Tuple
import json
import random
import math
from PIL import Image
import os
import argparse
import copy

from utils_ccm.module_ccm_nohs import *
# from utils_ccm.module_ccm import *




"""
v0.0113
支持半精度

v1.4
尝试同样程序下的text head训练

v1.5
试着解决显存爆炸问题

v1.7 加入minicpm支持

"""

# 保存检查点的函数
def save_checkpoint(epoch, model, optimizer, scheduler, loss, is_best, filefoldname='ckpt_store', filename='checkpoint.pth'):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': {
            'compressor': model.state_dict(),
        },
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }
    # 保存最新的检查点
    torch.save(checkpoint, f'{filefoldname}/last_{filename}')
    print(f"Saved checkpoint at epoch {epoch}")

    # 如果是最佳模型,再单独保存一份
    if is_best:
        best_filename = f'{filefoldname}/best_{filename}'
        torch.save(checkpoint, best_filename)
        print(f"Saved best model checkpoint at epoch {epoch}")


def load_checkpoint(model, optimizer, scheduler, filename='checkpoint.pth'):
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)

        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict']['compressor'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        loss = checkpoint['loss']

        print(f"Loaded checkpoint '{filename}' (epoch {start_epoch})")
        return start_epoch, loss
    else:
        print(f"No checkpoint found at '{filename}'")
        return 0, float('inf')

def load_checkpoint_only_model(model, filename='checkpoint.pth'):
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)

        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict']['compressor'])
        loss = checkpoint['loss']

        print(f"Loaded checkpoint '{filename}' (epoch {start_epoch}, loss {loss})")
        return start_epoch, loss
    else:
        print(f"No checkpoint found at '{filename}'")
        return 0, float('inf')


def load_pretrained_encoder(encoder_path, input_dim, hidden_dim, num_layers, kv_dim, device):
    encoder = MAEEncoder(input_dim, hidden_dim, num_layers, kv_dim)
    checkpoint = torch.load(encoder_path)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder.to(device)
    return encoder

def load_finetune_decoder(decoder_path, input_dim, hidden_dim, output_dim, compression_ratio, num_layers, num_heads, num_llm_layers, device):
    decoder = CompressedDecoder(input_dim, hidden_dim, output_dim, compression_ratio, num_layers, num_heads, num_llm_layers)
    checkpoint = torch.load(decoder_path)
    decoder.load_state_dict(checkpoint['decoder'])
    decoder.to(device)
    return decoder


def load_finetune_compressor(compressor_path, num_llm_layers, output_dim, num_llm_head, device):
    compressor = KVCacheLinearCompressor(num_llm_layers, output_dim, num_llm_head, compression_factor=2, min_seq_len=64)
    checkpoint = torch.load(compressor_path)
    compressor.load_state_dict(checkpoint['compressor'])
    compressor.to(device)
    return compressor


def load_finetune_cross_compressor(compressor_path, model_config, dmodel_config, device, compression_factor=1, mode=2):
    # compressor = KVCacheLinearCompressor(num_llm_layers, output_dim, num_llm_head, compression_factor=2, min_seq_len=64)
    # compressor = CrossModelKVCompressor(model_config, dmodel_config, compression_factor=compression_factor, device=device)
    if mode == 1:
        compressor = CrossModelKVCompressor(model_config, dmodel_config, compression_factor=compression_factor,
                                                device=device)
    elif mode == 2:
        compressor = CrossModelRealKVCompressor(model_config, dmodel_config, compression_factor=compression_factor, device=device)
    checkpoint = torch.load(compressor_path)
    compressor.load_state_dict(checkpoint['compressor'])
    compressor.to(device)
    return compressor

def load_data(json_path: str, num_samples: int) -> List[str]:
    with open(json_path, 'r') as f:
        data = json.load(f)
    texts = [' '.join(d['conversations'][0]['value'].split()) for d in data if d['conversations'] and d['conversations'][0]["from"] == "human"]
    # texts = [' '.join(d['conversations'][0]['value'].split()) for d in data if d['conversations']]
    random.shuffle(texts)
    if num_samples < 0:
        target_texts = texts
    else:
        target_texts = texts[:num_samples]
    return target_texts


def parse_string_with_image(text, special_token="<image>"):
    is_exist_flag = 0
    # 检查是否包含<image>标记
    if special_token in text:
        # 提取<image>标记
        image_part = special_token
        # 移除<image>标记并拼接剩余文本
        remaining_text = text.replace(special_token, '')
        remaining_text = remaining_text.strip()
        is_exist_flag = 1
    else:
        remaining_text = text

    return remaining_text, is_exist_flag

def load_image_data_minicpm(json_path: str, imgsets_path: str, num_samples: int, mode=1) -> List[str]:
    target_datasets = ['gqa']
    with open(json_path, 'r') as f:
        data = json.load(f)
    data = [d for d in data if 'image' in d.keys() and d['image'].split('/')[0] in target_datasets]
    random.shuffle(data)
    if num_samples < 0:
        target_data = data
    else:
        target_data = data[:num_samples]
    target_texts_image = []
    if mode == 0:
        # texts_image = [' '.join(d['conversations'][0]['value'].split()) for d in data
        # target_texts_image = [(f"USER: {d['conversations'][0]['value']} ASSISTANT:", Image.open(f"{imgsets_path}{d['image']}").convert("RGB"))
        target_texts_image = [ 
            [
                [{"role": "user", 
                    "content": 
                        "(<image>./</image>)\nWhat's the content of the image? Please describe every detail in the image with specific and detailed language."
                    
                }],
                Image.open(f"{imgsets_path}{d['image']}").convert("RGB")
            ]
                       for d in target_data if (d['conversations']  and d['conversations'][0]["from"] == "human")]
    elif mode == 1: # gqa多轮对话
        for d_idx, d in enumerate(target_data):
            questions_list = []
            answers_list = []
            if (d['conversations'] and d['conversations'][0]["from"] == "human"):
                for d_sentence in d['conversations']:
                    if d_sentence['from'] == 'human':
                        remaining_text, image_flag = parse_string_with_image(d_sentence['value'], '<image>')
                        if image_flag == 1:
                            questions_list.append(
                                [
                                    [{"role": "user", 
                                        "content": 
                                            "(<image>./</image>)\nWhat's the content of the image? Please describe every detail in the image with specific and detailed language."
                                        
                                    }],
                                    Image.open(f"{imgsets_path}{d['image']}").convert("RGB")
                                ]
                            )
                            answers_list.append(1)
                        questions_list.append(remaining_text)
                    else:
                        answers_list.append(d_sentence['value'])
            target_texts_image.append([questions_list, answers_list])
    
    return target_texts_image

def load_image_data(json_path: str, imgsets_path: str, num_samples: int, mode=1) -> List[str]:
    target_datasets = ['gqa']
    with open(json_path, 'r') as f:
        data = json.load(f)
    data = [d for d in data if 'image' in d.keys() and d['image'].split('/')[0] in target_datasets]
    random.shuffle(data)
    if num_samples < 0:
        target_data = data
    else:
        target_data = data[:num_samples]
    target_texts_image = []
    if mode == 0:
        # texts_image = [' '.join(d['conversations'][0]['value'].split()) for d in data
        # target_texts_image = [(f"USER: {d['conversations'][0]['value']} ASSISTANT:", Image.open(f"{imgsets_path}{d['image']}").convert("RGB"))
        target_texts_image = [(f"USER: <image> \nWhat's the content of the image? "
                               f"Please describe every detail in the image with specific and detailed language. ASSISTANT:", Image.open(f"{imgsets_path}{d['image']}").convert("RGB"))
                       for d in target_data if (d['conversations']  and d['conversations'][0]["from"] == "human")]
    elif mode == 1: # gqa多轮对话
        for d_idx, d in enumerate(target_data):
            questions_list = []
            answers_list = []
            if (d['conversations'] and d['conversations'][0]["from"] == "human"):
                for d_sentence in d['conversations']:
                    if d_sentence['from'] == 'human':
                        remaining_text, image_flag = parse_string_with_image(d_sentence['value'], '<image>')
                        if image_flag == 1:
                            questions_list.append((f"USER: <image> \nWhat's the content of the image? "
                                   f"Please describe every detail in the image with specific and detailed language. ASSISTANT:", Image.open(f"{imgsets_path}{d['image']}").convert("RGB")))
                            answers_list.append(1)
                        questions_list.append(remaining_text)
                    else:
                        answers_list.append(d_sentence['value'])
            target_texts_image.append([questions_list, answers_list])

    return target_texts_image

def load_text_data(json_path: str, num_samples: int) -> List[str]:
    with open(json_path, 'r') as f:
        data = json.load(f)
    texts = [' '.join(d['conversations'][0]['value'].split()) for d in data if d['conversations'] and d['conversations'][0]["from"] == "human"]
    random.shuffle(texts)
    if num_samples < 0:
        target_texts = texts
    else:
        target_texts = texts[:num_samples]
    return target_texts


# def get_selected_queries(query_states, generated_len):
def get_selected_positions(pd_len, num_prompt_end_tokens, num_random_tokens, device):
    """选择要对比的token positions"""
    # 1. 生成部分的所有token位置
    # num_prompt_end_tokens = 5
    # num_random_tokens = 10

    prompt_len, decoding_len = pd_len
    generated_len = decoding_len
    seq_len = prompt_len + decoding_len
    generate_positions = torch.arange(prompt_len, seq_len, device=device)

    # 2. prompt末尾的tokens位置
    prompt_end_positions = torch.arange(
        seq_len - generated_len - num_prompt_end_tokens,
        seq_len - generated_len,
        device=device
    )

    # 3. prompt前面的随机token位置
    prompt_random_len = seq_len - generated_len - num_prompt_end_tokens
    if prompt_random_len > 0:
        random_positions = torch.randperm(prompt_len, device=device)[:num_random_tokens]
    else:
        random_positions = torch.tensor([], device=device, dtype=torch.long)

    return torch.cat([generate_positions, prompt_end_positions, random_positions]), \
           [generate_positions.shape[0], prompt_end_positions.shape[0], random_positions.shape[0]]


def split_kvcache(kvcache, pd_len, decode_split_len=0):
    (prompt_len, decoding_len) = pd_len
    split_list = [prompt_len+decode_split_len, decoding_len-decode_split_len]
    target_len = split_list[0]
    # if isinstance(kvcache, (type(None), Cache)):
    #     # 情况1: kvcache 是 None
    #     if kvcache is None:
    #         return None, split_list

    #     # 情况2: kvcache 是 Cache 对象 (StaticCache, DynamicCache等)
    #     # 创建一个深拷贝，以防修改原始的kvcache对象
    #     new_cache = copy.deepcopy(kvcache)

    #     # 对于DynamicCache，我们可以直接截断张量来节省内存
    #     if isinstance(new_cache, DynamicCache):
    #         new_cache.key_cache = [k[:, :, :target_len, :] for k in new_cache.key_cache]
    #         new_cache.value_cache = [v[:, :, :target_len, :] for v in new_cache.value_cache]

    #     # 对于所有Cache类型，都必须更新 seen_tokens 来反映其新的逻辑长度。
    #     # 对于 StaticCache，这是唯一安全的操作，因为它的张量是预分配的，不能改变形状。
    #     new_cache.seen_tokens = target_len
        
    #     return new_cache, split_list
    # else:
        # tuple cache
    def process_inner_tuple(inner_tuple, split_list):
        # 处理内部元组
        # return tuple(torch.split(tensor.detach(), split_list, dim=2)[0] for tensor in inner_tuple)
        return tuple(torch.split(tensor, split_list, dim=2)[0] for tensor in inner_tuple)

    # 处理外部元组
    kvcache_ilist = tuple(process_inner_tuple(inner_tuple, split_list) for inner_tuple in kvcache)

    return kvcache_ilist, split_list
    # return split_kvcaches[0], split_kvcaches[1], split_list

def proc_original_input(original_outputs, input_ids, pd_len):
    (prompt_len, decoding_len) = pd_len
    # original_input_ids = original_outputs.logits.argmax(dim=2)
    original_input_ids = original_outputs.sequences
    original_out_logits = torch.softmax(torch.stack(original_outputs.scores, dim=1), dim=-1)
    # original_out_logits = logits_to_onehot(torch.stack(original_outputs.scores, dim=1))
    batch_size, prompt_len = input_ids.shape
    # decoding_len = original_input_ids.shape[1] - prompt_len
    decoding_tensor = original_input_ids[:, prompt_len-1:]
    predict_num = 0

    # attn_mask
    # predict_idx_list = []
    # min_len = decoding_len
    # for i in range(batch_size):
    #     valid_output_num = torch.argmax((decoding_tensor == 0).int()[i])
    #     if valid_output_num == 0:
    #         valid_output_num = decoding_len
    #     predict_idx_list.append(int(valid_output_num))
    #     if min_len > valid_output_num and valid_output_num != 0:
    #         min_len = valid_output_num
    #     # cache_num = valid_output_num // 2
    # predict_num = min_len // 2
    # predict_idx_list.append(original_input_ids[i,predict_num:])
    compress_input = decoding_tensor[:, predict_num:]
    compress_input_logits = original_out_logits[:, prompt_len-1+predict_num:]
    # compress_mask = torch.zeros_like(compress_input)
    # for i_idx, i_num in enumerate(predict_idx_list):
    #     compress_mask[i_idx,:(i_num-predict_num)] = 1
    return compress_input, compress_input_logits, predict_num

def padding_index(input_idx, num_pad=0):
    return torch.concat((torch.zeros((input_idx.shape[0],num_pad),device=input_idx.device,dtype=input_idx.dtype),input_idx[:, :1]),dim=-1)

def padding_mask(attention_mask, num_pad=0):
    return torch.concat((torch.ones((attention_mask.shape[0],num_pad),device=attention_mask.device,dtype=attention_mask.dtype),attention_mask),dim=-1)

def clean_kv_cache_for_padding(kv_cache, attention_mask):
    """
    清理 KV cache 中 padding 位置的缓存
    Args:
        kv_cache: tuple(tuple(k_cache, v_cache))
            k_cache: [batch_size, num_heads, seq_len, head_dim]
            v_cache: [batch_size, num_heads, seq_len, head_dim]
        attention_mask: [batch_size, seq_len], 1表示非padding, 0表示padding
    """
    batch_size, seq_len = attention_mask.shape

    # 扩展 attention mask 维度以匹配 cache
    # [batch_size, seq_len] -> [batch_size, 1, seq_len, 1]
    padding_mask = attention_mask.unsqueeze(1).unsqueeze(-1)

    # 遍历每一层的 KV cache
    cleaned_kv_cache = []
    for layer_cache in kv_cache:
        k_cache, v_cache = layer_cache

        # 使用广播机制将 padding 位置置 0
        # padding_mask 会广播到 [batch_size, num_heads, seq_len, head_dim]
        k_cache = k_cache * padding_mask
        v_cache = v_cache * padding_mask

        cleaned_kv_cache.append((k_cache, v_cache))

    return tuple(cleaned_kv_cache)


def generate_with_grad(llm, input_ids, attention_mask, max_new_tokens, pixel_values=None,
                          output_hidden_states=True, past_key_values=None,
                          do_sample=True, temperature=1., output_attentions=False,
                          return_dict_in_generate=False, output_scores=False):
    batch_size = input_ids.shape[0]
    cur_len = input_ids.shape[1]
    outputs = []

    # 存储中间状态
    hidden_states = []
    attentions = []
    scores = []
    # past_key_values = None
    is_prefill = True

    for _ in range(max_new_tokens):
        # 前向传播
        model_outputs = llm.model(
            input_ids=input_ids[:, -1:] if not is_prefill else input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values ,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions
        )

        # 保存状态
        if output_hidden_states:
            hidden_states.append(model_outputs.hidden_states)
        if output_attentions:
            attentions.append(model_outputs.attentions)
        if output_scores:
            scores.append(model_outputs.logits)
        past_key_values = model_outputs.past_key_values

        # 获取下一个token的概率
        next_token_logits = model_outputs.logits[:, -1, :]

        # 采样或贪婪解码
        if do_sample:
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
        else:
            next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        outputs.append(next_tokens)

        # 更新输入
        input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        attention_mask = torch.cat([
            attention_mask,
            attention_mask.new_ones((batch_size, 1))
        ], dim=-1)

        # 检查是否生成结束符号
        if next_tokens[0].item() == llm.processor.tokenizer.eos_token_id:
            break

    # 整合所有输出
    generated_tokens = torch.cat([input_ids, outputs], dim=-1)

    attentions_ready = None
    hidden_states_ready = None
    scores_ready = None
    if output_attentions:
        attentions_ready = attentions
    if output_hidden_states:
        hidden_states_ready = hidden_states
    if output_scores:
        scores_ready = scores

    if do_sample:
        return SampleDecoderOnlyOutput(
            sequences=generated_tokens,
            scores=scores_ready,  # 如果需要可以存储
            hidden_states=hidden_states_ready,
            attentions=attentions_ready,
            past_key_values=past_key_values
        )
    else:
        GreedySearchDecoderOnlyOutput(
            sequences=generated_tokens,
            scores=scores_ready,  # 如果需要可以存储
            hidden_states=hidden_states_ready,
            attentions=attentions_ready,
            past_key_values=past_key_values
        )


def finetune_com(llm, train_data, val_data, compressor_load_path, compressor_save_path, num_epochs, learning_rate, device, compression_ratio, prompt_len, is_continue):
    output_tokens = 512
    batch_size = 1
    patch_size = 14.
    # num_end_tokens = 5
    # num_random_tokens = 300
    num_end_tokens = 0
    num_random_tokens = 0
    pd_len = [prompt_len, output_tokens]
    learning_rate_de = learning_rate
    # processor.tokenizer.added_tokens_encoder
    # output_dim = 128
    # output_dim = llm.model.config.hidden_size
    # head_dim = output_dim // num_heads
    # label_strategy = "logit"
    # label_strategy = "label"
    label_strategy = "label_twice"
    compressor_path = compressor_load_path
    # compressor_path = "./ckpt_store/best_finetune_mlp_1030_mm_8.pth"
    llm.model.eval()

    # decoder = CompressedDecoder_OnlyMLP(input_dim, hidden_dim, output_dim, compression_ratio, num_layers, num_heads).to(device)
    # decoder = CompressedDecoder(input_dim, hidden_dim, output_dim, compression_ratio, num_decoder_layers, num_heads, num_llm_layers).to(device)

    if llm.model_mode == 'minicpm':
        compressor = KVCacheHybridCompressor(llm.model.config, compression_factor=compression_ratio, min_seq_len=2).to(device)
    else:
        compressor = KVCacheLinearDecoupleCompressor(llm.model.config, compression_factor=compression_ratio, min_seq_len=2).to(device)
    if llm.torch_dtype == torch.float16:
        compressor = compressor.half()
        print(111)

    # compressor.train()
    # compressor = KVCacheLinearCompressor(llm.model.config, compression_factor=5, min_seq_len=64).to(device)
    # compressor, start_epoch, loss_d = load_finetune_compressor_with_config(compressor_path, llm.model.config, device, compression_factor=5)
    # decoder_load_path =
    # './val28.71-best_model_finetune.pth'
    # decoder = load_finetune_decoder(decoder_load_path, input_dim, hidden_dim, output_dim, compression_ratio, num_layers, num_heads, device)

    loss_fn = KVCacheCompressionLoss(alpha=0.5, beta=0.3, label_strategy=label_strategy)

    # # 冻结 encoder 参数
    # for param in pretrained_encoder.parameters():
    #     param.requires_grad = False

    # optimizer_en = optim.Adam(decoder.parameters(), lr=learning_rate_en)
    optimizer_de = optim.Adam(compressor.parameters(), lr=learning_rate_de)
    # optimizer_de = optim.Adam(compressor.parameters(), lr=learning_rate_de, weight_decay=1e-5)
    # optimizer_de = optim.Adam(compressor.parameters(), lr=learning_rate_de, weight_decay=0.3)
    # 一直训不起来的原因是weight_decay=0.3
    # scheduler_en = ReduceLROnPlateau(optimizer_en, mode='min', factor=0.5, patience=2, verbose=True)
    scheduler_de = ReduceLROnPlateau(optimizer_de, mode='min', factor=0.5, patience=2, verbose=True)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    best_val_loss = float('inf')
    epoch = 0
    if is_continue:
        epoch, best_val_loss = load_checkpoint(compressor, optimizer_de, scheduler_de, compressor_path)
        epoch += 1
    while (True):
        if epoch >= num_epochs:
            break
        # pretrained_encoder.train()
        # decoder.train()
        compressor.train()
        total_loss = 0
        total_acc = 0
        total_acc2 = 0
        i_batch = 0

        for multi_dialog_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            is_preview = True

            # multi_dialog_batch process
            for i_dialog, dialog_batch in enumerate(multi_dialog_batch):
                if is_preview:
                    input_ids = dialog_batch['input_ids'].to(device)
                    attention_mask = dialog_batch['attention_mask'].to(device)
                    pixel_values = dialog_batch['pixel_values'].to(device)
                    answer_idx = None
                else:
                    input_ids = dialog_batch['input_ids'].to(device)
                    attention_mask = dialog_batch['attention_mask'].to(device)
                    pixel_values = None
                    answer_idx = dialog_batch['answer'].to(device)

                if is_preview: # preview
                    is_preview = False
                    with torch.no_grad():

                        original_outputs = llm.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                              pixel_values=pixel_values, output_hidden_states=True,
                                              max_new_tokens=output_tokens,
                                              do_sample=True, temperature=1., use_cache=True, return_dict_in_generate=True,
                                              output_scores=True, output_attentions=True)
                        original_kv_cache = original_outputs.past_key_values
                        original_h_states = original_outputs.hidden_states
                        pd_len[0] = round(pixel_values.shape[2]/patch_size)*round(pixel_values.shape[3]/patch_size) + input_ids.shape[-1] - 2
                        pd_len[1] = original_kv_cache[0][0].shape[2] - pd_len[0]
                        compress_input, compress_input_logits, decode_split_len = proc_original_input(original_outputs, input_ids, pd_len)
                        split_cache_l, split_list = split_kvcache(original_kv_cache, pd_len, decode_split_len)
                    it_len = [0,0]
                    it_len[1] = input_ids[:,torch.where(input_ids==32000)[1]+1:].shape[-1]
                    it_len[0] = split_cache_l[0][0].shape[2]-it_len[1]
                    compressed_kv_cache = compressor(split_cache_l, it_len, forward_mode='train')
                    hidden_embeds = loss_fn.cvt_hidden_states(original_h_states)[0]
                    selected_positions, pos_len = get_selected_positions(pd_len, num_end_tokens, num_random_tokens, device)
                    selected_embeds = hidden_embeds[:,selected_positions,:]
                    compressed_attention_mask = torch.ones((selected_embeds.shape[0], selected_embeds.shape[1]+compressed_kv_cache[0][0].shape[2]),device=selected_embeds.device)

                    compressed_kv_cache = CompressibleCache.from_legacy_cache(compressed_kv_cache)
                    compressed_outputs = llm.model(inputs_embeds=selected_embeds,
                                                   attention_mask=compressed_attention_mask,
                                                   past_key_values=compressed_kv_cache,
                                                   output_attentions=True, output_hidden_states=True)
                    preview_len = (compressed_kv_cache[0][0].shape[2], pos_len[0]) #image_cache_len, decoding_cache_len
                    # std_compress_attn = 0.5 * (compressed_outputs.attentions[2].mean(dim=1) + compressed_outputs.attentions[2].max(dim=1)[0])
                    preview_kvcache, _ = split_kvcache(compressed_outputs.past_key_values, (preview_len[0]+preview_len[1],pos_len[1]+pos_len[2]))
                    full_cache = original_kv_cache
                    std_compress_hs = compressed_outputs.hidden_states
                    with torch.no_grad():
                        original_twice_attention_mask = torch.ones(
                            (selected_embeds.shape[0], selected_embeds.shape[1] + split_cache_l[0][0].shape[2]),
                            device=selected_embeds.device)
                        original_twice_outputs = llm.model(inputs_embeds=selected_embeds,
                                                       attention_mask=original_twice_attention_mask,
                                                       past_key_values=split_cache_l,
                                                   output_attentions=True, output_hidden_states=True)
                        # std_original_attn = 0.5 * (original_twice_outputs.attentions[2].mean(dim=1))
                        std_original_hs = original_twice_outputs.hidden_states
                        # todo delete
                        compressed_gen_attention_mask = compressed_attention_mask[:, :preview_len[0]+1]
                        input_ids_comp_generate = torch.ones_like(compressed_gen_attention_mask).type(torch.int64)
                        input_ids_comp_generate[:, -input_ids.shape[1]:] = input_ids
                        compressed_generate_outputs = llm.model.generate(input_ids=input_ids_comp_generate,
                                           attention_mask=compressed_gen_attention_mask,
                                           past_key_values=compressed_kv_cache,do_sample=True, temperature=0.7,
                                       return_dict_in_generate=True, use_cache=True, max_new_tokens=output_tokens)

                        original_twice_gen_attention_mask = original_twice_attention_mask[:, :pd_len[0] + 1]
                        input_ids_generate = torch.ones_like(original_twice_gen_attention_mask).type(torch.int64)
                        input_ids_generate[:, -input_ids.shape[1]:] = input_ids
                        original_twice_generate_outputs = llm.model.generate(input_ids=input_ids_generate,
                                                                             attention_mask=original_twice_gen_attention_mask,
                                                                             past_key_values=split_cache_l,
                                                                             do_sample=True, temperature=1.,
                                                                             return_dict_in_generate=True,
                                                                             use_cache=True,
                                                                             max_new_tokens=output_tokens)
                        print(llm.processor.batch_decode(original_twice_generate_outputs.sequences[:, input_ids_generate.shape[1]-1:]),
                              llm.processor.batch_decode(compressed_generate_outputs.sequences[:, input_ids_comp_generate.shape[1]-1:]))

                    compress_output_logits = torch.softmax(compressed_outputs.logits, dim=-1)
                    original_twice_outputs_logits = torch.softmax(original_twice_outputs.logits, dim=-1)
                    loss = loss_fn.forward_attn(original_twice_outputs_logits,
                                                compress_output_logits , pos_len,
                                                std_original_hs, std_compress_hs)
                    acc1, acc2, perplexity, mid_idx, output_idx = loss_fn.calc_metrics3(original_twice_outputs_logits[:, :(pos_len[0]+pos_len[1]), :], compress_output_logits[:, :(pos_len[0]+pos_len[1]), :])
                    is_preview = False
                    optimizer_de.zero_grad()
                    loss.backward()
                    optimizer_de.step()
                    total_loss += loss.item()
                    total_acc += acc1
                    total_acc2 += acc2

                    i_batch += input_ids.shape[0] # add shape batchsize

                    avg_mini_train_loss = total_loss / i_batch
                    avg_mini_train_acc = total_acc / i_batch
                    avg_mini_train_acc2 = total_acc2 / i_batch
                    print(f"minibatch Train Loss: {avg_mini_train_loss:.4f}, cur_loss: {loss.item():.4f}, "
                          f"avg_acc : {avg_mini_train_acc * 100:.2f}%, acc : {acc1 * 100:.2f}%, "
                          f"avg_ff : {avg_mini_train_acc2:.2f}, ff: {acc2:.2f}, perplexity:{perplexity:.2f}, "
                          f"{compressed_kv_cache[0][0].shape[2]}")
                else: # exam
                    with torch.no_grad():

                        attention_mask_padding = padding_mask(attention_mask, preview_kvcache[0][0].shape[2])
                        original_outputs = llm.model.generate(input_ids=input_ids, attention_mask=attention_mask_padding,
                                                              output_hidden_states=True, max_new_tokens=output_tokens,
                                                              do_sample=True, temperature=1., past_key_values=preview_kvcache,
                                                              return_dict_in_generate=True,
                                                              output_scores=True)
                        # original_outputs_gradgen = generate_with_grad(llm, input_ids=input_ids, attention_mask=attention_mask_padding,
                        #                                       output_hidden_states=True, max_new_tokens=output_tokens,
                        #                                       do_sample=True, temperature=1., past_key_values=preview_kvcache,
                        #                                       return_dict_in_generate=True,
                        #                                       output_scores=True)

                        # attention_mask_padding_full = padding_mask(attention_mask, full_cache[0][0].shape[2])
                        # original_full_outputs = llm.model.generate(input_ids=input_ids,
                        #                                       attention_mask=attention_mask_padding_full,
                        #                                       output_hidden_states=True, max_new_tokens=output_tokens,
                        #                                       do_sample=True, temperature=1.,
                        #                                       past_key_values=full_cache,
                        #                                       return_dict_in_generate=True,
                        #                                       output_scores=True, output_attentions=True)
                        original_kv_cache = original_outputs.past_key_values
                        original_h_states = original_outputs.hidden_states

                        pd_len[0] = preview_kvcache[0][0].shape[2] + input_ids.shape[-1] - 1

                        pd_len[1] = original_kv_cache[0][0].shape[2] - pd_len[0]
                        compress_input, compress_input_logits, decode_split_len = proc_original_input(original_outputs, input_ids, pd_len)
                        split_cache_l, split_list = split_kvcache(original_kv_cache, pd_len, decode_split_len)
                    it_len = [0, 0]
                    it_len[1] = input_ids[:, torch.where(input_ids == 32000)[1] + 1:].shape[-1]
                    it_len[0] = split_cache_l[0][0].shape[2] - it_len[1]
                    compressed_kv_cache = compressor(split_cache_l, it_len, forward_mode='train')
                    hidden_embeds = loss_fn.cvt_hidden_states(original_h_states)[0]
                    selected_positions, pos_len = get_selected_positions(pd_len, num_end_tokens, num_random_tokens,
                                                                         device)
                    selected_embeds = hidden_embeds[:, selected_positions, :]
                    compressed_attention_mask = torch.ones(
                        (selected_embeds.shape[0], selected_embeds.shape[1] + compressed_kv_cache[0][0].shape[2]),
                        device=selected_embeds.device)
                    compressed_outputs = llm.model(inputs_embeds=selected_embeds,
                                                   attention_mask=compressed_attention_mask,
                                                   past_key_values=compressed_kv_cache,
                                                   output_attentions=True, output_hidden_states=True)
                    # std_compress_attn = 0.5 * (compressed_outputs.attentions[2].mean(dim=1) + compressed_outputs.attentions[2].max(dim=1)[0])
                    std_compress_hs = compressed_outputs.hidden_states
                    with torch.no_grad():
                        original_twice_attention_mask = torch.ones(
                            (selected_embeds.shape[0], selected_embeds.shape[1] + split_cache_l[0][0].shape[2]),
                            device=selected_embeds.device)
                        original_twice_outputs = llm.model(inputs_embeds=selected_embeds,
                                                           attention_mask=original_twice_attention_mask,
                                                           past_key_values=split_cache_l,
                                                           output_attentions=True, output_hidden_states=True)
                        # std_original_attn = 0.5 * (original_twice_outputs.attentions[2].mean(dim=1))
                        std_original_hs = original_twice_outputs.hidden_states
                    compress_output_logits = torch.softmax(compressed_outputs.logits, dim=-1)
                    original_twice_outputs_logits = torch.softmax(original_twice_outputs.logits, dim=-1)
                    loss = loss_fn.forward_attn(original_twice_outputs_logits,
                                                compress_output_logits, pos_len,
                                                std_original_hs, std_compress_hs)

                    acc1, acc2, perplexity, mid_idx, output_idx = loss_fn.calc_metrics3(
                        original_twice_outputs_logits[:, (pos_len[0] + pos_len[1]):, :],
                        compress_output_logits[:, (pos_len[0] + pos_len[1]):, :])
                    is_preview = False
                    optimizer_de.zero_grad()
                    loss.backward()
                    optimizer_de.step()
                    total_loss += loss.item()
                    total_acc += acc1
                    total_acc2 += acc2

                    i_batch += input_ids.shape[0]  # add shape batchsize

                    avg_mini_train_loss = total_loss / i_batch
                    avg_mini_train_acc = total_acc / i_batch
                    avg_mini_train_acc2 = total_acc2 / i_batch
                    print(f"minibatch Train Loss: {avg_mini_train_loss:.4f}, cur_loss: {loss.item():.4f}, "
                          f"avg_acc : {avg_mini_train_acc * 100:.2f}%, acc : {acc1 * 100:.2f}%, "
                          f"avg_ff : {avg_mini_train_acc2:.2f}, ff: {acc2:.2f}, perplexity:{perplexity:.2f}, "
                          f"{compressed_kv_cache[0][0].shape[2]}")


        avg_train_loss = total_loss / len(train_loader)
        avg_train_acc = total_acc / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc : {avg_train_acc*100:.2f}%")

        # Validation
        val_loss, val_acc = validate_llava(compressor, loss_fn, llm, val_loader, output_tokens, pd_len, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

        scheduler_de.step(val_loss)

        # Save the best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        save_checkpoint(
            epoch=epoch,
            model=compressor,
            optimizer=optimizer_de,
            scheduler=scheduler_de,
            loss=val_loss,
            is_best=is_best,
            filefoldname=f'ckpt_store',
            filename=compressor_save_path,
            # filename=f'finetune_mlp_1030_mm_8.pth'  Val Loss: 0.3794, Val Acc: 87.76% nocor_position
            # filename=f'finetune_mlp_1030_mm_7.pth'  0.4222, Val Acc: 84.96% temperature
            # filename=f'finetune_mlp_1030_mm_6.pth' yes 78.59%
            # filename=f'finetune_mlp_1030_mm_5.pth' yes 5-69.84%
        )
        epoch += 1

    return compressor


def validate_llava(compressor, loss_fn, llm, val_loader, output_tokens, pd_len, device):
# def validate(pretrained_encoder, decoder, loss_fn, llm, val_loader, pd_len, device):
#     pretrained_encoder.eval()
#     decoder.eval()
    compressor.eval()
    total_loss = 0
    total_accuracy = 0
    patch_size = 14
    i_batch = 0

    with torch.no_grad():
        for multi_dialog_batch in tqdm(val_loader, desc=f"Validate"):
        # for multi_dialog_batch in val_loader:
            is_preview = True
            for i_dialog, dialog_batch in enumerate(multi_dialog_batch):
                if is_preview == True:
                    is_preview = False
                    input_ids = dialog_batch['input_ids'].to(device)
                    attention_mask = dialog_batch['attention_mask'].to(device)
                    pixel_values = dialog_batch['pixel_values'].to(device)

                    original_outputs = llm.model.generate(input_ids=input_ids, max_new_tokens=output_tokens,attention_mask=attention_mask,
                                                          pixel_values=pixel_values,
                                                          do_sample = True, temperature = 1., use_cache=True, return_dict_in_generate=True,
                                                          output_scores=True)
                    original_kv_cache = original_outputs.past_key_values

                    pd_len[0] = round(pixel_values.shape[2]/patch_size)*round(pixel_values.shape[3]/patch_size) + input_ids.shape[-1] - 2
                    pd_len[1] = original_kv_cache[0][0].shape[2] - pd_len[0]

                    compress_input, compress_input_logits, decode_split_len = proc_original_input(original_outputs, input_ids,pd_len)
                    split_cache_l, split_list = split_kvcache(original_kv_cache, pd_len, decode_split_len)
                    # encoded_kv_cache = pretrained_encoder(split_cache_l)
                    # compressed_kv_cache = decoder(encoded_kv_cache)

                    # compressed_kv_cache = compressor(split_cache_l)
                    it_len = [0, 0]
                    it_len[1] = input_ids[:, torch.where(input_ids == 32000)[1] + 1:].shape[-1]
                    it_len[0] = split_cache_l[0][0].shape[2] - it_len[1]
                    compressed_kv_cache = compressor(split_cache_l, it_len, forward_mode='val')
                    compressed_attention_mask = torch.ones(
                        (compress_input.shape[0], compress_input.shape[1] + compressed_kv_cache[0][0].shape[2]),
                        device=input_ids.device)
                    compressed_outputs = llm.model(input_ids=compress_input,
                                                   attention_mask=compressed_attention_mask,
                                                   past_key_values=compressed_kv_cache)
                    compress_output_logits = torch.softmax(compressed_outputs.logits, dim=-1)

                    # loss = loss_fn(compress_input_logits, compress_output_logits[:, :-1, :])
                    # cur_accuracy, _, _ = loss_fn.calc_metrics(compress_input_logits, compress_output_logits[:, :-1, :])
                    # loss = loss_fn(compress_input_logits, compress_output_logits[:, :-1, :],[compress_input_logits.shape[1],0,0])
                    loss = loss_fn(compress_input_logits, compress_output_logits[:, :-1, :])
                    cur_accuracy, _, _ = loss_fn.calc_metrics(compress_input_logits, compress_output_logits[:, :-1, :])
                    # 困惑度
                    # calc accuracy


                    # original_outputs = llm.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
                    # original_kv_cache = original_outputs.past_key_values
                    #
                    # encoded_kv_cache = pretrained_encoder(original_kv_cache)
                    # compressed_kv_cache = decoder(encoded_kv_cache)
                    #
                    # compressed_outputs = llm.model(input_ids=input_ids, attention_mask=attention_mask,
                    #                                past_key_values=compressed_kv_cache)
                    #
                    # loss = kv_cache_consistency_loss(original_outputs, compressed_outputs)
                    total_loss += loss.item()
                    total_accuracy += cur_accuracy
                else:
                    pass

                    # break

    return total_loss / len(val_loader),  total_accuracy / len(val_loader)



def finetune_com_minicpm(llm, train_data, val_data, compressor_load_path, compressor_save_path, num_epochs, learning_rate, device, compression_ratio, prompt_len, is_continue):
    output_tokens = 512
    batch_size = 1
    patch_size = 14.
    # num_end_tokens = 5
    # num_random_tokens = 300
    num_end_tokens = 0
    num_random_tokens = 0
    pd_len = [prompt_len, output_tokens]
    learning_rate_de = learning_rate
    # processor.tokenizer.added_tokens_encoder
    # output_dim = 128
    # output_dim = llm.model.config.hidden_size
    # head_dim = output_dim // num_heads
    # label_strategy = "logit"
    # label_strategy = "label"
    label_strategy = "label_twice"
    compressor_path = compressor_load_path
    # compressor_path = "./ckpt_store/best_finetune_mlp_1030_mm_8.pth"
    llm.model.eval()

    # decoder = CompressedDecoder_OnlyMLP(input_dim, hidden_dim, output_dim, compression_ratio, num_layers, num_heads).to(device)
    # decoder = CompressedDecoder(input_dim, hidden_dim, output_dim, compression_ratio, num_decoder_layers, num_heads, num_llm_layers).to(device)

    if llm.model_mode == 'minicpm':
        compressor = KVCacheHybridCompressor(llm.model.config, compression_factor=compression_ratio, min_seq_len=2).to(device)
    else:
        compressor = KVCacheLinearDecoupleCompressor(llm.model.config, compression_factor=compression_ratio, min_seq_len=2).to(device)
    if llm.torch_dtype == torch.float16:
        compressor = compressor.half()
    elif llm.torch_dtype == torch.bfloat16:
        compressor = compressor.bfloat16()

    # compressor.train()
    # compressor = KVCacheLinearCompressor(llm.model.config, compression_factor=5, min_seq_len=64).to(device)
    # compressor, start_epoch, loss_d = load_finetune_compressor_with_config(compressor_path, llm.model.config, device, compression_factor=5)
    # decoder_load_path =
    # './val28.71-best_model_finetune.pth'
    # decoder = load_finetune_decoder(decoder_load_path, input_dim, hidden_dim, output_dim, compression_ratio, num_layers, num_heads, device)

    loss_fn = KVCacheCompressionLoss(alpha=0.5, beta=0.3, label_strategy=label_strategy)

    # # 冻结 encoder 参数
    # for param in pretrained_encoder.parameters():
    #     param.requires_grad = False

    # optimizer_en = optim.Adam(decoder.parameters(), lr=learning_rate_en)
    optimizer_de = optim.Adam(compressor.parameters(), lr=learning_rate_de)
    # optimizer_de = optim.Adam(compressor.parameters(), lr=learning_rate_de, weight_decay=1e-5)
    # optimizer_de = optim.Adam(compressor.parameters(), lr=learning_rate_de, weight_decay=0.3)
    # 一直训不起来的原因是weight_decay=0.3
    # scheduler_en = ReduceLROnPlateau(optimizer_en, mode='min', factor=0.5, patience=2, verbose=True)
    # scheduler_de = ReduceLROnPlateau(optimizer_de, mode='min', factor=0.5, patience=2, verbose=True)
    scheduler_de = ReduceLROnPlateau(optimizer_de, mode='min', factor=0.5, patience=2)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    best_val_loss = float('inf')
    epoch = 0
    if is_continue:
        epoch, best_val_loss = load_checkpoint(compressor, optimizer_de, scheduler_de, compressor_path)
        epoch += 1
    while (True):
        if epoch >= num_epochs:
            break
        # pretrained_encoder.train()
        # decoder.train()
        compressor.train()
        total_loss = 0
        total_acc = 0
        total_acc2 = 0
        i_batch = 0

        for multi_dialog_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            is_preview = True

            # multi_dialog_batch process
            for i_dialog, dialog_batch in enumerate(multi_dialog_batch):
                if is_preview:
                    input_ids = dialog_batch['input_ids'].to(device)
                    attention_mask = dialog_batch['attention_mask'].to(device)
                    inputs_embeds = dialog_batch['inputs_embeds'].to(device)
                    # pixel_values = dialog_batch['pixel_values'].to(device)
                    # answer_idx = None
                else:
                    input_ids = dialog_batch['input_ids'].to(device)
                    attention_mask = dialog_batch['attention_mask'].to(device)
                    inputs_embeds = dialog_batch['inputs_embeds'].to(device)
                    # pixel_values = None
                    # answer_idx = dialog_batch['answer'].to(device)

                if is_preview: # preview
                    is_preview = False
                    with torch.no_grad():

                        original_outputs = llm.model.llm.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                              output_hidden_states=True,
                                              pad_token_id=0,
                                              eos_token_id=llm.terminators,
                                              max_new_tokens=output_tokens,
                                              do_sample=True, temperature=1., use_cache=True, return_dict_in_generate=True,
                                              output_scores=True, output_attentions=True)
                        original_kv_cache = original_outputs.past_key_values
                        original_h_states = original_outputs.hidden_states
                        # pd_len[0] = round(pixel_values.shape[2]/patch_size)*round(pixel_values.shape[3]/patch_size) + input_ids.shape[-1] - 2
                        pd_len[0] = attention_mask.shape[-1] #  prompt w padding
                        pd_len[1] = original_kv_cache[0][0].shape[2] - pd_len[0]
                        compress_input, compress_input_logits, decode_split_len = proc_original_input(original_outputs, input_ids, pd_len)
                        split_cache_l, split_list = split_kvcache(original_kv_cache, pd_len, decode_split_len)
                        split_cache_l_lens = split_cache_l[0][0].shape[2]
                        split_cache_l = CompressibleCache.from_legacy_cache(split_cache_l)
                        split_cache_l.set_seen_tokens(split_cache_l_lens)
                    it_len = [0,0] # image_Lens, text_Lens
                    # it_len[1] = input_ids[:,torch.where(input_ids==32000)[1]+1:].shape[-1]
                    # it_len[0] = split_cache_l[0][0].shape[2]-it_len[1]
                    it_len[1] = split_cache_l[0][0].shape[2]-it_len[0]
                    compressed_kv_cache = compressor(split_cache_l, it_len, forward_mode='train') # cache

                    hidden_embeds = loss_fn.cvt_hidden_states(original_h_states)[0]
                    selected_positions, pos_len = get_selected_positions(pd_len, num_end_tokens, num_random_tokens, device)
                    selected_embeds = hidden_embeds[:,selected_positions,:]
                    compressed_attention_mask = torch.ones((selected_embeds.shape[0], selected_embeds.shape[1]+compressed_kv_cache[0][0].shape[2]),device=selected_embeds.device)
                    

                    
                    # split_cache_l_backup = split_cache_l.clone()
                    # compressed_kv_cache = CompressibleCache.from_legacy_cache(compressed_kv_cache)
                    # compressed_kv_cache.set_seen_tokens(split_cache_l_lens)

                    # compressed_kv_cache_backup = compressed_kv_cache.clone()
                    compressed_cache_l_lens = compressed_kv_cache.get_actual_seq_length()

                    compressed_outputs = llm.model.llm(inputs_embeds=selected_embeds,
                                                   attention_mask=compressed_attention_mask,
                                                   past_key_values=compressed_kv_cache,
                                                   output_attentions=True, output_hidden_states=True)
                    preview_len = (compressed_cache_l_lens, pos_len[0]) #image_cache_len, decoding_cache_len
                    # std_compress_attn = 0.5 * (compressed_outputs.attentions[2].mean(dim=1) + compressed_outputs.attentions[2].max(dim=1)[0])
                    preview_kvcache, _ = split_kvcache(compressed_outputs.past_key_values, (preview_len[0]+preview_len[1],pos_len[1]+pos_len[2]))
                    full_cache = original_kv_cache
                    std_compress_hs = compressed_outputs.hidden_states
                    with torch.no_grad():
                        original_twice_attention_mask = torch.ones(
                            (selected_embeds.shape[0], selected_embeds.shape[1] + split_cache_l[0][0].shape[2]),
                            device=selected_embeds.device)
                        original_twice_outputs = llm.model.llm(inputs_embeds=selected_embeds,
                                                       attention_mask=original_twice_attention_mask,
                                                       past_key_values=split_cache_l,
                                                   output_attentions=True, output_hidden_states=True)
                        # std_original_attn = 0.5 * (original_twice_outputs.attentions[2].mean(dim=1))
                        std_original_hs = original_twice_outputs.hidden_states


                        # # todo delete
                        # compressed_gen_attention_mask = compressed_attention_mask[:, :preview_len[0]+1]
                        # # input_ids_comp_generate = torch.zeros_like(compressed_gen_attention_mask).type(torch.int64)
                        # # input_ids_comp_generate[:, -input_ids.shape[1]:] = input_ids
                        # inputs_embeds_generate = inputs_embeds[:, :preview_len[0]+1]
                        # min_corr_length = min(inputs_embeds_generate.shape[1], inputs_embeds.shape[1])
                        # inputs_embeds_generate[:, -min_corr_length:] = inputs_embeds[:, -min_corr_length:]
                        # compressed_kv_cache = compressed_kv_cache_backup.clone()

                        # compressed_generate_outputs = llm.model.llm.generate(inputs_embeds=inputs_embeds_generate,
                        #                  pad_token_id=0,
                        #                  eos_token_id=llm.terminators,
                        #                 attention_mask=compressed_gen_attention_mask,
                        #                 past_key_values=compressed_kv_cache,do_sample=True, temperature=0.7,
                        #                 return_dict_in_generate=True, use_cache=True, max_new_tokens=output_tokens)

                        # original_twice_gen_attention_mask = original_twice_attention_mask[:, :pd_len[0] + 1]
                        # # input_ids_generate = torch.zeros_like(original_twice_gen_attention_mask).type(torch.int64)
                        # # input_ids_generate[:, -input_ids.shape[1]:] = input_ids
                        # inputs_embeds_generate = inputs_embeds[:, :pd_len[0] + 1]
                        # min_corr_length = min(inputs_embeds_generate.shape[1], inputs_embeds.shape[1])
                        # inputs_embeds_generate[:, -min_corr_length:] = inputs_embeds[:, -min_corr_length:]
                        # split_cache_l = split_cache_l_backup.clone()

                        # original_twice_generate_outputs = llm.model.llm.generate(inputs_embeds=inputs_embeds_generate,
                        #                                                      attention_mask=original_twice_gen_attention_mask,
                        #                                                      past_key_values=split_cache_l,
                        #                                                      pad_token_id=0,
                        #                                                      eos_token_id=llm.terminators,
                        #                                                      do_sample=True, temperature=1.,
                        #                                                      return_dict_in_generate=True,
                        #                                                      use_cache=True,
                        #                                                      max_new_tokens=output_tokens)
                        # print(llm.processor.batch_decode(original_twice_generate_outputs.sequences[:, inputs_embeds_generate.shape[1]-1:]),
                        #       llm.processor.batch_decode(compressed_generate_outputs.sequences[:, inputs_embeds_generate.shape[1]-1:]))

                    compress_output_logits = torch.softmax(compressed_outputs.logits, dim=-1)
                    original_twice_outputs_logits = torch.softmax(original_twice_outputs.logits, dim=-1)
                    loss = loss_fn.forward_attn(original_twice_outputs_logits,
                                                compress_output_logits , pos_len,
                                                std_original_hs, std_compress_hs)
                    acc1, acc2, perplexity, mid_idx, output_idx = loss_fn.calc_metrics3(original_twice_outputs_logits[:, :(pos_len[0]+pos_len[1]), :], compress_output_logits[:, :(pos_len[0]+pos_len[1]), :])
                    is_preview = False
                    optimizer_de.zero_grad()
                    loss.backward()
                    optimizer_de.step()
                    total_loss += loss.item()
                    total_acc += acc1
                    total_acc2 += acc2

                    i_batch += input_ids.shape[0] # add shape batchsize

                    avg_mini_train_loss = total_loss / i_batch
                    avg_mini_train_acc = total_acc / i_batch
                    avg_mini_train_acc2 = total_acc2 / i_batch
                    print(f"minibatch Train Loss: {avg_mini_train_loss:.4f}, cur_loss: {loss.item():.4f}, "
                          f"avg_acc : {avg_mini_train_acc * 100:.2f}%, acc : {acc1 * 100:.2f}%, "
                          f"avg_ff : {avg_mini_train_acc2:.2f}, ff: {acc2:.2f}, perplexity:{perplexity:.2f}, "
                          f"{compressed_kv_cache[0][0].shape[2]}")
                else: # exam
                    with torch.no_grad():

                        attention_mask_padding = padding_mask(attention_mask, preview_kvcache[0][0].shape[2])
                        original_outputs = llm.model.generate(input_ids=input_ids, attention_mask=attention_mask_padding,
                                                              output_hidden_states=True, max_new_tokens=output_tokens,
                                                              do_sample=True, temperature=1., past_key_values=preview_kvcache,
                                                              return_dict_in_generate=True,
                                                              output_scores=True)
                        # original_outputs_gradgen = generate_with_grad(llm, input_ids=input_ids, attention_mask=attention_mask_padding,
                        #                                       output_hidden_states=True, max_new_tokens=output_tokens,
                        #                                       do_sample=True, temperature=1., past_key_values=preview_kvcache,
                        #                                       return_dict_in_generate=True,
                        #                                       output_scores=True)

                        # attention_mask_padding_full = padding_mask(attention_mask, full_cache[0][0].shape[2])
                        # original_full_outputs = llm.model.generate(input_ids=input_ids,
                        #                                       attention_mask=attention_mask_padding_full,
                        #                                       output_hidden_states=True, max_new_tokens=output_tokens,
                        #                                       do_sample=True, temperature=1.,
                        #                                       past_key_values=full_cache,
                        #                                       return_dict_in_generate=True,
                        #                                       output_scores=True, output_attentions=True)
                        original_kv_cache = original_outputs.past_key_values
                        original_h_states = original_outputs.hidden_states

                        pd_len[0] = preview_kvcache[0][0].shape[2] + input_ids.shape[-1] - 1

                        pd_len[1] = original_kv_cache[0][0].shape[2] - pd_len[0]
                        compress_input, compress_input_logits, decode_split_len = proc_original_input(original_outputs, input_ids, pd_len)
                        split_cache_l, split_list = split_kvcache(original_kv_cache, pd_len, decode_split_len)
                    # it_len = [0, 0]
                    # it_len[1] = input_ids[:, torch.where(input_ids == 32000)[1] + 1:].shape[-1]
                    # it_len[0] = split_cache_l[0][0].shape[2] - it_len[1]
                    it_len = [0,0] # image_Lens, text_Lens
                    it_len[1] = split_cache_l[0][0].shape[2]-it_len[0]
                    compressed_kv_cache = compressor(split_cache_l, it_len, forward_mode='train')
                    hidden_embeds = loss_fn.cvt_hidden_states(original_h_states)[0]
                    selected_positions, pos_len = get_selected_positions(pd_len, num_end_tokens, num_random_tokens,
                                                                         device)
                    selected_embeds = hidden_embeds[:, selected_positions, :]
                    compressed_attention_mask = torch.ones(
                        (selected_embeds.shape[0], selected_embeds.shape[1] + compressed_kv_cache[0][0].shape[2]),
                        device=selected_embeds.device)
                    compressed_outputs = llm.model(inputs_embeds=selected_embeds,
                                                   attention_mask=compressed_attention_mask,
                                                   past_key_values=compressed_kv_cache,
                                                   output_attentions=True, output_hidden_states=True)
                    # std_compress_attn = 0.5 * (compressed_outputs.attentions[2].mean(dim=1) + compressed_outputs.attentions[2].max(dim=1)[0])
                    std_compress_hs = compressed_outputs.hidden_states
                    with torch.no_grad():
                        original_twice_attention_mask = torch.ones(
                            (selected_embeds.shape[0], selected_embeds.shape[1] + split_cache_l[0][0].shape[2]),
                            device=selected_embeds.device)
                        original_twice_outputs = llm.model(inputs_embeds=selected_embeds,
                                                           attention_mask=original_twice_attention_mask,
                                                           past_key_values=split_cache_l,
                                                           output_attentions=True, output_hidden_states=True)
                        # std_original_attn = 0.5 * (original_twice_outputs.attentions[2].mean(dim=1))
                        std_original_hs = original_twice_outputs.hidden_states
                    compress_output_logits = torch.softmax(compressed_outputs.logits, dim=-1)
                    original_twice_outputs_logits = torch.softmax(original_twice_outputs.logits, dim=-1)
                    loss = loss_fn.forward_attn(original_twice_outputs_logits,
                                                compress_output_logits, pos_len,
                                                std_original_hs, std_compress_hs)

                    acc1, acc2, perplexity, mid_idx, output_idx = loss_fn.calc_metrics3(
                        original_twice_outputs_logits[:, (pos_len[0] + pos_len[1]):, :],
                        compress_output_logits[:, (pos_len[0] + pos_len[1]):, :])
                    is_preview = False
                    optimizer_de.zero_grad()
                    loss.backward()
                    optimizer_de.step()
                    total_loss += loss.item()
                    total_acc += acc1
                    total_acc2 += acc2

                    i_batch += input_ids.shape[0]  # add shape batchsize

                    avg_mini_train_loss = total_loss / i_batch
                    avg_mini_train_acc = total_acc / i_batch
                    avg_mini_train_acc2 = total_acc2 / i_batch
                    print(f"minibatch Train Loss: {avg_mini_train_loss:.4f}, cur_loss: {loss.item():.4f}, "
                          f"avg_acc : {avg_mini_train_acc * 100:.2f}%, acc : {acc1 * 100:.2f}%, "
                          f"avg_ff : {avg_mini_train_acc2:.2f}, ff: {acc2:.2f}, perplexity:{perplexity:.2f}, "
                          f"{compressed_kv_cache[0][0].shape[2]}")


        avg_train_loss = total_loss / len(train_loader)
        avg_train_acc = total_acc / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc : {avg_train_acc*100:.2f}%")

        # Validation
        val_loss, val_acc = validate_minicpm(compressor, loss_fn, llm, val_loader, output_tokens, pd_len, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

        scheduler_de.step(val_loss)

        # Save the best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        save_checkpoint(
            epoch=epoch,
            model=compressor,
            optimizer=optimizer_de,
            scheduler=scheduler_de,
            loss=val_loss,
            is_best=is_best,
            filefoldname=f'ckpt_store',
            filename=compressor_save_path,
            # filename=f'finetune_mlp_1030_mm_8.pth'  Val Loss: 0.3794, Val Acc: 87.76% nocor_position
            # filename=f'finetune_mlp_1030_mm_7.pth'  0.4222, Val Acc: 84.96% temperature
            # filename=f'finetune_mlp_1030_mm_6.pth' yes 78.59%
            # filename=f'finetune_mlp_1030_mm_5.pth' yes 5-69.84%
        )
        epoch += 1

    return compressor


def validate_minicpm(compressor, loss_fn, llm, val_loader, output_tokens, pd_len, device):
# def validate(pretrained_encoder, decoder, loss_fn, llm, val_loader, pd_len, device):
#     pretrained_encoder.eval()
#     decoder.eval()
    compressor.eval()
    total_loss = 0
    total_accuracy = 0
    patch_size = 14
    i_batch = 0

    with torch.no_grad():
        for multi_dialog_batch in tqdm(val_loader, desc=f"Validate"):
        # for multi_dialog_batch in val_loader:
            is_preview = True
            for i_dialog, dialog_batch in enumerate(multi_dialog_batch):
                if is_preview == True:
                    is_preview = False
                    input_ids = dialog_batch['input_ids'].to(device)
                    attention_mask = dialog_batch['attention_mask'].to(device)
                    inputs_embeds = dialog_batch['inputs_embeds'].to(device)
                    # pixel_values = dialog_batch['pixel_values'].to(device)

                    original_outputs = llm.model.llm.generate(inputs_embeds=inputs_embeds, max_new_tokens=output_tokens,
                                                            attention_mask=attention_mask,
                                                            pad_token_id=0,
                                                            eos_token_id=llm.terminators,
                                                            do_sample = True, temperature = 1., use_cache=True, return_dict_in_generate=True,
                                                            output_scores=True)
                    original_kv_cache = original_outputs.past_key_values
                    

                    # pd_len[0] = round(pixel_values.shape[2]/patch_size)*round(pixel_values.shape[3]/patch_size) + input_ids.shape[-1] - 2
                    pd_len[0] = attention_mask.shape[-1]
                    pd_len[1] = original_kv_cache[0][0].shape[2] - pd_len[0]

                    compress_input, compress_input_logits, decode_split_len = proc_original_input(original_outputs, input_ids,pd_len)
                    split_cache_l, split_list = split_kvcache(original_kv_cache, pd_len, decode_split_len)
                    # encoded_kv_cache = pretrained_encoder(split_cache_l)
                    # compressed_kv_cache = decoder(encoded_kv_cache)

                    # compressed_kv_cache = compressor(split_cache_l)
                    it_len = [0, 0]
                    # it_len[1] = input_ids[:, torch.where(input_ids == 32000)[1] + 1:].shape[-1]
                    it_len[1] = split_cache_l[0][0].shape[2] - it_len[0]
                    compressed_kv_cache = compressor(split_cache_l, it_len, forward_mode='val')
                    compressed_kv_cache = CompressibleCache.from_legacy_cache(compressed_kv_cache)
                    # split_cache_l[0][0].shape[2] 应该用去掉padding的部分
                    compressed_kv_cache.set_seen_tokens(split_cache_l[0][0].shape[2])

                    compressed_attention_mask = torch.ones(
                        (compress_input.shape[0], compress_input.shape[1] + compressed_kv_cache[0][0].shape[2]),
                        device=input_ids.device)
                    try:
                        # TODO skip it
                        compressed_outputs = llm.model.llm(input_ids=compress_input,
                                                    attention_mask=compressed_attention_mask,
                                                    past_key_values=compressed_kv_cache)
                        compress_output_logits = torch.softmax(compressed_outputs.logits, dim=-1)

                        # loss = loss_fn(compress_input_logits, compress_output_logits[:, :-1, :])
                        # cur_accuracy, _, _ = loss_fn.calc_metrics(compress_input_logits, compress_output_logits[:, :-1, :])
                        # loss = loss_fn(compress_input_logits, compress_output_logits[:, :-1, :],[compress_input_logits.shape[1],0,0])
                        loss = loss_fn(compress_input_logits[:, 1:, :], compress_output_logits[:, :-1, :])
                        cur_accuracy, _, _ = loss_fn.calc_metrics(compress_input_logits[:, 1:, :], compress_output_logits[:, :-1, :])
                        # 困惑度
                        # calc accuracy


                        # original_outputs = llm.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
                        # original_kv_cache = original_outputs.past_key_values
                        #
                        # encoded_kv_cache = pretrained_encoder(original_kv_cache)
                        # compressed_kv_cache = decoder(encoded_kv_cache)
                        #
                        # compressed_outputs = llm.model(input_ids=input_ids, attention_mask=attention_mask,
                        #                                past_key_values=compressed_kv_cache)
                        #
                        # loss = kv_cache_consistency_loss(original_outputs, compressed_outputs)
                        total_loss += loss.item()
                        total_accuracy += cur_accuracy
                    except Exception as e:
                        print(e)
                        continue
                else:
                    pass

                    # break

    return total_loss / len(val_loader),  total_accuracy / len(val_loader)


def test(pretrained_encoder, finetune_decoder, llm, train_data, val_data, num_epochs, learning_rate, device, compression_ratio, prompt_len):
    # output_tokens = 16384 # 991-321 (64best)
    # output_tokens = 8192 # 333 - 162 (64best)
    output_tokens = 4096 # 126 - 80 (64best)
    batch_size = 1
    pd_len = [prompt_len, output_tokens]

    pretrained_encoder.eval()
    finetune_decoder.eval()

    # # 冻结 encoder 参数
    # for param in pretrained_encoder.parameters():
    #     param.requires_grad = False

    # optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    for batch in tqdm(train_loader):
        input_ids = batch['input_ids'].to(device)


        start_time2 = time.time()
        with torch.no_grad():
            total_tokens = 0
            kv_len = pd_len[0]
            no_kv = True
            compressed_kv_cache = 0
            while(total_tokens < output_tokens):
                # forloop_token = 512
                pred_kv_len = 64
                rest_token = output_tokens - total_tokens
                # cur_token = min(pred_kv_len-kv_len+1,rest_token)
                if no_kv == True:
                    cur_token = min(max(pred_kv_len - kv_len, 1), rest_token)
                    original_outputs = llm.model.generate(input_ids=input_ids, max_new_tokens=cur_token,
                                                          do_sample=False, use_cache=True, return_dict_in_generate=True)
                    # print('122')
                    kv_len = original_outputs.past_key_values[0][0].shape[2]
                    no_kv = False
                    compress_time = kv_len//pred_kv_len
                else:
                    cur_token = min(pred_kv_len - kv_len, rest_token)
                    original_outputs = llm.model.generate(input_ids=original_outputs.sequences[:, -(kv_len+1):], max_new_tokens=cur_token,
                                            do_sample=False, use_cache=True, return_dict_in_generate=True, past_key_values=compressed_kv_cache)
                    kv_len = original_outputs.past_key_values[0][0].shape[2]
                    compress_time = 1
                total_tokens += cur_token
                # else
                if total_tokens < output_tokens:
                    # compress kvcache
                    start_time3 = time.time()
                    target_kvcache = original_outputs.past_key_values
                    for _ in range(compress_time):
                        encoded_kv_cache = pretrained_encoder(target_kvcache)
                        compressed_kv_cache = finetune_decoder(encoded_kv_cache)
                        target_kvcache = compressed_kv_cache
                    final_time3 = time.time()
                    compressed_time = final_time3 - start_time3
                    print(compressed_time)
                    kv_len = compressed_kv_cache[0][0].shape[2]

            # total_tokens += cur_token
        # print(final_time1-)
        final_time2 = time.time()
        fold_time = final_time2 - start_time2
        print("fold_time", fold_time)

        start_time1 = time.time()
        with torch.no_grad():
            original_outputs = llm.model.generate(input_ids=input_ids, max_new_tokens=output_tokens,
                                                  do_sample=False, use_cache=True, return_dict_in_generate=True)
        final_time1 = time.time()
        directly_time = final_time1 - start_time1
        print(directly_time, fold_time)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="minicpm压缩")

    # 模型参数
    # parser.add_argument("--model-path", type=str, required=True, help="主模型路径")
    # parser.add_argument("--draft-model-path", type=str, required=True, help="Draft模型路径")
    # parser.add_argument("--model-path", type=str, default="/home/wanghaojie/vllm/huggingface/Llama-2-70b-chat-hf", help="主模型路径")
    # parser.add_argument("--draft-model-path", type=str, default="/home/wanghaojie/vllm/huggingface/Llama-2-7b-chat-hf", help="Draft模型路径")
    # parser.add_argument("--speculative-mode", type=str, default="specr", choices=["sd", "nsd", "specr", "trispecr"], help="推测解码模式")
    # parser.add_argument("--special-str", type=str, default="70b7b_kernel", help="特殊标识")
    # # parser.add_argument("--enable-speculative", '-sd', action='store_true', default=False, help="是否启用推测解码")
    # parser.add_argument("--tp-size", type=int, default=4, help="张量并行程度")

    # # 推测解码参数
    # parser.add_argument("--num-steps", type=int, default=4, help="推测步数")
    # parser.add_argument("--num-draft-tokens", type=int, default=4, help="每步生成的draft token数量")
    # parser.add_argument("--eagle-topk", type=int, default=1, help="Eagle验证的top-k")

    # # 测试参数
    # parser.add_argument("--output-len", type=int, default=256, help="生成的输出长度")
    # parser.add_argument("--num-prompts", type=int, default=128, help="测试的prompt数量")
    # parser.add_argument("--result-path", type=str, default="./results/specrouter_human", help="结果文件路径")
    # parser.add_argument("--profile", action="store_true", default=False, help="是否启用profile")
    parser.add_argument("--datasets-type", type=str, default="image", choices=["text", "image"], help="模式")
    parser.add_argument("--model-mode", type=str, default="minicpm", choices=["llava", "minicpm"], help="模式")


    return parser.parse_args()

def main():
    # 参数设置
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = 5e-4
    max_input_len = 512
    random.seed(48)
    
    # 模型和数据路径
    
    # 区分模式: 'image' 或 'text'
    mode = "image"  # 可以修改为 'image' 使用原有功能
    # mode = "text"  # 可以修改为 'image' 使用原有功能
    
    # 基于模式选择不同的数据路径和函数
    if mode == "text":
        # model_path = "/data/huggingface/DeepSeek-R1-Distill-Llama-8B"  # 填入您的模型路径
        # tokenizer_path = "/data/huggingface/Llama-3.1-8B/"  # 填入您的模型路径
        model_path = "/data/huggingface/Llama-2-7b-chat-hf"  # 填入您的模型路径
        json_path = '/data/huggingface/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json'  # 填入您的文本数据集路径
        compressor_path = './ckpt_store/best_finetune_mlp_text_2.pth'
    else:  # image 模式
        model_path = "/data/huggingface/MiniCPM-V-2_6"  # 填入您的模型路径
        tokenizer_path = "/data/huggingface/MiniCPM-V-2_6"
        # model_path = "/data/huggingface/llava-1.5-7b-hf"  # 填入您的模型路径
        json_path = '/data/huggingface/LLaVA-Instruct-150K/llava_v1_5_mix665k.json'  # 填入您的图像数据集路径
        imgsets_path = str(DATASETS_DIR) + "/"  # 填入您的图像文件路径
        # compressor_load_path = "./ckpt_store/best_finetune_mlp_1030_mm_8.pth"
        compressor_load_path = ""
        compressor_save_path = f'finetune_mlp_13B_mm_2_minicpm.pth'
    # save_path = 'finetune_mlp_text_13B.pth'
    # save_path = 'finetune_mlp_text_2.pth'# Epoch 4/40, Val Loss: 0.5760, Val Acc: 78.08%
    
    # 通用参数
    num_samples = 320  # 调整样本数量
    # num_samples = 500  # 调整样本数量
    train_ratio = 0.8
    num_epochs = 40
    # is_continue = True
    is_continue = False
    launch_padding_flag = False
    compression_ratio = 5  # 压缩比例
    # compression_ratio = 5  # 压缩比例

    # 加载模型
    # torch_dtype = torch.float16  # 可选 torch.float16 用于半精度
    torch_dtype = torch.bfloat16  # 可选 torch.float16 用于半精度
    # torch_dtype = torch.float32  # 可选 torch.float16 用于半精度
    
    # 基于模式加载不同类型的数据
    if mode == "text":
        data = load_text_data(json_path, num_samples)
        train_size = int(len(data) * train_ratio)
        train_data = CustomTextDataset(data[:train_size], llm.tokenizer, max_length=max_input_len, launch_padding_flag=launch_padding_flag)
        val_data = CustomTextDataset(data[train_size:], llm.tokenizer, max_length=max_input_len, launch_padding_flag=launch_padding_flag)
        
        # 进行文本模式的训练
        compressor = finetune_com_text(llm, train_data, val_data, num_epochs, learning_rate, device,
                                           compression_ratio, max_input_len, is_continue, compressor_path, save_path)
    else:  # image 模式
        
        
        # 使用原有的多模态训练函数
        if args.model_mode == "minicpm":
            llm = LLM(model_path, processor_path=tokenizer_path, torch_dtype=torch_dtype, mode=mode, model_mode=args.model_mode, trust_remote_code=True)
            device = llm.device
            data = load_image_data_minicpm(json_path, imgsets_path, num_samples)
            train_size = int(len(data) * train_ratio)
            train_data = CustomImageTextMinicpmDataset(data[:train_size], llm.model, llm.processor, max_length=max_input_len, launch_padding_flag=launch_padding_flag)
            val_data = CustomImageTextMinicpmDataset(data[train_size:], llm.model, llm.processor, max_length=max_input_len, launch_padding_flag=launch_padding_flag)
            compressor = finetune_com_minicpm(llm, train_data, val_data, compressor_load_path, compressor_save_path,
                                                num_epochs, learning_rate, device,
                                                compression_ratio, max_input_len, is_continue)
        else:
            llm = LLM(model_path, processor_path=tokenizer_path, torch_dtype=torch_dtype, mode=mode, model_mode=args.model_mode)
            device = llm.device
            data = load_image_data(json_path, imgsets_path, num_samples)
            train_size = int(len(data) * train_ratio)
            train_data = CustomImageTextDataset(data[:train_size], llm.processor, max_length=max_input_len, launch_padding_flag=launch_padding_flag)
            val_data = CustomImageTextDataset(data[train_size:], llm.processor, max_length=max_input_len, launch_padding_flag=launch_padding_flag)
            compressor = finetune_com(llm, train_data, val_data, compressor_load_path, compressor_save_path,
                                            num_epochs, learning_rate, device,
                                            compression_ratio, max_input_len, is_continue)

    print("Finetuning completed!")

if __name__ == "__main__":
    main()