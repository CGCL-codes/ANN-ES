import os
import argparse
import pickle
import torch
import json
import sys
import io
import random
import time
import numpy as np

from multiprocessing.pool import ThreadPool

from tqdm import tqdm, trange
from collections import OrderedDict

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_transformers.optimization import WarmupLinearSchedule
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.modeling_utils import WEIGHTS_NAME

from biencoder import BiEncoderRanker, load_biencoder
import logging

import common.utils as utils
import common.data_process as data
from common.zeshel_utils import DOC_PATH, WORLDS, world_to_id
from common.optimizer import get_bert_optimizer
from common.params import ANNESParser
import common.knn_retriever as knn


logger = None

def load_bm25_candidates(params):
    bm25_candidates_path = os.path.join(params["data_path"], "bm25_negative_indices.pkl")
    if os.path.exists(bm25_candidates_path):
        with open(bm25_candidates_path, "rb") as f:
            bm25_candidates = pickle.load(f)
    else:
        # TODO 创建b25 candidates
        pass
    return bm25_candidates


def load_entity_dict_tensor(params, tokenizer, logger):
    """
    load whole kb entity and convert to tensor input
    
    Parameters:
    params: command input params
    mode: train, valid or test. corresponding to the data file name.
    logger: logger output
    
    Return:
    entity_tensor_dict: use src and doc index to get doc tensor. src is the domain index
    entity_map: use src and doc_id to get doc index in domain. src is the domain index
    entity_dict: use src and doc index to get doc content. src is the domain index
    """
    entity_dict_tensor_pkl_path = os.path.join(params["data_path"], "entity_dict_tensor.pkl")
    if os.path.exists(entity_dict_tensor_pkl_path):
        with open(entity_dict_tensor_pkl_path, "rb") as f:
            entity_tensor_dict, entity_src_index_offset_map = pickle.load(f)
        # entity_src_index_offset_map = None
    else:
        entity_dict = {}
        entity_map = {}
        entity_tensor_dict = {}
        entity_src_index_offset_map = {}
        offset = 0
        for src in WORLDS:
            fname = os.path.join(params["document_path"], src + ".json")
            assert os.path.isfile(fname), "File not found! %s" % fname
            doc_map = {}
            doc_list = []
            doc_tensor_list = []
            with open(fname, 'rt') as f:
                for line in f:
                    line = line.rstrip()
                    item = json.loads(line)
                    doc_id = item["document_id"]
                    title = item["title"]
                    text = item["text"]
                    doc_map[doc_id] = len(doc_list)
                    doc_list.append(item)
                    
                    # doc_tensor_list.append(data.get_candidate_representation(text, tokenizer, params["max_cand_length"], title))
                    doc_tensor_list.append(data.get_candidate_sumy_representation(text, tokenizer, params["max_cand_length"], title))
                    offset += 1

            logger.info("Load for world %s." % src)
            entity_dict[src] = doc_list
            entity_map[src] = doc_map
            entity_tensor_dict[src] = doc_tensor_list
            entity_src_index_offset_map[src] = offset
        
        with open(entity_dict_tensor_pkl_path, "wb") as f:
            pickle.dump((entity_tensor_dict, entity_src_index_offset_map), f)
    return entity_tensor_dict, entity_src_index_offset_map

def load_data(params, tokenizer, mode:str, logger):
    """
    Parameters:
    params: command input params
    mode: train, valid or test. corresponding to the data file name.
    
    Return:
    dataset tensor
    """
    samples_pkl_path = os.path.join(params["data_path"], mode + "_samples.pkl")
    if os.path.exists(os.path.join(samples_pkl_path)):
        with open(samples_pkl_path, "rb") as f:
            samples = pickle.load(f)
    else:
        samples = utils.read_dataset(mode, params["data_path"])
        with open(samples_pkl_path, 'wb') as f:
            pickle.dump(samples, f)
    logger.info("Read %d %s samples." % (len(samples), mode))

    # tensor_data_path = os.path.join(params["data_path"], mode + "_tensor_data.pkl")
    tensor_data_path = os.path.join(params["data_path"], mode + "_data.pkl")
    if os.path.exists(os.path.join(tensor_data_path)):
        with open(tensor_data_path, "rb") as f:
            dict_data, tensor_data = pickle.load(f)
    else:
        if params["summ_mode"] == "sumy":
            # _, tensor_data = data.process_mention_data(
            dict_data, tensor_data = data.process_mention_data_by_sumy( # 先进行文本摘要，再截断
                samples,
                tokenizer,
                params["max_context_length"],
                params["max_cand_length"],
                context_key=params["context_key"],
                silent=params["silent"],
                logger=logger,
                debug=params["debug"],
            )
        else:
            dict_data, tensor_data = data.process_mention_data(
                samples,
                tokenizer,
                params["max_context_length"],
                params["max_cand_length"],
                context_key=params["context_key"],
                silent=params["silent"],
                logger=logger,
                debug=params["debug"],
            )
            
        with open(tensor_data_path, 'wb') as f:
            pickle.dump((dict_data, tensor_data), f)
    return dict_data, tensor_data


def merge_bm25_and_hard_negative_indices(bm25_indices, hard_indices, hard_k, epoch):
    assert(len(bm25_indices) == len(hard_indices))
    if epoch == 0:
        # 优先全用bm25
        for i in range(len(bm25_indices)):
            entity_indices = bm25_indices[i]
            entity_indices = torch.Tensor(entity_indices)
            # num = len(entity_indices)
            num = min(len(entity_indices), hard_k)
            hard_indices[i, :num] = entity_indices[:num]
    else:
        # 全用hard_indices
        pass
    return hard_indices

# The evaluate function during training uses in-batch negatives:
# for a batch of size B, the labels from the batch are used as label candidates
# B is controlled by the parameter eval_batch_size
def evaluate(
    reranker, eval_dataloader, params, device, logger,
):
    reranker.model.eval()
    if params["silent"]:
        iter_ = eval_dataloader
    else:
        iter_ = tqdm(eval_dataloader, desc="Evaluation")

    results = {}

    eval_accuracy = 0.0
    nb_eval_examples = 0
    nb_eval_steps = 0

    for step, batch in enumerate(iter_):
        batch = tuple(t.to(device) for t in batch)
        context_input, candidate_input, _, _ = batch
        with torch.no_grad():
            eval_loss, logits = reranker(context_input, candidate_input)

        logits = logits.detach().cpu().numpy()
        # Using in-batch negatives, the label ids are diagonal
        label_ids = torch.LongTensor(
                torch.arange(params["eval_batch_size"])
        ).numpy()
        tmp_eval_accuracy, _ = utils.accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += context_input.size(0)
        nb_eval_steps += 1

    normalized_eval_accuracy = eval_accuracy / nb_eval_examples
    logger.info("Eval accuracy: %.5f" % normalized_eval_accuracy)
    results["normalized_accuracy"] = normalized_eval_accuracy
    return results


def get_optimizer(model, params):
    return get_bert_optimizer(
        [model],
        params["type_optimization"],
        params["learning_rate"],
        fp16=params.get("fp16"),
    )


def get_scheduler(params, optimizer, len_train_data, logger):
    batch_size = params["train_batch_size"]
    grad_acc = params["gradient_accumulation_steps"]
    epochs = params["num_train_epochs"]

    num_train_steps = int(len_train_data / batch_size / grad_acc) * epochs
    num_warmup_steps = int(num_train_steps * params["warmup_proportion"])

    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=num_warmup_steps, t_total=num_train_steps,
    )
    logger.info(" Num optimization steps = %d" % num_train_steps)
    logger.info(" Num warmup steps = %d", num_warmup_steps)
    return scheduler


def main(params):
    model_output_path = params["output_path"]
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    logger = utils.get_logger(params["output_path"])

    # Init model
    reranker = BiEncoderRanker(params)
    tokenizer = reranker.tokenizer
    model = reranker.model

    device = reranker.device
    n_gpu = reranker.n_gpu

    if params["gradient_accumulation_steps"] < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                params["gradient_accumulation_steps"]
            )
        )

    # An effective batch size of `x`, when we are accumulating the gradient accross `y` batches will be achieved by having a batch size of `z = x / y`
    # args.gradient_accumulation_steps = args.gradient_accumulation_steps // n_gpu
    params["train_batch_size"] = (
        params["train_batch_size"] // params["gradient_accumulation_steps"]
    )
    train_batch_size = params["train_batch_size"]
    eval_batch_size = params["eval_batch_size"]
    grad_acc_steps = params["gradient_accumulation_steps"]

    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if reranker.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
        
    # load entity dict data
    entity_tensor_dict, entity_src_index_offset_map = load_entity_dict_tensor(params, tokenizer, logger)
    # if entity_src_index_offset_map is None:
    #     entity_src_index_offset_map = {}
    #     offset = 0
    #     for src in WORLDS:
    #         entity_src_index_offset_map[src] = offset
    #         offset += len(entity_tensor_dict[src])
            
    # load bm25 candidates
    bm25_candidates = load_bm25_candidates(params)

    # load train data
    train_dict_data, train_tensor_data = load_data(params, tokenizer, "train", logger)
    # if params["shuffle"]:
    #     train_sampler = RandomSampler(train_tensor_data)
    # else:
    #     train_sampler = SequentialSampler(train_tensor_data)
    train_sampler = SequentialSampler(train_tensor_data)

    train_dataloader = DataLoader(
        train_tensor_data, sampler=train_sampler, batch_size=train_batch_size
    )

    # Load eval data
    valid_dict_data, valid_tensor_data = load_data(params, tokenizer, "valid", logger)
    valid_sampler = SequentialSampler(valid_tensor_data)
    valid_dataloader = DataLoader(
        valid_tensor_data, sampler=valid_sampler, batch_size=eval_batch_size
    )

    # evaluate before training
    if False:
        results = evaluate(
            reranker, valid_dataloader, params, device=device, logger=logger,
        )

    number_of_samples_per_dataset = {}

    time_start = time.time()

    utils.write_to_file(
        os.path.join(model_output_path, "training_params.txt"), str(params)
    )

    logger.info("Starting training")
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, False)
    )

    optimizer = get_optimizer(model, params)
    scheduler = get_scheduler(params, optimizer, len(train_tensor_data), logger)

    model.train()

    best_epoch_idx = -1
    best_score = -1

    num_train_epochs = params["num_train_epochs"]
    for epoch_idx in trange(int(num_train_epochs), desc="Epoch"):
        tr_loss = 0
        results = None
        
        # 每个epoch开始前生成所有entity的embeddings
        all_entity_embeddings_path = os.path.join(params["output_path"], "all_entity_embeddings_epoch{}.t7".format(epoch_idx))
        if epoch_idx == 0 and os.path.exists(all_entity_embeddings_path): # 调试用，第一个epoch保存所有entity的embeddings
            all_entity_embeddings = torch.load(all_entity_embeddings_path)
        else:
            all_entity_embeddings = knn.get_all_entity_embeddings(reranker=reranker, entity_tensor_dict=entity_tensor_dict,
                                        params=params, logger=logger, batch_size=params["encoding_batch_size"])
            # if epoch_idx == 0:
            #     torch.save(all_entity_embeddings, all_entity_embeddings_path)
        # 创建faiss-index
        indexSavePath = os.path.join(params["output_path"], "entity_embeddings_indexer_epoch{}".format(epoch_idx))
        hnsw = params["hnsw"]
        if epoch_idx == 0 and os.path.exists(indexSavePath):
            indexer = knn.load_faiss_index(all_entity_embeddings, indexSavePath, hnsw=hnsw)
        else:
            indexer = knn.build_faiss_index(all_entity_embeddings, hnsw=hnsw, save=False, indexSavePath=indexSavePath)
            
        # 构建hard-negatives
        hard_negatives_path = os.path.join(params["output_path"], "hard_negatives_epoch{}.pkl".format(epoch_idx))
        exclude_gold = params["exclude_glod"]
        if epoch_idx == 0 and os.path.exists(hard_negatives_path): # 调试用，第一个epoch保存所有entity的embeddings
            with open(hard_negatives_path, "rb") as f:
                hard_indices = pickle.load(f)
        else:
            ctx_dataloader = DataLoader(
                train_tensor_data, sampler=train_sampler, batch_size=params["encoding_batch_size"]
            )
            hard_indices = knn.get_hard_negatives(indexer, ctx_dataloader, reranker, all_entity_embeddings, 
                                                    entity_src_index_offset_map, num_hards=params["k"], num_rands=0, exclude_golds=exclude_gold)
            if epoch_idx == 0:
                with open(hard_negatives_path, "wb") as f:
                    pickle.dump((hard_indices), f)
                    
        # hard_indices = merge_bm25_and_hard_negative_indices(bm25_candidates, hard_indices, params["k"], epoch_idx) # 第一轮epoch使用bm25进行warm up
        negative_ds = data.NegativesDataset(entity_tensor_dict, hard_indices, train_dict_data)
        if params["shuffle"]:
            negative_sampler = RandomSampler(negative_ds)
        else:
            negative_sampler = SequentialSampler(negative_ds)
        negative_dataloader = DataLoader(negative_ds, sampler=negative_sampler, batch_size=train_batch_size)
        
        model.train()

        if params["silent"]:
            iter_ = negative_dataloader
        else:
            iter_ = tqdm(negative_dataloader, desc="Batch")
        # if params["silent"]:
        #     iter_ = zip(train_dataloader, negative_dataloader)
        # else:
        #     iter_ = tqdm(zip(train_dataloader, negative_dataloader), desc="Batch")

        # for step, (batch, batch_neg) in enumerate(iter_):
        #     batch = tuple(t.to(device) for t in batch)
        #     # batch_neg = tuple(t.to(device) for t in batch_neg)
        #     batch_neg = batch_neg.to(device)
        #     context_input, candidate_input, _, _ = batch
        
        for step, batch in enumerate(iter_):
            batch = tuple(t.to(device) for t in batch)
            context_input, candidate_input, _, _, negatives = batch
            loss, _ = reranker(context_input, candidate_input, negatives)

            # if n_gpu > 1:
            #     loss = loss.mean() # mean() to average on multi-gpu.

            if grad_acc_steps > 1:
                loss = loss / grad_acc_steps

            tr_loss += loss.item()

            if (step + 1) % (params["print_interval"] * grad_acc_steps) == 0:
                logger.info(
                    "Step {} - epoch {} average loss: {}\n".format(
                        step,
                        epoch_idx,
                        tr_loss / (params["print_interval"] * grad_acc_steps),
                    )
                )
                tr_loss = 0

            loss.backward()

            if (step + 1) % grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), params["max_grad_norm"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % (params["eval_interval"] * grad_acc_steps) == 0:
                logger.info("Evaluation on the development dataset")
                evaluate(
                    reranker, valid_dataloader, params, device=device, logger=logger,
                )
                model.train()
                logger.info("\n")

        logger.info("***** Saving fine - tuned model *****")
        epoch_output_folder_path = os.path.join(
            model_output_path, "epoch_{}".format(epoch_idx)
        )
        utils.save_model(model, tokenizer, epoch_output_folder_path)

        output_eval_file = os.path.join(epoch_output_folder_path, "eval_results.txt")
        results = evaluate(
            reranker, valid_dataloader, params, device=device, logger=logger,
        )

        ls = [best_score, results["normalized_accuracy"]]
        li = [best_epoch_idx, epoch_idx]

        best_score = ls[np.argmax(ls)]
        best_epoch_idx = li[np.argmax(ls)]
        logger.info("\n")

    execution_time = (time.time() - time_start) / 60
    utils.write_to_file(
        os.path.join(model_output_path, "training_time.txt"),
        "The training took {} minutes\n".format(execution_time),
    )
    logger.info("The training took {} minutes\n".format(execution_time))

    # save the best model in the parent_dir
    logger.info("Best performance in epoch: {}".format(best_epoch_idx))
    params["path_to_model"] = os.path.join(
        model_output_path, 
        "epoch_{}".format(best_epoch_idx),
        WEIGHTS_NAME,
    )
    reranker = load_biencoder(params)
    utils.save_model(reranker.model, tokenizer, model_output_path)

    if params["evaluate"]:
        params["path_to_model"] = model_output_path
        evaluate(params, logger=logger)


if __name__ == "__main__":
    parser = ANNESParser(add_model_args=True)
    parser.add_training_args()
    parser.add_eval_args()

    args = parser.parse_args()
    print(args)

    params = args.__dict__
    main(params)
