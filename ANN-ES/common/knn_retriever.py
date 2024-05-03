import os
import torch
import faiss
import numpy as np
import random
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from common.zeshel_utils import DOC_PATH, WORLDS, world_to_id, Stats

logger = logging.getLogger()

class DenseIndexer(object):
    def __init__(self, buffer_size: int = 50000):
        self.buffer_size = buffer_size
        self.index_id_to_db_id = []
        self.index = None

    def index_data(self, data: np.array):
        raise NotImplementedError

    def search_knn(self, query_vectors: np.array, top_docs: int):
        raise NotImplementedError

    def serialize(self, index_file: str):
        logger.info("Serializing index to %s", index_file)
        faiss.write_index(self.index, index_file)

    def deserialize_from(self, index_file: str):
        logger.info("Loading index from %s", index_file)
        self.index = faiss.read_index(index_file)
        logger.info(
            "Loaded index of type %s and size %d", type(self.index), self.index.ntotal
        )


# DenseFlatIndexer does exact search
class DenseFlatIndexer(DenseIndexer):
    def __init__(self, vector_sz: int = 1, buffer_size: int = 50000):
        super(DenseFlatIndexer, self).__init__(buffer_size=buffer_size)
        self.index = faiss.IndexFlatIP(vector_sz)

    def index_data(self, data: np.array):
        n = len(data)
        # indexing in batches is beneficial for many faiss index types
        logger.info("Indexing data, this may take a while.")
        cnt = 0
        for i in range(0, n, self.buffer_size):
            vectors = [np.reshape(t, (1, -1)) for t in data[i : i + self.buffer_size]]
            vectors = np.concatenate(vectors, axis=0)
            self.index.add(vectors)
            cnt += self.buffer_size

        logger.info("Total data indexed %d", n)

    def search_knn(self, query_vectors, top_k):
        scores, indexes = self.index.search(query_vectors, top_k)
        return scores, indexes


# DenseHNSWFlatIndexer does approximate search
class DenseHNSWFlatIndexer(DenseIndexer):
    """
     Efficient index for retrieval. Note: default settings are for hugh accuracy but also high RAM usage
    """

    def __init__(
        self,
        vector_sz: int,
        buffer_size: int = 50000,
        store_n: int = 128,
        ef_search: int = 256,
        ef_construction: int = 200,
    ):
        super(DenseHNSWFlatIndexer, self).__init__(buffer_size=buffer_size)

        # IndexHNSWFlat supports L2 similarity only
        # so we have to apply DOT -> L2 similairy space conversion with the help of an extra dimension
        index = faiss.IndexHNSWFlat(vector_sz + 1, store_n)
        index.hnsw.efSearch = ef_search
        index.hnsw.efConstruction = ef_construction
        self.index = index
        self.phi = 0

    def index_data(self, data: np.array):
        n = len(data)

        # max norm is required before putting all vectors in the index to convert inner product similarity to L2
        if self.phi > 0:
            raise RuntimeError(
                "DPR HNSWF index needs to index all data at once,"
                "results will be unpredictable otherwise."
            )
        phi = 0
        for i, item in enumerate(data):
            doc_vector = item
            norms = (doc_vector ** 2).sum()
            phi = max(phi, norms)
        logger.info("HNSWF DotProduct -> L2 space phi={}".format(phi))
        self.phi = 0

        # indexing in batches is beneficial for many faiss index types
        logger.info("Indexing data, this may take a while.")
        cnt = 0
        for i in range(0, n, self.buffer_size):
            vectors = [np.reshape(t, (1, -1)) for t in data[i : i + self.buffer_size]]

            norms = [(doc_vector ** 2).sum() for doc_vector in vectors]
            aux_dims = [np.sqrt(phi - norm) for norm in norms]
            hnsw_vectors = [
                np.hstack((doc_vector, aux_dims[i].reshape(-1, 1)))
                for i, doc_vector in enumerate(vectors)
            ]
            hnsw_vectors = np.concatenate(hnsw_vectors, axis=0)

            self.index.add(hnsw_vectors)
            cnt += self.buffer_size
            logger.info("Indexed data %d" % cnt)

        logger.info("Total data indexed %d" % n)

    def search_knn(self, query_vectors, top_k):
        aux_dim = np.zeros(len(query_vectors), dtype="float32")
        query_nhsw_vectors = np.hstack((query_vectors, aux_dim.reshape(-1, 1)))
        logger.info("query_hnsw_vectors %s", query_nhsw_vectors.shape)
        scores, indexes = self.index.search(query_nhsw_vectors, top_k)
        return scores, indexes

    def deserialize_from(self, file: str):
        super(DenseHNSWFlatIndexer, self).deserialize_from(file)
        # to trigger warning on subsequent indexing
        self.phi = 1

def get_all_wiki_entity_embeddings(reranker, entity_tensor_list, params, logger, batch_size=1024, device="cuda"):
    logger.info("##########get_all_wiki_entity_embeddings##########")
    entity_tensors = []
    entity_tensors.extend([item["ids"] for item in entity_tensor_list])
    entity_tensors = torch.tensor(entity_tensors)
    
    entity_dataset = TensorDataset(entity_tensors)
    train_sampler = SequentialSampler(entity_dataset)

    dataloader = DataLoader(
        entity_dataset, sampler=train_sampler, batch_size=batch_size
    )
    reranker.model.eval()
    all_en_embeds = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            batch = tuple(t.to(device) for t in batch)
            en_embeds = reranker(None, batch[0], None, True, False) # context_input, cand_input, negative_input, encode_only, encode_ctx_or_cand
            # en_embeds = reranker.encode_candidate(batch[0])
            all_en_embeds.append(en_embeds)
    
    all_en_embeds = torch.cat(all_en_embeds, dim=0)
    return all_en_embeds
    

def get_all_entity_embeddings(reranker, entity_tensor_dict, params, logger, batch_size=1024, device="cuda", isTraining=True):
    logger.info("##########get_all_entity_embeddings##########")
    if isTraining:
        training_keys = ['american_football',
                        'doctor_who',
                        'fallout',
                        'final_fantasy',
                        'military',
                        'pro_wrestling',
                        'starwars',
                        'world_of_warcraft']
        entity_tensors = []
        for key in training_keys:
            entity_tensors.extend([item["ids"] for item in entity_tensor_dict[key]])
    else:
        for key in entity_tensor_dict.keys():
            entity_tensors.extend([item["ids"] for item in entity_tensor_dict[key]])
    entity_tensors = torch.tensor(entity_tensors)
    
    entity_dataset = TensorDataset(entity_tensors)
    train_sampler = SequentialSampler(entity_dataset)

    dataloader = DataLoader(
        entity_dataset, sampler=train_sampler, batch_size=batch_size
    )
    reranker.model.eval()
    all_en_embeds = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            batch = tuple(t.to(device) for t in batch)
            en_embeds = reranker(None, batch[0], None, True, False) # context_input, cand_input, negative_input, encode_only, encode_ctx_or_cand
            all_en_embeds.append(en_embeds)
    
    all_en_embeds = torch.cat(all_en_embeds, dim=0)
    return all_en_embeds


def build_faiss_index(candidate_embedding, index_buffer=50000, hnsw=True, save=False, indexSavePath=None):
    """
    Parameters:
    candidate_embedding: all the entity embeddings
    index_buffer: Temporal memory data buffer size (in samples) for indexer
    hnsw: If enabled, use inference time efficient HNSW index
    save: whether to save index data
    indexSavePath: index saving path. if save is True, you must provide this path.
    
    Returns:
    faiss_indexer, for the use of knn search
    """
    vector_size = candidate_embedding.size(1)
    if hnsw:
        index = DenseHNSWFlatIndexer(vector_size, index_buffer)
    else:
        index = DenseFlatIndexer(vector_size, index_buffer)

    index.index_data(candidate_embedding.numpy())
    if save:
        index.serialize(indexSavePath)
    return index

def load_faiss_index(candidate_embedding, indexSavePath, index_buffer=50000, hnsw=True):
    vector_size = candidate_embedding.size(1)
    if hnsw:
        index = DenseHNSWFlatIndexer(vector_size, index_buffer)
    else:
        index = DenseFlatIndexer(vector_size, index_buffer)
    index.deserialize_from(indexSavePath)
    return index


def get_hard_negatives(indexer, mention_loader, reranker, all_entity_embeddings,
                       entity_src_index_offset_map,
                       num_hards=32,
                       num_rands=32,
                       exclude_golds=True,
                       device="cuda"):
    reranker.model.eval()
    negative_indices = []
    hard_inputs = []
    
    random_cands_pool = list(range(len(all_entity_embeddings)))
    
    with torch.no_grad():
        print(len(mention_loader))
        for i, batch in enumerate(tqdm(mention_loader)):
            batch = tuple(t.to(device) for t in batch)
            context_input, candidate_input, src_vecs, label_idx = batch
            mention_embeds = reranker(context_input, None, None, True, True) # context_input, cand_input, negative_input, encode_only, encode_ctx_or_cand
            mention_embeds = mention_embeds.numpy()
            scores, hard_indicies = indexer.search_knn(mention_embeds, num_hards + 1)
            
            indicies = np.zeros((len(hard_indicies), num_hards + num_rands), dtype=hard_indicies.dtype)
            
            if exclude_golds:
                label_idx = label_idx.cpu().detach().numpy()
                w_ids = [WORLDS[src] for src in src_vecs]
                offsets = [entity_src_index_offset_map[wid] for wid in w_ids]
                label_idx_w = [label_idx[i] + offsets[i] for i in range(len(offsets))]
                for i, lblid in enumerate(label_idx_w):
                    col = np.where(hard_indicies[i] == lblid)
                    if len(col[0]) != 0:
                        col = col[0][0]
                        hard_indicies[i, col:num_hards] = hard_indicies[i, (col + 1):] # 将gold后面的前移一位
                    
                    # 生成rand negatives
                    if num_rands > 0:
                        rand_cands = np.array(random.sample(random_cands_pool, num_rands + 1))
                        col_rand = np.where(rand_cands == lblid)
                        if len(col_rand[0]) != 0:
                            col = col_rand[0][0]
                            rand_cands[col:num_rands] = rand_cands[(col + 1):]
                    
                    indicies[i, :num_hards] = hard_indicies[i, :num_hards]
                    if num_rands > 0:
                        indicies[i, num_hards:] = rand_cands[:num_rands]
            else:
                indicies[:, :num_hards] = hard_indicies[:, :num_hards]
                # todo generate rand negatives
            negative_indices.append(torch.tensor(indicies))
    negative_indices = torch.cat(negative_indices, dim=0)
    return negative_indices


def get_wiki_hard_negatives(indexer, mention_loader, reranker, all_entity_embeddings,
                        num_hards=32,
                        num_rands=32,
                        exclude_golds=True,
                        device="cuda"):
    reranker.model.eval()
    negative_indices = []
    hard_inputs = []
    
    random_cands_pool = list(range(len(all_entity_embeddings)))
    
    with torch.no_grad():
        print(len(mention_loader))
        for i, batch in enumerate(tqdm(mention_loader)):
            batch = tuple(t.to(device) for t in batch)
            context_input, candidate_input, label_idx = batch
            mention_embeds = reranker(context_input, None, None, True, True) # context_input, cand_input, negative_input, encode_only, encode_ctx_or_cand
            mention_embeds = mention_embeds.numpy()
            scores, hard_indicies = indexer.search_knn(mention_embeds, num_hards + 1)
            
            indicies = np.zeros((len(hard_indicies), num_hards + num_rands), dtype=hard_indicies.dtype)
            
            if exclude_golds:
                label_idx = label_idx.cpu().detach().numpy()
                for i, lblid in enumerate(label_idx):
                    col = np.where(hard_indicies[i] == lblid)
                    if len(col[0]) != 0:
                        col = col[0][0]
                        hard_indicies[i, col:num_hards] = hard_indicies[i, (col + 1):] # 将gold后面的前移一位
                    
                    # 生成rand negatives
                    if num_rands > 0:
                        rand_cands = np.array(random.sample(random_cands_pool, num_rands + 1))
                        col_rand = np.where(rand_cands == lblid)
                        if len(col_rand[0]) != 0:
                            col = col_rand[0][0]
                            rand_cands[col:num_rands] = rand_cands[(col + 1):]
                    
                    indicies[i, :num_hards] = hard_indicies[i, :num_hards]
                    if num_rands > 0:
                        indicies[i, num_hards:] = rand_cands[:num_rands]
            else:
                indicies[:, :num_hards] = hard_indicies[:, :num_hards]
                # todo generate rand negatives
            
            negative_indices.append(torch.tensor(indicies))
    negative_indices = torch.cat(negative_indices, dim=0)
    return negative_indices


def eval_wiki_data(indexer, mention_loader, reranker, all_entity_embeddings,
                   entity_tensor_list,
                    top_k=64,
                    save_result=False,
                    device="cuda"):
    reranker.model.eval()    
    stats = Stats(top_k)
    
    entity_tensors = []
    entity_tensors.extend([item["ids"] for item in entity_tensor_list])
    
    nn_context = []
    nn_candidates = []
    nn_labels = []
    
    def get_candidate_tensors(indicies, entity_tensors):
        candidate_tensors = []
        for ind in indicies:
            candidate_tensors.append(entity_tensors[ind])
        return candidate_tensors
    
    with torch.no_grad():
        print(len(mention_loader))
        for i, batch in enumerate(tqdm(mention_loader)):
            batch = tuple(t.to(device) for t in batch)
            context_input, candidate_input, label_idx = batch
            mention_embeds = reranker(context_input, None, None, True, True) # context_input, cand_input, negative_input, encode_only, encode_ctx_or_cand
            mention_embeds = mention_embeds.numpy()
            scores, hard_indicies = indexer.search_knn(mention_embeds, top_k)
            label_idx = label_idx.cpu().detach().numpy()
            for i, lblid in enumerate(label_idx):
                col = np.where(hard_indicies[i] == lblid)
                if len(col[0]) != 0:
                    stats.add(col[0][0])
                    
                    if save_result:
                        # add examples in new_data
                        cur_candidates = get_candidate_tensors(hard_indicies[i], entity_tensors) # entity_tensors[hard_indicies[i]]
                        nn_context.append(context_input[i].cpu().tolist())
                        nn_candidates.append(np.array(cur_candidates))
                        nn_labels.append(col[0][0])
                else:
                    stats.add(-1)
                    
                    if save_result:
                        cur_candidates = []
                        cur_candidates.append(entity_tensors[lblid[0]])
                        cur_candidates = cur_candidates + get_candidate_tensors(hard_indicies[i][:top_k - 1], entity_tensors)
                        nn_context.append(context_input[i].cpu().tolist())
                        nn_candidates.append(np.array(cur_candidates))
                        nn_labels.append(0)

    if not save_result:
        return stats
    else:
        nn_context = torch.LongTensor(nn_context)
        nn_candidates = torch.LongTensor(nn_candidates)
        nn_labels = torch.LongTensor(nn_labels)
        nn_data = {
            'context_vecs': nn_context,
            'candidate_vecs': nn_candidates,
            'labels': nn_labels,
        }
        return stats, nn_data