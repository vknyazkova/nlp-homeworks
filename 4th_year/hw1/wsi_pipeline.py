from abc import abstractmethod
import pickle
from typing import Iterable, Tuple, List

from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel


class ContextEmb(Dataset):
    """
        PyTorch Dataset for contextual embeddings.

        Args:
            dataset (pd.DataFrame): DataFrame containing the dataset.
            word_col (str): Column name for word information.
            context_col (str): Column name for sentence context information.
            word_idx_col (str): Column name for word index in sentences.
        """
    def __init__(self,
                 dataset: pd.DataFrame,
                 word_col: str = 'word',
                 context_col: str = 'sent',
                 word_idx_col: str = 'word_idx'):
        self.data = dataset
        self.indices = list(dataset.index)
        self.word_col = word_col
        self.context_col = context_col
        self.word_idx_col = word_idx_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[int, str, str]:
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: Tuple containing (sample dataset index, target wordform, context) for the given index.
        """
        idx = self.indices[idx]
        word_idx = self.data.loc[idx, self.word_idx_col]
        wordform = self.data.loc[idx, self.context_col].split()[word_idx]
        return idx, wordform, self.data.loc[idx, self.context_col]


class EmbeddingModel:

    @abstractmethod
    def get_contextualized_embeddings(self,
                                      contexts: Iterable[str],
                                      target_words: Iterable[str]):
        ...


class BertEmbeddings(EmbeddingModel):
    def __init__(self, model_name: str = 'bert-base-multilingual-cased'):
        self.model = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def get_contextualized_embeddings(self, contexts: Iterable[str], target_words: Iterable[str]) -> List[np.ndarray]:
        """
        Get contextualized embeddings using BERT the target words in their context.

        Args:
            contexts (Iterable[str]): Iterable of sentence contexts.
            target_words (Iterable[str]): Iterable of target words to retrieve embedding for.

        Returns:
            List[np.ndarray]: List of contextualized embeddings.
        """
        tokenized_sentences = self.tokenizer(contexts, return_tensors='pt', padding='longest', truncation=True)
        with torch.no_grad():
            outputs = self.model(**tokenized_sentences)
        embeddings = outputs.last_hidden_state

        word_indices = [
            tokenized_sentences['input_ids'][i].tolist().index(self.tokenizer.encode(target_word)[1])
            for i, target_word in enumerate(target_words)
        ]

        word_embeddings = [embeddings[i, index, :].numpy() for i, index in enumerate(word_indices)]

        return word_embeddings


class WSIPipeline:
    """
    Word Sense Induction (WSI) pipeline.

    Args:
        dataset (pd.DataFrame): DataFrame containing the dataset.

    Attributes:
        dataset (pd.DataFrame): The input dataset.
        embeddings (np.ndarray): Array to store embeddings for clustering.
        clustering (dict): Dictionary to store clustering models {target_word: clustering_model}.
    """

    def __init__(self,
                 dataset: pd.DataFrame,
                 ):
        self.dataset = dataset
        self.embeddings = None
        self.clustering = {}

    def vectorize_contexts(self,
                           dataloader,
                           embedding_model: EmbeddingModel,
                           verbose=True):
        embeddings = {}
        for idx, word, context in tqdm(dataloader, disable=not verbose):
            emb = embedding_model.get_contextualized_embeddings(context, word)
            embeddings.update({idx: e for idx, e in zip(idx, emb)})
        embeddings = sorted(embeddings.items(), key=lambda x: x[0])
        embeddings = np.array([emb[1] for emb in embeddings])
        self.embeddings = embeddings

    def save_embeddings(self, path):
        np.save(path, self.embeddings)

    def load_embeddings(self, path):
        self.embeddings = np.load(path)

    @staticmethod
    def kmeans_clustering_max_sil_score(embeddings: np.ndarray,
                                        nclusters: Iterable[int]) -> Tuple[np.ndarray, KMeans]:
        """
        Perform k-means clustering with the number of clusters that maximizes silhouette score.

        Args:
            embeddings (np.ndarray): Array of embeddings.
            nclusters (Iterable[int]): Iterable of cluster numbers.

        Returns:
            Tuple[np.ndarray, KMeans]: Tuple containing cluster labels and fitted k-means model.
        """

        kmeans_model = None
        max_silhouette_score = -1

        for n in nclusters:
            kmeans = KMeans(n_clusters=n, random_state=42, n_init="auto")
            labels = kmeans.fit_predict(embeddings)

            silhouette_avg = silhouette_score(embeddings, labels)

            if silhouette_avg > max_silhouette_score:
                max_silhouette_score = silhouette_avg
                cluster_labels = labels
                kmeans_model = kmeans

        return cluster_labels, kmeans_model

    def save_cluster_models(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self.clustering, file)

    def load_cluster_models(self, path):
        with open(path, 'rb') as file:
            self.clustering = pickle.load(file)

    def clusterize_instances(self,
                             word: str,
                             k_range: Iterable[int] = range(2, 10),
                             fit: bool = False) -> Tuple[List[int], np.ndarray]:
        """
        Cluster instances of a specific word using k-means clustering.

        Args:
            word (str): The target word which contexts to cluster.
            k_range (Iterable[int]): Iterable of number of clusters to choose the best from.
            fit (bool): Flag indicating whether to fit a new clustering model or use one from self.clustering

        Returns:
            Tuple[List[int], np.ndarray]: Tuple containing a list of dataset indices and cluster labels.
        """
        dataset_idx = list(self.dataset.index[self.dataset.word == word])
        word_context_embs = self.embeddings[self.dataset.word == word]
        if fit:
            cluster_labels, kmeans_model = self.kmeans_clustering_max_sil_score(word_context_embs, k_range)
            self.clustering[word] = kmeans_model
        else:
            kmeans_model = self.clustering[word]
            cluster_labels = kmeans_model.predict(word_context_embs)
        return dataset_idx, cluster_labels

    def cluster_all_dataset(self,
                            k_range: Iterable[int] = range(2, 10),
                            fit: bool = False):
        predicted_labels = {}
        for word in pd.unique(self.dataset.word):
            dataset_idx, labels = self.clusterize_instances(word, k_range, fit)
            predicted_labels[word] = (dataset_idx, labels)
        return predicted_labels
