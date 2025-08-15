import os
import os.path as osp
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
# from gen_target_subg import parse_target_data
class PrototypicalBatchSampler(object):
    '''
    PrototypicalBatchSampler: yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
    In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.

    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    '''

    def __init__(self, labels, classes_per_it, num_samples, iterations):
        '''
        Initialize the PrototypicalBatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be infered from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_samples: number of samples for each iteration for each class (support + query)
        - iterations: number of iterations (episodes) per epoch
        '''
        super(PrototypicalBatchSampler, self).__init__()
        self.labels = labels
        self.classes_per_it = classes_per_it
        self.sample_per_class = num_samples
        self.iterations = iterations

        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)

        # create a matrix, indexes, of dim: classes X max(elements per class)
        # fill it with nans
        # for every class c, fill the relative row with the indices samples belonging to c
        # in numel_per_class we store the number of samples for each class/row
        self.idxs = range(len(self.labels))
        self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.indexes = torch.Tensor(self.indexes)
        self.numel_per_class = torch.zeros_like(self.classes)
        for idx, label in enumerate(self.labels):
            label_idx = np.argwhere(self.classes == label).item()
            self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx
            self.numel_per_class[label_idx] += 1

    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        spc = self.sample_per_class
        cpi = self.classes_per_it

        for it in range(self.iterations):
            batch_size = spc * cpi
            batch = torch.LongTensor(batch_size)
            c_idxs = torch.randperm(len(self.classes))[:cpi]
            for i, c in enumerate(self.classes[c_idxs]):
                s = slice(i * spc, (i + 1) * spc)
                # FIXME when torch.argwhere will exists
                label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
                sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:spc]
                batch[s] = self.indexes[label_idx][sample_idxs]

            batch = batch[torch.randperm(len(batch))]

            yield batch

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations

'''Simulate the data loading process of GraphCLIP, use the same dataset provided by paper'''
def reverse_dict(my_dict):
    return {value: key.replace('_', ' ') for key, value in my_dict.items()}

def init_dataset(opt):
    if opt.dataset_type == 'Cora':
        dataset = CoraGraphDataset(seed=8, use_text=True)
    elif opt.dataset_type == 'PubMed':
        dataset = PubMedGraphDataset(seed=8, use_text=True)#OOM error
    elif opt.dataset_type == 'Citeseer':
        dataset = CiteseerGraphDataset(seed=8, use_text=True)
    else:
        raise ValueError(f"Unsupported dataset type: {opt.dataset_type}")
    graph_data = dataset.get_graph_data()

    for key in ['x', 'edge_index', 'y', 'pe', 'root_n_index', 'batch']:
        graph_data[key] = graph_data[key].to(opt.device)

    # target_graph = parse_target_data(d, data)
    opt.graph_data = graph_data
    opt.num_classes = len(dataset.classes)
    opt.classes = dataset.classes
    opt.c_descs = dataset.c_descs
    dataset.idx_to_class = reverse_dict(dataset.class_to_idx)
    return dataset


def init_sampler(opt, labels):
    classes_per_it = opt.C
    num_samples = opt.K + opt.num_query
    return PrototypicalBatchSampler(labels=labels,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=opt.iterations)


def init_dataloader(opt):
    dataset = init_dataset(opt)
    sampler = init_sampler(opt, dataset.data.y.cpu().numpy())
    dataloader = torch.utils.data.DataLoader(dataset,batch_sampler=sampler,pin_memory=True)
    return dataset,dataloader



class CoraGraphDataset:
    def __init__(self, class_to_idx=None, seed=0, use_text=False):
        
        self.data, self.raw_texts = self._load_raw_data(seed, use_text)
    
        self.classes = ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
                      'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory']
        self.c_descs = [' which refers to research papers focusing on case-based reasoning (CBR) in the field of artificial intelligence. Case-based reasoning is a problem-solving approach that utilizes specific knowledge of previously encountered, concrete problem situations (cases). In this method, a new problem is solved by finding similar past cases and reusing them in the new situation. The approach relies on the idea of learning from past experiences to solve new problems, which makes it relevant in many applications including medical diagnosis, legal decision-making, and others. Thus, the ""Case Based"" category would include papers that primarily focus on this particular methodology and its various aspects.',
                   ' which would include research papers related to genetic algorithms (GAs). Genetic algorithms are a type of optimization and search algorithms inspired by the process of natural selection and genetics. These algorithms generate solutions to optimization problems using techniques inspired by natural evolution, such as inheritance, mutation, selection, and crossover. In practice, genetic algorithms can be used to find solutions to complex problems that are difficult to solve with traditional methods, particularly in domains where the search space is large, complex, or poorly understood. This category would cover various aspects of genetic algorithms, including their design, analysis, implementation, theoretical background, and diverse applications.',
                   " which refers to research papers revolving around the concept of artificial neural networks (ANNs). Neural networks are a subset of machine learning algorithms modelled after the human brain, designed to ""learn"" from observational data. They are the foundation of deep learning technologies and can process complex data inputs, find patterns, and make decisions. The network consists of interconnected layers of nodes, or ""neurons"", and each connection is assigned a weight that shapes the data and helps produce a meaningful output. Topics covered under this category could range from the architecture and function of different neural network models, advancements in training techniques, to their application in a multitude of fields such as image and speech recognition, natural language processing, and medical diagnosis.",
                   " which pertains to research papers that focus on probabilistic methods and models in machine learning and artificial intelligence. Probabilistic methods use the mathematics of probability to make predictions and decisions. They provide a framework to handle and quantify the uncertainty and incomplete information, which is a common scenario in real-world problems. This category could include topics like Bayesian networks, Gaussian processes, Markov decision processes, and statistical techniques for prediction and inference. These methods have applications in various areas such as computer vision, natural language processing, robotics, and data analysis, among others, due to their ability to model complex, uncertain systems and make probabilistic predictions.",
                   " which refers to research papers focusing on the area of machine learning known as reinforcement learning (RL). Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment to achieve a goal. The agent learns from the consequences of its actions, rather than from being explicitly taught, and adjusts its behavior based on the positive or negative feedback it receives, known as rewards or penalties. This category would include research exploring various RL algorithms, methodologies, theoretical underpinnings, performance enhancements, and practical applications. This field is particularly relevant in areas where decision making is crucial, such as game playing, robotics, resource management, and autonomous driving.",
                   " which pertains to research papers that concentrate on the domain of rule-based learning, also known as rule-based machine learning. Rule learning is a method in machine learning that involves the generation of a set of rules to predict the output in a decision-making system based on the patterns discovered from the data. These rules are often in an ""if-then"" format, making them interpretable and transparent. This category would encompass research involving various rule learning algorithms, their enhancements, theoretical foundations, and applications. Rule learning methods are particularly beneficial in domains where interpretability and understanding of the learned knowledge is important, such as in medical diagnosis, credit risk prediction, and more.",
                   ' which likely refers to research papers that delve into the theoretical aspects of machine learning and artificial intelligence. This includes a broad array of topics such as theoretical foundations of various machine learning algorithms, performance analysis, studies on learning theory, statistical learning, information theory, and optimization methods. Additionally, it could encompass the development of new theoretical frameworks, investigations into the essence of intelligence, the potential for artificial general intelligence, as well as the ethical implications surrounding AI. Essentially, the ""Theory"" category encapsulates papers that primarily focus on theoretical concepts and discussions, contrasting with more application-oriented research which centers on specific techniques and their practical implementation.']
        
        if class_to_idx is None:
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        else:
            self.class_to_idx = class_to_idx
            
        # Additional Attributes Required for Preparing GraphCLIP
        self.data.pe = torch.zeros(self.data.num_nodes, 32) 
        self.data.root_n_index = torch.arange(self.data.num_nodes) 
        self.data.batch = torch.zeros(self.data.num_nodes).long()

    def _load_raw_data(self, seed, use_text):

        if osp.exists(f"./processed_data/cora.pt"):
            data = torch.load(f"./processed_data/cora.pt", map_location='cpu', weights_only=False)
            # data.x = data.x.float() # Half into Float
            edge_index = to_undirected(data.edge_index)
            # edge_index, _ = add_self_loops(data.edge_index)
            data.edge_index = edge_index
            data.num_nodes = data.y.shape[0]
            return data, data.raw_texts

    def __len__(self):
        return self.data.num_nodes

    def __getitem__(self, idx):
        return {
            'node_id': idx,
            'label': self.data.y[idx].item()
        }

    def get_graph_data(self):
        return self.data


class PubMedGraphDataset:# this is not suitable for few-shot, because the class number is too small
    def __init__(self, class_to_idx=None, seed=0, use_text=False):

        self.data, self.raw_texts = self._load_raw_data(seed, use_text)
        self.classes = ['Diabetes Mellitus Experimental', 'Diabetes Mellitus Type1', 'Diabetes Mellitus Type2']
        self.c_descs = [
            ' which is a category of scientific literature found on PubMed that encompasses research related to experimental studies on diabetes mellitus. This category includes studies conducted in laboratory settings, often using animal models or cell cultures, to investigate various aspects of diabetes, such as its pathophysiology, treatment strategies, and potential interventions. Researchers in this field aim to better understand the underlying mechanisms of diabetes and develop experimental approaches to prevent or manage the disease. Experimental studies in this category may explore topics like insulin resistance, beta cell function, glucose metabolism, and the development of novel therapies for diabetes.',
            ' which focuses on scientific research related specifically to Type 1 diabetes mellitus. This category encompasses a wide range of studies, including clinical trials, epidemiological investigations, and basic research, all centered on understanding, diagnosing, managing, and potentially curing Type 1 diabetes. Researchers in this field explore areas such as the autoimmune processes underlying the disease, insulin therapy, glucose monitoring, pancreatic islet transplantation, and novel treatments aimed at improving the lives of individuals with Type 1 diabetes. It serves as a valuable resource for healthcare professionals, scientists, and policymakers interested in advancements related to Type 1 diabetes management and research.',
            ' which focuses on research related to Type 2 diabetes (T2D), and it can be differentiated from Diabetes Mellitus Type 1 (T1D) in the following ways: Etiology (Cause): Type 2 Diabetes (T2D): T2D is primarily characterized by insulin resistance, where the body\'s cells do not respond effectively to insulin, and relative insulin deficiency that develops over time. It is not primarily an autoimmune condition.']
        self.num_classes = 3
        # class_map = 'Experimental induced diabetes, Type 1 diabetes, Type 2 diabetes'
        # class_map = 'Diabetes Mellitus Experimental, Diabetes Mellitus Type1, Diabetes Mellitus Type2'
        if class_to_idx is None:
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        else:
            self.class_to_idx = class_to_idx

        # Additional Attributes Required for Preparing GraphCLIP
        self.data.pe = torch.zeros(self.data.num_nodes, 32)
        self.data.root_n_index = torch.arange(self.data.num_nodes)
        self.data.batch = torch.zeros(self.data.num_nodes).long()

    def _load_raw_data(self, seed, use_text):
        import torch
        import os.path as osp

        if osp.exists(f"./processed_data/pubmed.pt"):
            data = torch.load(f"./processed_data/pubmed.pt", map_location='cpu', weights_only=False)
            data.num_nodes = data.y.shape[0]
            raw_texts = ['None']*data.num_nodes  # we do not need raw texts for source data, because we already transform them into node features use miniLM
            edge_index = to_undirected(data.edge_index)
            # edge_index, _ = add_self_loops(data.edge_index)
            data.edge_index = edge_index
            return data, raw_texts
        else:
            raise NotImplementedError('No existing pubmed dataset!')

    def __len__(self):
        return self.data.num_nodes

    def __getitem__(self, idx):
        return {
            'node_id': idx,
            'label': self.data.y[idx].item()
        }

    def get_graph_data(self):
        return self.data


class CiteseerGraphDataset:# this is not suitable for few-shot, because the class number is too small
    def __init__(self, class_to_idx=None, seed=0, use_text=False):

        self.data, self.raw_texts = self._load_raw_data(seed, use_text)
        self.classes = ['Agents', 'Machine Learning', 'Information Retrieval', 'Database', 'Human Computer Interaction',
                   'Artificial Intelligence']
        self.c_descs = [
            ". Specifically, agents are autonomous entities that perceive their environment through sensors and act upon it using actuators. They are designed to achieve specific goals or tasks.",
            ". Specifically, ML research investigates how to create systems that can automatically improve their performance on tasks by identifying patterns and insights from vast amounts of data. Researchers in Machine Learning explore diverse techniques such as supervised learning, unsupervised learning, reinforcement learning, and deep learning to build systems that can predict outcomes, classify data, and make intelligent decisions.",
            ". Specifically, IR research focuses on the study of information retrieval systems, which are designed to help users find relevant information in large collections of data. Researchers in Information Retrieval explore techniques such as indexing, querying, and ranking to build systems that can efficiently retrieve information based on user queries.",
            ". Specifically, DB research investigates how to design, build, and manage databases, which are organized collections of data that can be accessed, managed, and updated. Researchers in Database Systems explore techniques such as data modeling, query languages, and transaction processing to build systems that can store, retrieve, and manipulate data.",
            ". Specifically, HCI research focuses on the study of human-computer interaction, which explores how people interact with computers and other digital technologies. Researchers in Human-Computer Interaction investigate how to design user-friendly interfaces, improve usability, and enhance user experience to build systems that are intuitive, efficient, and effective.",
            ". Specifically, AI research investigates how to create intelligent systems that can perform tasks that typically require human intelligence, such as perception, reasoning, learning, and decision-making. Researchers in Artificial Intelligence explore diverse techniques such as knowledge representation, planning, and natural language processing to build systems that can solve complex problems, adapt to new environments, and interact with humans.",
        ]
        self.num_classes = 6
        # class_map = 'Experimental induced diabetes, Type 1 diabetes, Type 2 diabetes'
        # class_map = 'Diabetes Mellitus Experimental, Diabetes Mellitus Type1, Diabetes Mellitus Type2'
        if class_to_idx is None:
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        else:
            self.class_to_idx = class_to_idx

        # Additional Attributes Required for Preparing GraphCLIP
        self.data.pe = torch.zeros(self.data.num_nodes, 32)
        self.data.root_n_index = torch.arange(self.data.num_nodes)
        self.data.batch = torch.zeros(self.data.num_nodes).long()

    def _load_raw_data(self, seed, use_text):
        import torch
        import os.path as osp

        data = torch.load(f"./processed_data/citeseer.pt", map_location='cpu', weights_only=False)
        data.edge_index = to_undirected(data.edge_index)
        return data, data.raw_texts

    def __len__(self):
        return self.data.num_nodes

    def __getitem__(self, idx):
        return {
            'node_id': idx,
            'label': self.data.y[idx].item()
        }

    def get_graph_data(self):
        return self.data
