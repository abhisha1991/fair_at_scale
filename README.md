# Welcome to Fair at Scale!

# W210: UC Berkeley
#### Michael Golas, Abhinav Sharma, Emily Robles, Justin Wong
#### Guided by Prof. Puya Vahabi
#### Spring 2024


## Influence Maximization [website](https://sites.google.com/berkeley.edu/fairimpact?usp=sharing)
It refers to a class of algorithms that aim to maximize information spread in a graph under some budget constraints.
The aim is to find K most influential (seed) nodes from which diffusion of a specific message should start.
This project takes things a step further by adding a constraint of fairness when trying to still maximize influence.

## Hardware Requirements:
64 GB RAM is sufficient for loading the weibo graph, extracting features, ranking nodes, and training imfector.

However, 128 GB RAM is required for loading the source and target embedding matrices and running the fair imfector diffusion probabilities and running the IMINFECTOR algorithm. At peak, 95GB RAM is used. If you do not have enough RAM, the program will crash because you are out of memory.

## Getting Started
Our GitHub project allows you to run the IM with fairness at scale FPS algorithm in a streamlined Jupyter notebook. Data from the Weibo social media network can be accessed through Google Drive [here](https://drive.google.com/file/d/1AFuShgAdyoqodqR1oFlCRp7okEYDdeLt/view). To get started you will need:
- Scalable GPU and high RAM (> 50 GB) enabled environment for running notebooks
- Environment session must have long timeouts (preferably over 1 day)
- Storage environment with over 100 GB of storage
- For quick iterations, we used google drive (2 TB monthly subscription) + Google collab paid subscription


Prefer to use the notebooks. There's no guarantee the `models/` code will run effectively.

Ideally, set up Colab Enterprise with Vertex AI in a new GCP (Google Cloud Platform) project. Instead of google drive, you can use GCS Fuse to directly mount Google Cloud Storage to your local file system. This way, you can:
- create a [e2-standard-32 runtime](https://cloud.google.com/compute/docs/general-purpose-machines#e2_machine_types_table) to run your notebook (costs around $32 / 24-hours to run)
- mount the data directly from Google Cloud Storage available at [fair-influence-maximization-mounted](https://console.cloud.google.com/storage/browser/fair-influence-maximization-mounted?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&hl=en&project=d4w3-369005&prefix=&forceOnObjectsSortingFiltering=false)

## Optimizations compared to original [fair_at_scale](https://github.com/yu-ting-feng/fair_at_scale):
The following optimizations allowed us to reduce the full data loading, feature extraction, training, and evaluation for a single 1 epoch from 6 hours down to 3.5 hours.
- `remove_duplicates_fast`: optimized to come up with unique nodes and times in one pass. (4347x speedup from 61 seconds down to 13ms)
- `mapped_uid`: creating in-place dictionary and iterating through file lines instead of reading first. (2% speedup)
- parallelized `fair_im_processed_D_parallel` and renamed to `batched_fair_im_process_D_parallel`: chunked the normalization and softmax calculations across a multiprocessing Pool. (5x speedup from ~10 minutes to 2.5 minutes)


## Results of Running on [e2-standard-32](https://cloud.google.com/compute/docs/general-purpose-machines#e2_machine_types_table) in GCP using one attribute:
e2-standard-32 consists of 32 vCPUs and 128 GB Memory.

See IM_Notebook_Justin_Epochs1_synthetic_noisy_politics_x_age.ipynb for the most up to date notebook with optimizations included.

More info on outputs of the runs at this [doc](https://docs.google.com/document/d/13kgl_4QY2T9ODUrLtasu9MxsyAUduMjZT5W6Rg0m190/edit).
- [Gender on full weibo dataset for 10 epochs](https://console.cloud.google.com/storage/browser/fair-influence-maximization-mounted/data/Data/Weibo/Output_Full-attempt_2024-03-12?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&hl=en&project=d4w3-369005)
- [Gender on full weibo dataset for 1 epoch](https://console.cloud.google.com/storage/browser/fair-influence-maximization-mounted/data/Data/Weibo/Output_Full-attempt_2024-03-12_epoch1?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&hl=en&project=d4w3-369005)
- [Age groups from synthetic distribution](https://console.cloud.google.com/storage/browser/fair-influence-maximization-mounted/data/Data/Weibo/Output_Full-attempt_2024-03-30_synethic_age?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&hl=en&project=d4w3-369005)
- [Political lean (Republican vs Democrat)](https://console.cloud.google.com/storage/browser/fair-influence-maximization-mounted/data/Data/Weibo/Output_Full-attempt_2024-03-30_synethic_political_position?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&hl=en&project=d4w3-369005)

![image](https://github.com/abhisha1991/fair_at_scale/assets/10823325/15ebb264-919f-4e70-8fb9-f386894c356f)

### Definitions
1. Diffusion: the process of spreading a message in a network
2. Cascade: A collection of (source, target, timestamp) for a specific message (post). It can be thought of as a replay of the network spread of a message
3. Group: graph is divided into groups of people with similar characteristics (age, gender)
4. Maxmin: minimize gap of information spread between groups
   - 70% of old males and 69% of young females influenced with same message
   - Proportions are preserved between groups
5. Equity: any person’s probability of being influenced is (almost) the same, regardless of group, preserving the principle of demographic parity
6. Diffusion Probability P(s, t): Probability that node "t" will be found in cascade started by node "s"
   - So t is influenced by s
7. Fairness is defined as a metric between 0 and 1 for a given characteristic (say gender or region) which is calculated on the basis of influenced ratio values of the graph population which is partitioned by the current characteristic 

Diffusion Probability
![image](https://github.com/abhisha1991/fair_at_scale/assets/10823325/3abaec34-5a22-4da2-8ddf-b0c55403e9ba)

Fairness
![image](https://github.com/abhisha1991/fair_at_scale/assets/10823325/a7b9c497-9d7c-473a-9b58-e539d3e15f21)

![image](https://github.com/abhisha1991/fair_at_scale/assets/10823325/229ea178-03dd-4bb0-a871-7928e9df4580)


## Resources
Main paper: [link](https://arxiv.org/pdf/2306.01587.pdf)
Dataset: [link](https://drive.google.com/file/d/1AFuShgAdyoqodqR1oFlCRp7okEYDdeLt/view)


![image](https://github.com/abhisha1991/fair_at_scale/assets/10823325/9b2a79fd-4d81-411b-8c91-c06d57756ac0)

Team Resources: [link](https://drive.google.com/drive/u/0/folders/1KeuMFnr6hQwNyUvglY103j8hvADN9pzC)
