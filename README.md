# Welcome to Fair at Scale!

# W210: UC Berkeley
#### Michael Golas, Abhinav Sharma, Emily Robles, Justin Wong
#### Guided by Prof. Puya Vahabi
#### Spring 2024


## Influence Maximization
It refers to a class of algorithms that aim to maximize information spread in a graph under some budget constraints.
The aim is to find K most influential (seed) nodes from which diffusion of a specific message should start.
This project takes things a step further by adding a constraint of fairness when trying to still maximize influence.


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
