# Master Thesis: Study of Deep Learning algorithms: incremental solutions.
Master Thesis presented for the diploma of Computer Engineer (Université Libre de Bruxelles, 2017-2018).

- Name of the student: Antonio García Díaz.
- Full name of the Master cycle: Master en Ingénieur Civil en informatique, à finalité spécialisée.
- Academic year: 2017-2018
- Full name of the Master Thesis: Study of Deep Learning algorithms: incremental solutions.
- Director: Prof. Hugues Bersini
- Department: IRIDIA
- Keywords: Deep Learning, Neural Networks, topology, self-structuring, incremental, TensorFlow.

Abstract:

Deep Learning is a promising subfield of Machine Learning, itself a subfield of Artificial Intelligence. Machine Learning focuses on the development of algorithms which can automatically learn a task and improve on it, through the usage of external data that is relevant to the task. Deep Learning is related to the development of one kind of Machine Learning algorithms, called artificial Neural Networks.

Neural Networks consist of various simple interconnected processors called neurons, organized together in layers. One of the key problems when implementing Neural Networks is to determine their topology: how many layers, and how many neurons per layer, the network should have. In 1994, an algorithm called EMANN was developed in the IRIDIA laboratories at the Université Libre de Bruxelles that allowed for self-structuring Neural Network topologies thanks to an incremental solution. As the network is trained, new layers and neurons are added and/or pruned at different moments, following certain criteria, until an optimal network structure is selected.

In this thesis, new implementations of EMANN called EMANN-like algorithms were created using the TensorFlow library. They were applied to a simple classification task, the CIFAR-10 benchmark dataset, to test them and compare them in terms of accuracy. The effect of certain parameter values on their accuracy was also assessed through up to six groups of tests and experiments. It was mainly concluded that EMANN performs much better when the activation functions used for its neurons behave in a similar way to a sigmoid, and that parameters correlated with an increase in the number of neurons are likely to have a very positive effect on the accuracy of the Neural Network constructed by EMANN.
