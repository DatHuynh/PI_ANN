#Implementation of Back analysis of model parameters in geotechnical engineering by means of soft computing

Disclaimer: We are not the authors of the paper. This implementation is only for educational purpose and there is no guarantee on its performance. The brilliant python implementation of neural network belongs to Michal Daniel Dobrzanski (https://github.com/mnielsen/neural-networks-and-deep-learning)

Original paper info:
https://www.researchgate.net/publication/200116978_Back_analysis_of_model_parameters_in_geotechnical_engineering_by_means_of_soft_computing

Project Summary:
In this project, we implement the above paper to estimate the soil parameters in Ho Chi Minh construction site. The soil structure is modeled by Finite Element Method (FEM). Our goal is to estimate the parameter when run through FEM yield the closest result to the on-site measurement.

The estimation contains of 3 main steps:

1.	Approximation FEM by a shallow neural network. The weight of the neural network is determined by Generic Algorithm (50 individuals over 800 generations). In the end, the best neural network is selected as best approximating the input and output pair produced by FEM.

2.	Soil parameters is estimated by propagating gradient from output layer to input layer which is the soil parameter. The initial soil parameter is chosen randomly and is refined through 10000 iterations.

3.	The estimated soil parameters are substitute back to FEM for evaluation. 

    a.	If the errors are larger than threshold, a new data point, which consist of the estimated soil parameters and its measurement through FEM, is added to the training set and step 1 is repeated.
    
    b.	If the errors are within the threshold, then we stop.
    
This github repository only contains step 1 and step 2 since we have problems of integrating FEM into python model. Therefore, FEM is performed manually.

Project directory:

  •	PlayGround: the main file running step 1 and 2.
  
  •	PGA: the Generic Algorithm module
  
  •	WGA: the neural network module written to adapt Generic Algorithm operation
  
  •	PGD: the neural network module propagating gradient to update input soil parameter

 


